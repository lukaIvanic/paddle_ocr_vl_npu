# Qwen3-Reranker-4B prompt-role investigation

Verified on Ascend 910B2, 2026-10-06. No training or cache implementation.

Changing field meanings and instructions strongly changes the yes/no decision.
Content order also has a substantial effect on ranking quality. None of the
tested document-first instruction edits recovered the normal prompt's ranking
quality on the previously unused Touché queries.

## Data and method

The initial calibration set is the previous 22-pair probe: 16 explicitly judged
Touché documents across query IDs 1, 9, 33, plus three synthetic questions with
one relevant and one unrelated document each. Its normal scores reproduced
exactly, with maximum absolute difference 0.

The broader evaluation excludes those three Touché queries. For each of the
remaining 46 queries and each grade 0/1/2, select the shortest and median-length
eligible candidate, counting duplicates only once. Eligibility requires an
explicit benchmark judgment and at most 400 document tokens. This gives 257
documents and 478 comparisons between documents with different grades belonging
to the same query. No input is truncated. Maximum full input length across the
main held-out variants is 466 tokens.

A correct comparison requires the higher-grade document to score strictly
higher. Ties are recorded separately and do not count as correct. This is a
selected-document diagnostic, not NDCG@10 on the original top-100 candidates.
Comparisons share documents and queries. The short-document selection limits
generalization to the original benchmark and longer documents.

Runtime: Qwen3-Reranker-4B, BF16, B1, Transformers 5.5.4, torch 2.10.0+cpu,
torch_npu 2.10.0.post2, SDPA, use_cache=False, logits_to_keep=1. Scoring uses the
official lowercase yes/no token IDs, 9693 and 2152:

```text
margin = yes_logit - no_logit
score = sigmoid(margin)
```

The model remains BF16. A shadow calculation applies only the final two-row
output projection in FP32 to the same BF16-derived hidden state. This checks
final projection/output rounding; it is not a full FP32 model comparison.

## Independent order and label controls

Every body begins with the original `<Instruct>:` field. The four core layouts
are:

| Variant | First content field | Second content field |
|---|---|---|
| normal | `<Query>: question` | `<Document>: passage` |
| document_first | `<Document>: passage` | `<Query>: question` |
| swapped_contents | `<Query>: passage` | `<Document>: question` |
| query_first_wrong_labels | `<Document>: question` | `<Query>: passage` |

For calibration, each has an additional `atomic` variant: tokenize the labels,
leading-space content strings, and separating newlines independently, then
concatenate their IDs. All four atomic variants have exactly the same token-ID
multiset, including identical prefix and suffix positions, while their field
order/label assignment differs. These atomic controls are a different
tokenization protocol from normal body tokenization and have their own baseline.

All explicit `<|im_start|>`, `<|im_end|>`, `<think>`, and `</think>` tokens were
audited, including tokens omitted from tokenizer.all_special_ids. Their ordered
sequence is identical across every variant. The suffix is unchanged throughout.
The prefix is identical except for the deliberately remapped system sentence.
The tokenizer has no BOS token; no original framing token was removed.

## Instruction interventions

The original Touché instruction is:

```text
Given a question, retrieve detailed and persuasive arguments that answer the question
```

`normal_clarified` and `document_first_clarified` append:

```text
. The question is in <Query>; the passage is in <Document>.
```

`swapped_contents_remapped` appends instead:

```text
. The question is in <Document>; the passage is in <Query>.
```

`swapped_contents_reverse_task` replaces the Touché instruction with:

```text
Given an argument, retrieve a question that the argument addresses with detailed and persuasive reasoning
```

Its synthetic counterpart uses:

```text
Given a passage, retrieve a question that the passage answers
```

The two `general_task` variants use:

```text
Given a web search query, retrieve relevant passages that answer the query
```

`swapped_contents_system_remapped` changes the system sentence to judge whether
the Query meets the requirements based on the Document and Instruct. Everything
else in that prefix, including its framing and yes/no requirement, remains.

The follow-up `swapped_contents_joint_remap` combines this system change with
the explicit task-field remapping. It evaluates the same 257 held-out documents
plus 21 short calibration pairs; these partitions are recorded separately. One
long initial Touché pair is excluded from this follow-up, without truncation.

## Ranking results

Main evaluation on the 46 previously unused Touché queries:

| Variant | Correct / 478 | Correct fraction | Ties |
|---|---:|---:|---:|
| normal | 328 | 68.62% | 13 |
| document_first | 237 | 49.58% | 8 |
| swapped_contents | 270 | 56.49% | 8 |
| query_first_wrong_labels | 313 | 65.48% | 10 |
| normal_clarified | 318 | 66.53% | 6 |
| document_first_clarified | 229 | 47.91% | 13 |
| swapped_contents_remapped | 260 | 54.39% | 9 |
| swapped_contents_reverse_task | 271 | 56.69% | 10 |
| normal_general_task | 282 | 59.00% | 7 |
| swapped_contents_general_task | 272 | 56.90% | 4 |
| swapped_contents_system_remapped | 277 | 57.95% | 13 |
| swapped_contents_joint_remap, follow-up | 265 | 55.44% | 6 |

The follow-up normal baseline again gives 328/478, with exactly the same scores
as the main held-out run. The 20 completed pairs from an earlier failed NPU-6
attempt also have identical scores to their successful NPU-1 counterparts for
every variant.

Keeping the question text first preserves much more accuracy even with wrong
labels. Reordering content hurts with either label assignment. Labels also
matter: when the passage comes first, retaining Query then Document labels
outperforms the semantically correct Document then Query layout.

Query-cluster bootstrap diagnostics are in query_metrics.json. For this selected
sample, the macro comparison-accuracy differences from normal are approximately
-19.7 percentage points for document_first, -12.0 for swapped_contents, and
-12.7 for joint_remap; their descriptive 95% query-bootstrap intervals are
[-26.5, -13.4], [-18.7, -5.4], and [-17.9, -7.4] percentage points. These are not
confidence intervals for full-benchmark NDCG or unseen datasets.

## What causes the apparent score inversion?

### Task wording can change the direction of the shift

On the exact same 257 documents:

| Instruction | Mean normal score | Mean swapped-content score |
|---|---:|---:|
| Original Touché argument instruction | 0.383 | 0.541 |
| Generic web-search instruction | 0.728 | 0.665 |

The average swapped-minus-normal logit margin changes from +0.815 with the
original task to -0.336 with the generic task. This is a controlled instruction
effect; it does not depend on comparing synthetic text with different Touché
text. The generic instruction also reduces normal ranking correctness from
328 to 282 comparisons, so changing the score shift does not recover the
original benchmark judgment.

With the original instruction, swapped contents raise the average margin by
+0.978 for grade 0, +0.964 for grade 1, and +0.511 for grade 2. Lower-grade
documents are boosted more, reducing their separation from stronger documents.

For the three synthetic relevant answers:

| Answer | Normal | Swapped contents | Swapped + reverse task |
|---|---:|---:|---:|
| Paris is France's capital | 0.928 | 0.469 | 0.963 |
| Password-reset steps | 0.766 | 0.349 | 0.965 |
| Chlorophyll explanation | 0.955 | 0.156 | 0.844 |

Reversing the retrieval task matches a passage-to-question pairing and recovers
these confidence scores. For Touché, that also changes the scoring criterion:
finding a question addressed by an argument differs from assessing how detailed
and persuasive the argument is as an answer. This is a plausible explanation
for why confidence recovery does not restore graded argument rankings. It is
an interpretation of the interventions, not proof of an internal mechanism.

### Causal order changes what information each token can use

The exact installed Qwen3 implementation creates a causal attention mask and
passes it through every decoder layer; its relevant code is preserved in
runtime_attention_contract.txt.

With question first, document tokens can attend to the question, so their
representations can become question-dependent throughout the decoder. With
document first, document representations cannot use the later question; only
the later question and assistant-suffix tokens can combine the two. This is
also the property that allows static document caching.

The order/label experiment is consistent with this difference contributing to
the accuracy loss. It does not establish how much of the loss comes from causal
information flow, learned positional/format preferences, or other internal
model behavior. We have not measured attention attribution or intervened on
the hidden states to isolate those mechanisms.

### Token boundaries matter, but do not explain the large shifts alone

The normal swap changes tokens such as `?\n` versus `?`, and `.` versus `.\n`.
The atomic controls eliminate those token-inventory differences. The calibration
Touché swapped-content margin shift, relative to its atomic baseline, remains
+0.836, versus +0.715 for normal body tokenization. The synthetic relevant-answer
margin shift is -3.083 with either protocol. Thus the large confidence effects
survive unchanged token inventories and preserved framing.

Boundary tokenization still affects rankings: the initial document_first control
changes from 14/28 to 18/28 correct, while its normal baseline changes from
21/28 to 20/28. It should remain controlled in subsequent prompt comparisons.

### The logit change is much larger than final-head rounding

For the chlorophyll answer:

| Layout | Yes logit | No logit | Margin | Score |
|---|---:|---:|---:|---:|
| normal | 18.125 | 15.0625 | +3.0625 | 0.955 |
| swapped_contents | 14.6875 | 16.375 | -1.6875 | 0.156 |

Both logits move against the answer, giving a -4.75 margin change. The FP32-head
shadow retains this reversal. Across the main held-out run, the mean absolute
BF16-versus-FP32-head margin difference is 0.034, with maximum 0.119. FP32-head
rank counts remain separated: normal 333/478, document_first 241/478,
swapped_contents 272/478. Rounding creates some ties but does not account for
the observed accuracy gap.

The official score normalizes only the lowercase yes/no logits. Full-vocabulary
yes+no probability mass in the normal held-out prompts ranges from 0.084 to
0.638; uppercase Yes/No and other tokens also receive probability. A displayed
score of 0.9 therefore does not mean a 90% full-vocabulary probability of
generating lowercase yes. The scoring formula itself remains fixed in every
comparison.

## Evidence and reproducibility

Code: investigate_prompt_roles.py. Main calibration source commit abfb14e4;
successful held-out source commit 2995437b; joint follow-up source commit
8318a904. Exact commands, exit codes, logs, summaries, token IDs, final logits,
top-five vocabulary tokens, and full-vocabulary yes/no probabilities are saved.
Raw JSON outputs are committed as gzip files with SHA-256 manifests; their
uncompressed copies also remain in the authoring workspace.

- [Calibration and token controls](../tmp/22_qwen3_embedding_benchmark/role_investigation_abfb14e4/summary.json)
- [Held-out evaluation](../tmp/22_qwen3_embedding_benchmark/role_holdout_npu1_2995437b/summary.json)
- [Per-query statistics](../tmp/22_qwen3_embedding_benchmark/role_holdout_npu1_2995437b/query_metrics.json)
- [Joint remapping, separate partitions](../tmp/22_qwen3_embedding_benchmark/role_joint_8318a904/summary.json)

Calibration ran on physical NPU 6. The first held-out attempt (900-token
selection limit) and a 400-token retry on NPU 6 hit OOM as a resident shared
workload's memory allocation increased. Their failed logs and partial output
are preserved. The 400-token run and joint follow-up completed on physical
NPU 1, sharing an existing worker without terminating or modifying it. These
are accuracy probes, not serving-latency measurements. All results are 910B2;
there is no 310P or CUDA validation claim.

The result narrows the problem: instruction semantics, field formatting, and
causal content order all affect behavior. Simple role explanations and tested
task reversals do not provide an accuracy-preserving document-first prompt yet.

The subsequent [explicit first/second-field instruction rerun](POSITIONAL_INSTRUCTION_RESULTS.md)
tests five layouts with the position explanation in the task, system, or both,
using the same 46-query sample and initial short examples.
