# Explicit first/second-field instruction results

Verified on Ascend 910B2, 2026-10-06. This is a follow-up to [the prompt-role investigation](PROMPT_ROLE_INVESTIGATION.md).

The original relevance task is retained. The new explanation identifies document/query roles by field position, regardless of labels. Its order is adapted to the actual content order.

For document-first Touché inputs, the task instruction is:

```text
<Instruct>: Given a question, retrieve detailed and persuasive arguments that answer the question. The first input text field contains the document. The second input text field contains the query. The task instruction is not an input text field. Identify the document and query by field position, regardless of the field labels.
```

The new system instruction is:

```text
Judge whether the document meets the requirements of the query and the task instruction. The first input text field contains the document. The second input text field contains the query. The task instruction is not an input text field. Identify the document and query by field position, regardless of the field labels. Note that the answer can only be "yes" or "no".
```

For query-first inputs, the explanation says the first input field contains the query and the second contains the document. The task instruction is explicitly excluded from the count of input fields, including the layout where it occurs between the document and query.

Each layout is tested unchanged, with the explanation in the task only, in the system only, and in both. Ordinary body tokenization is used for the broad comparison. The initial short examples additionally repeat every layout with separately tokenized labels, contents, and newlines.

## Results on the same 46-query Touché sample

Every cell is correct comparisons out of 478. A tie is not counted as correct.

| Layout | Unchanged | Task explanation | System explanation | Both |
|---|---:|---:|---:|---:|
| Normal query-first | 328 | 287 | 320 | 288 |
| Document-first, correct labels | 237 | 244 | 238 | 246 |
| Document-first, swapped contents | 270 | 265 | 271 | 262 |
| Query-first, swapped labels | 313 | 272 | 291 | 283 |
| Document, instruction, query | 276 | 264 | 260 | 257 |

The position explanation in both places improves correctly labeled document-first inputs from 237 to 246 correct comparisons, a gain of 9/478 (1.9 percentage points). For swapped contents, the strongest tested placement is system-only: 270 to 271. None of the new document-first variants matches the unchanged query-first baseline of 328/478; the strongest document-first result remains the unchanged document/instruction/query layout at 276/478. These results describe this wording and sample, not every possible instruction.

The same 257 explicitly judged documents from the other 46 Touché queries were selected with the unchanged shortest/median-per-grade rule and a 400-document-token eligibility limit. No text was truncated. This is graded pair-ordering accuracy on a selected short-document sample, not full top-100 NDCG@10.

## Ties

| Layout | Unchanged | Task explanation | System explanation | Both |
|---|---:|---:|---:|---:|
| Normal query-first | 13 | 6 | 4 | 9 |
| Document-first, correct labels | 8 | 5 | 8 | 8 |
| Document-first, swapped contents | 8 | 5 | 5 | 9 |
| Query-first, swapped labels | 10 | 9 | 17 | 20 |
| Document, instruction, query | 18 | 8 | 19 | 12 |

## Synthetic sanity tests

The six original synthetic pairs are retained. Each positive-versus-unrelated comparison is tested with every layout and instruction placement. Scores below show the three relevant answers with the explanation in both the system and task.

| Answer | Normal unchanged | Document-first, correct labels | Swapped contents |
|---|---:|---:|---:|
| Paris is the capital of France. | 0.928 | 0.593 | 0.423 |
| Click Forgot password, enter your email, and follow the reset link. | 0.766 | 0.665 | 0.349 |
| Chlorophyll absorbs red and blue light and reflects green light. | 0.955 | 0.392 | 0.101 |

All 20 ordinary variants still rank the relevant answer above the unrelated one in all three synthetic comparisons. All synthetic ordering counts, calibration results, atomic controls, FP32-head comparisons, and grade-specific margins are in summary.json.

## Validation and limits

The maximum absolute score change from previous unmodified baselines is 0.0. Input IDs for matching old layouts are also identical. Original control-token sequences and the assistant suffix were preserved; task-only changes preserve the full original prefix. System-only and combined changes deliberately modify system prose while retaining its chat markers.

Twenty-one short initial pairs are evaluated separately: 15 Touché documents across the initial three queries and six synthetic pairs. The single initial long document is excluded by the same 400-token selection limit, without truncation. Atomic token inventories match within groups sharing identical system/task wording; instructions that assign different first/second roles naturally have different wording and are not asserted to have identical inventories.

Maximum complete input length: 536 tokens. The backbone remains BF16; only the shadow two-row output projection is recomputed in FP32. These are accuracy runs on shared physical NPUs 2 and 3, not latency measurements.

The first attempt hit a shared-device memory limit after saving 115 completed rows. The continuation enabled expandable_segments in the NPU allocator, recomputed an overlap example with all 20 variants, and required an exact score match before reusing rows. It preserved the document selection, prompts, precision, attention implementation, and score formula. The allocator setting is documented by [Ascend](https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/memory_management/PYTORCH_NPU_ALLOC_CONF.md).

Code: investigate_prompt_roles.py --positional-instructions. Source commit: b4e6c4eb6a9277144b6f10c22316d5e26d4bc845.

Evidence: [summary](../tmp/22_qwen3_embedding_benchmark/positional_instructions_b4e6c4eb/summary.json), [exact command](../tmp/22_qwen3_embedding_benchmark/positional_instructions_b4e6c4eb/command.txt), [example prompts](../tmp/22_qwen3_embedding_benchmark/positional_instructions_b4e6c4eb/example_prompts.txt). Full raw token/logit output is saved as result.json.gz with a SHA-256 manifest. analyze.py and write_report.py reproduce the derived files.
