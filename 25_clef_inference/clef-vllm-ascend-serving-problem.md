# Serving our own Clef implementation through vLLM-Ascend

## Problem statement

We have a local PyTorch implementation of Clef-flash running on an Ascend NPU. We want to serve it through vLLM-Ascend while retaining ownership of the **complete model implementation**, including attention, recurrence, intermediate tensor layouts, buffers, and the decision head.

We do not yet know the smallest correct integration boundary that permits this. Registering a custom model is only part of the problem: vLLM-Ascend's runner, scheduler, initialization, memory management, and request handling may assume execution conventions that our implementation does not follow.

The question is:

> How can we adapt vLLM-Ascend to serve our full Clef implementation, with our own attention and storage layouts and without paged attention, while preserving correct request and response behavior and making the model easy to modify?

## Current model and intended workflow

Clef processes a state together with a schema of questions and answer options. Our current text-only implementation runs the backbone over the complete input sequence, then applies a joint decision head to produce option scores and structured answers. This is a single-forward decision workload, rather than autoregressive text generation.

The head uses sequence hidden states, question and option spans, question types, and token representations derived from the output embedding matrix. Requests can contain different numbers of questions and options.

Our local modeling file is intended to remain the main place for development. We want to edit layer computations, replace attention or recurrent operations, change layouts, and eventually optimize kernels without having to express every experiment through vLLM's existing model abstractions.

Reusing vLLM-Ascend's Qwen3.5 backbone is therefore not the integration objective. We need to bring **our implementation** into its serving infrastructure.

## Requirements

- Keep the full modeling implementation accessible and editable in our source tree.
- Use our own attention and recurrent computation; do not require vLLM's built-in attention operators.
- Do not use paged attention. Our model must be able to use contiguous tensors or other layouts that we choose.
- Preserve causal attention semantics inside the model, independently of whether vLLM manages any cache.
- Preserve input encoding, question/option boundaries, scoring, and answer formatting.
- Support the intended request payload and response schema, including validation, errors, request identity, and completion behavior. The exact HTTP contract still needs to be specified.
- Keep serving-specific adaptation separate enough that routine model changes do not require edits throughout the engine.
- Initially establish correctness with one NPU, BF16, one request at a time, and eager execution. Batching and other execution optimizations are subsequent work.

## Desired ownership boundary

```text
HTTP request
    |
Validation, encoding, request queue
    |
Serving adapter
  - complete request and sequence boundaries
  - token IDs and question/option metadata
  - conversion into our chosen input layout
    |
Our complete Clef implementation
  - attention and recurrence
  - intermediate layouts and temporary buffers
  - joint decision head
    |
Result formatting and request completion
    |
HTTP response
```

The uncertainty is how much of the serving adapter can use existing extension interfaces, and how much requires a custom runner, worker, or engine patch.

## Integration questions to resolve

| Area | Unresolved question |
|---|---|
| Scheduling | Can the engine guarantee that a complete sequence reaches our model in one execution, without partial prefill or decode assumptions? |
| Request boundaries | Where can the model obtain reliable lengths and boundaries when the runner packs tokens from multiple requests? |
| Metadata | How do question spans, option spans, types, and token IDs reach the correct request's head computation? |
| Layout | Can a narrow adapter unpack or reshape inputs while leaving all internal model layouts under our control? |
| Cache management | Can we use no vLLM-managed KV or recurrent cache throughout initialization and execution? |
| Memory admission | How should the engine account for our activations and scratch buffers when KV-cache capacity is not the relevant limit? |
| Initialization | Can loading, memory profiling, dummy execution, and warmup work without real schema metadata or registered attention backends? |
| Outputs | How do variable-length question/option scores become engine-compatible results without imposing inappropriate fixed-class or text-generation semantics? |
| Lifecycle | How are cancellation, errors, completion, and request-local state handled without leaking state between requests? |
| Endpoints | Which interface preserves our intended payload and response contract, and what requires an HTTP adapter? |

Disabling paged attention does not eliminate memory constraints. Full-sequence attention and other temporary tensors may make memory requirements strongly dependent on sequence length and batch composition.

## Candidate approaches, not yet validated

### Custom model with the existing pooling runner

Register our own model and use the pooling execution path to finish a request after one forward pass. Here, pooling is a serving interface; it does not mean we must average hidden states or use a particular attention implementation.

This could be the smallest integration if the existing runner exposes adequate request boundaries and metadata and supports our cache-free execution and initialization requirements.

### Dedicated runner or worker adaptation

If those interfaces are insufficient, own the execution adapter as well: assemble complete requests, invoke our model, and return engine-compatible results.

This gives more control but increases maintenance against vLLM-Ascend internals. Scheduler or engine changes should follow demonstrated incompatibilities rather than be assumed necessary in advance.

### I/O processor or endpoint adapter

Use an I/O processor or a thin HTTP adapter to translate the external contract into engine requests and reconstruct responses. This addresses payload handling; it does not solve worker execution, scheduling, or memory assumptions.

## Evidence and useful development references

The inspected server installation uses vLLM `0.21.0+empty` at `ad7125a431e176d4161099480a66f0169609a690` and vLLM-Ascend `0.21.0rc1` at `80610e4438dba05011b05f89fc45d91e96992671`. Both are editable source installations. Findings are tied to those revisions unless stated otherwise.

- The [Ascend model runner](https://github.com/vllm-project/vllm-ascend/blob/80610e4438dba05011b05f89fc45d91e96992671/vllm_ascend/worker/model_runner_v1.py) contains a zero-KV-cache-group path that skips attention-metadata construction and a pooling path after model execution. These are promising hooks, not proof that our complete execution path works.
- [Pooling parameters](https://github.com/vllm-project/vllm/blob/ad7125a431e176d4161099480a66f0169609a690/vllm/pooling_params.py) and [worker pooling metadata](https://github.com/vllm-project/vllm/blob/ad7125a431e176d4161099480a66f0169609a690/vllm/v1/pool/metadata.py) provide possible transport for request-specific data. Its use for Clef remains unvalidated.
- [DeepSeek-OCR2 Ascend PR #7737](https://github.com/vllm-project/vllm-ascend/pull/7737) is a useful anchor for substantial custom module development. Its [custom decoder](https://github.com/vllm-project/vllm-ascend/blob/80610e4438dba05011b05f89fc45d91e96992671/vllm_ascend/ops/qwen2_decoder.py) and [relative-position attention](https://github.com/vllm-project/vllm-ascend/blob/80610e4438dba05011b05f89fc45d91e96992671/vllm_ascend/ops/rel_pos_attention.py) demonstrate replacing significant computations. They do not establish a solution for our top-level scheduling and cache requirements.
- The [Ascend patch guide](https://github.com/vllm-project/vllm-ascend/blob/80610e4438dba05011b05f89fc45d91e96992671/docs/source/developer_guide/Design_Documents/patch.md) distinguishes main-process and worker patches. Any integration must account for where code actually executes across processes.

vLLM itself depends on Transformers for infrastructure such as configuration. Keeping our native modeling implementation does not imply that the serving process can prohibit every Transformers import.

## What would resolve the uncertainty

Before committing to an implementation, trace one request through validation, enqueueing, scheduling, worker input preparation, model execution, result extraction, and response completion. At each boundary, identify the required data and the applicable extension hook or specific missing capability.

A successful integration would demonstrate that:

1. A real request runs our complete modeling code without using vLLM attention operators or paged attention.
2. Input encoding, scores, and answers agree with the standalone reference under the same computation and precision; any numerical differences are measured and explained.
3. Different sequence lengths and question/option counts preserve request boundaries and produce correctly associated responses.
4. Loading, profiling, errors, and cancellation do not rely on missing cache state or real-request metadata during dummy runs.
5. Memory limits and admission behavior are explicit before concurrency is increased.
6. Editing our modeling file remains a practical development loop without widespread engine changes.

This document describes the integration problem and candidate boundaries. No port or optimization is assumed to have been implemented or validated.
