# Work-server 310P native-GQA PromptFA probe

This is a pull-only Atlas 310P validation brief. Do not edit tracked files,
commit, push, create a branch, install packages, or change the NPU stack.

## Objective

Test one real Qwen3-Reranker-0.6B shape with two otherwise matched PromptFA
lanes:

1. `expanded_gqa`: the current path, which repeats K/V from KV heads to query
   heads before PromptFA.
2. `native_gqa`: compact K/V plus PromptFA `num_key_value_heads`.

The probe loads the complete real checkpoint once. It builds layer-0 Q/K/V from
real token embeddings, input norm, projection weights, and RoPE. It compares
both PromptFA outputs with manual attention, compiles each static attention
core with `cache_compile`, and measures two warmups plus ten short post-warmup
calls. The default shape is B4, S128, BSND, FP16, square causal bool mask,
`sparse_mode=0`, and no actual-sequence-length arguments.

## Run

Resolve and update the existing checkout. Stop if tracked changes prevent the
pull. Do not discard them.

```bash
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git rev-parse HEAD
source npu-setup
export PYTHON_BIN="$(command -v python3)"
export MODEL_06B_DIR=/absolute/path/to/Qwen3-Reranker-0.6B
export DEVICE=npu:0
bash 13_qwen3_reranker/run_310p_promptfa_native_gqa.sh
```

Use the work server's established environment setup if it is not named
`npu-setup`. Select one free 310P through the normal server mechanism. Do not
terminate another process. Do not download the model.

## Report immediately

The script prints progress before each lane. Paste these items back as soon as
the command ends:

1. `git rev-parse HEAD`, hostname, Python path, Torch version, torch-npu
   version, device name, and `ASCEND_RT_VISIBLE_DEVICES`.
2. Both complete `LANE_RESULT` lines.
3. The complete `NATIVE_GQA_PROMPTFA_PROBE` line.
4. `PROBE_END`, including the output root.
5. For any failure, the first causal error from the lane's embedded traceback.

A native-lane failure is a useful compatibility result. Do not route around it.
If both lanes pass, compare `native_vs_expanded_speedup` and require native
`compiled_vs_expanded.allclose_atol_5e_2_rtol_5e_2=true` before considering the
native route usable. Compile/first-call time is not throughput; use only each
lane's `post_warmup_s.median` and `attention_query_tok_s`.
