# Work-server 310P Qwen3 reranker W8A8 handoff

This is a pull-only Atlas 310P validation brief. Do not edit tracked files,
commit, push, create a branch, install packages, or change the NPU software
stack. Run the committed matrix and report its output.

## Objective

Test the true W8A8 route on 310P in this order:

1. Compile and execute `npu_quantize` plus two `npu_quant_matmul` calls in
   isolation at the real 0.6B and 4B gate/up shapes.
2. Run matched static compiled dense and gate/up-W8A8 prefix-prefill benchmarks
   on Qwen3-Reranker-0.6B and Qwen3-Reranker-4B.
3. Report median latency, executed tok/s, speedup, weight-quantization time,
   numerical differences, compilation failures, and the exact device identity.

Accuracy is not a gate for this experiment. A score change is expected. The
operator must run, the compiled graph must complete, and the throughput numbers
must come from post-warmup calls.

## Rules

- Start from the existing checkout and resolve it with `git rev-parse`.
- Run `git pull --ff-only origin main`. If tracked changes prevent the pull,
  stop and report them. Do not discard them.
- Use the server's existing NPU environment and Python. Disable NPU JIT compile;
  the committed scripts do this themselves.
- Do not use TorchAir compilation beyond the committed static `cache_compile`
  calls. Do not use `torch.compile`, CPU fallback, or TorchAir's normal compile
  backend.
- Select one free 310P through the server's normal environment setup. Do not
  terminate another process.
- Use local model directories. Do not download models during this test.
- Keep the generated run directory. It contains commands, logs, exit codes,
  graph caches, result JSON files, and the final summary.

## Bootstrap

From the repository checkout:

```bash
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git rev-parse HEAD
```

Activate the work server's established NPU environment. If `npu-setup` is the
documented setup command on that server, use:

```bash
source npu-setup
```

Resolve the Python executable and both existing models. Example only:

```bash
export PYTHON_BIN="$(command -v python3)"
export MODEL_06B_DIR=/absolute/path/to/Qwen3-Reranker-0.6B
export MODEL_4B_DIR=/absolute/path/to/Qwen3-Reranker-4B
export DEVICE=npu:0
```

Before running, verify that both model directories contain `config.json`, the
tokenizer files, and all checkpoint shards. If either model is absent, stop and
report the missing model and the locations checked.

## One-command matrix

```bash
bash 13_qwen3_reranker/run_310p_w8a8_matrix.sh 2>&1 | tee /tmp/310p_w8a8_driver.log
```

Defaults are chosen to expose useful 310P potential without a broad sweep:

- isolated operator probes: `M=512` at each model's real `K` and `N`;
- 0.6B model: batch 16, continuation 128;
- 4B model: batch 4, continuation 128;
- prefix: physical 128-token cached block;
- continuation attention: real Q128, square-padded physical Q/KV256;
- PromptFA preset: `combined_bsnd`;
- dense and INT8 weights: one-time FRACTAL_NZ preparation;
- three warmup calls and twenty measured calls per graph.

The script continues after a failed phase so one unsupported shape does not
hide the other results. It returns nonzero if any phase failed.

If and only if the 4B B4 model phase fails with a clear NPU out-of-memory error,
rerun the matrix once with:

```bash
BATCH_4B=1 \
OUTPUT_ROOT="$REPO/tmp/13_qwen3_reranker/310p_w8a8_4b_b1_fallback_$(git rev-parse --short=12 HEAD)" \
bash 13_qwen3_reranker/run_310p_w8a8_matrix.sh 2>&1 | tee /tmp/310p_w8a8_4b_b1_driver.log
```

Do not use the B1 fallback for an operator-contract, compiler, or unsupported
format error.

## Report back

Paste these items without paraphrasing away failures:

1. `git rev-parse HEAD`, hostname, Python path, Torch version, torch-npu version,
   device name, and `ASCEND_RT_VISIBLE_DEVICES`.
2. Every `PHASE_END ... exit_code=...` line.
3. Both complete `W8A8_OP_PROBE` lines, or the first causal traceback for each
   failed probe.
4. Every `.THROUGHPUT` line from the four model phases.
5. The complete `W8A8_310P_MATRIX_SUMMARY` line.
6. `OUTPUT_ROOT` and the contents of any nonzero phase's `command.txt`,
   `exit_code.txt`, and first causal error from `run.log`.

Do not claim a speedup from first-call or compile time. The summary compares
the post-warmup median calls only.
