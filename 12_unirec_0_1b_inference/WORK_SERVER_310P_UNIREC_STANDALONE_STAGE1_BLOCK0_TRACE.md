# 310P UniRec stage-1 block-0 operation trace

## Established result

The previous 310P boundary trace found `stage_1_block_0` as the first divergent
boundary. Stage 2 is clean; stage 0 is not useful for localization because it
contains the already-bad stage 1.

This run traces the internal operations of stage-1 block 0. It compiles exactly
**one graph** and does not run the previous stage ladder.

The graph returns these eager-vs-TorchAir checkpoints from the same forward:

- shortcut, first LayerNorm, and the `192 -> 388` focal linear;
- Q/context/gate split outputs;
- Conv and GELU outputs for the 3x3, 5x5, and 7x7 depthwise paths;
- every gated accumulation and global-context output;
- 1x1 modulator, Q multiplication, and modulation projection;
- residual, second LayerNorm, MLP FC1/GELU/FC2, and block output.

Returning internal tensors can change fusion. A clean 310P result would mean
the prior fault depends on a fusion inhibited by the extra outputs. A divergent
result identifies the first corrupt operation boundary.

Verified 910B2 control at commit `b50777c`, physical NPU 7:

- one cold compiled call: 21.116 s;
- steady compiled forward: 10.535 ms;
- all focal/modulation checkpoints through the residual were bit-exact;
- the first normal FP16 difference appeared at the second LayerNorm
  (`max_abs=0.00390625`, cosine `0.99999994`);
- final block output remained clean (`max_abs=0.0009765625`, cosine
  `0.99999976`);
- no divergent boundary.

## Preconditions

Use the validated virtual-environment real binary (`python_nosym`). Do not use
`readlink -f` on a venv symlink if it escapes the venv. Select one free physical
310P device from 0-3. Do not edit tracked files on the work server.

```bash
cd "$(git rev-parse --show-toplevel)"
git pull --ff-only
export PYTHON_BIN=/absolute/path/to/the/validated/venv/bin/python_nosym
export ASCEND_RT_VISIBLE_DEVICES=0  # replace with the selected free 310P 0-3
"$PYTHON_BIN" -c 'import sys, torch, torch_npu; print(sys.executable, torch.__version__, torch_npu.__version__)'
npu-smi info
```

## Run in background

```bash
bash 12_unirec_0_1b_inference/run_310p_standalone_vision_stage1_block0_trace_background.sh
```

The launcher prints absolute `RUN_ROOT`, `RUN_LOG`, and PID. Show `RUN_LOG` to
Luka immediately, then follow it:

```bash
tail -f /absolute/RUN_LOG/from/the/launcher
```

Expected work is exactly one graph. The log marks `graph_registered`,
`compiled_first_begin`, and `compiled_first`. If a phase is unchanged for 30
seconds, report the phase, elapsed time, cache directory, and `npu-smi info`.
Do not delete the cache or launch another lane.

## Required report

Paste:

```text
UNIREC_STANDALONE_BLOCK0_TRACE_BEGIN
UNIREC_STANDALONE_VISION_PHASE ... graph_registered
UNIREC_STANDALONE_VISION_PHASE ... compiled_first_begin
UNIREC_STANDALONE_VISION_PHASE ... compiled_first
UNIREC_STANDALONE_BLOCK0_TRACE_BOUNDARY ...
UNIREC_STANDALONE_BLOCK0_TRACE_SUMMARY ...
UNIREC_STANDALONE_BLOCK0_TRACE_END
```

Also report the absolute run paths, exit code, process wall time, cache inventory
change, exact first divergent boundary, and any unexpected OM or
`compiled_module` creation.
