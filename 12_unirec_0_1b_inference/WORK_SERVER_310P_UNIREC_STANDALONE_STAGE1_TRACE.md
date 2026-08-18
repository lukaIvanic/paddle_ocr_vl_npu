# 310P UniRec standalone stage-1 boundary trace

## Purpose

The synthetic standalone probe already established:

- start stage 2: eager and TorchAir agree;
- start stage 1: eager and TorchAir diverge;
- start stage 0: also diverges, but this adds no localization because its graph
  contains stage 1.

This run compiles **one graph only**. The same compiled stage-1 suffix returns
intermediate tensors after every block/downsample plus final projection. The
report identifies the first eager-vs-TorchAir boundary that diverges.

Do not run the old stage ladder again. Do not delete any cache. Do not change
tracked files on the work server.

## Preconditions

Use the validated virtual-environment real binary (`python_nosym`), not a
`readlink -f` result that escapes the venv. Select one free physical 310P device
from 0-3. The runner rejects devices 5 and 6 and requires one visible device.

```bash
cd "$(git rev-parse --show-toplevel)"
git pull --ff-only
export PYTHON_BIN=/absolute/path/to/the/validated/venv/bin/python_nosym
export ASCEND_RT_VISIBLE_DEVICES=0  # replace with the selected free 310P 0-3
```

Before starting, verify the interpreter and NPU:

```bash
"$PYTHON_BIN" -c 'import sys, torch, torch_npu; print(sys.executable, torch.__version__, torch_npu.__version__)'
npu-smi info
```

## Start in background

```bash
bash 12_unirec_0_1b_inference/run_310p_standalone_vision_stage1_trace_background.sh
```

The launcher prints an absolute `RUN_LOG`. Show it to Luka immediately. Follow
it with:

```bash
tail -f /absolute/RUN_LOG/from/the/launcher
```

Expected work: exactly **one** TorchAir graph. The log prints:

- `expected_graphs=1` at start;
- `graph_registered` immediately before the first compiled call;
- `compiled_first_begin` immediately before cache load or compilation;
- `compiled_first` when that work ends;
- one result line for every boundary.

If no new phase line appears for 30 seconds, report the current phase, elapsed
wall time, cache directory, and current `npu-smi info`. Do not start another
lane and do not delete the cache.

## Required report

Paste these lines from `run.log`:

```text
UNIREC_STANDALONE_STAGE1_TRACE_BEGIN
UNIREC_STANDALONE_VISION_PHASE ... graph_registered
UNIREC_STANDALONE_VISION_PHASE ... compiled_first_begin
UNIREC_STANDALONE_VISION_PHASE ... compiled_first
UNIREC_STANDALONE_STAGE1_TRACE_BOUNDARY ...
UNIREC_STANDALONE_STAGE1_TRACE_SUMMARY ...
UNIREC_STANDALONE_STAGE1_TRACE_END
```

Also report:

- absolute `RUN_ROOT` and `RUN_LOG`;
- exit code and process wall seconds;
- whether cache inventory changed;
- the exact first divergent boundary;
- whether any unexpected extra OM or `compiled_module` appeared.

Do not summarize only the final projection. The first divergent boundary is the
point of this run.
