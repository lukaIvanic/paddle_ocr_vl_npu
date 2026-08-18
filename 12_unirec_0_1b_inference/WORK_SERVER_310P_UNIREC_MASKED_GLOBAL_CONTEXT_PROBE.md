# 310P UniRec ambiguous global-context shape probe

## Finding being tested

The full stage-1 trace stayed near parity through `accum_7x7`
(`max_abs=0.0019`) and then jumped to `max_abs=819` at
`global_context_pre_gelu`.

That first bad tensor has shape `[1,192,1,1]`. This is an ambiguous 4D format
family on Ascend. The problem is now suspected to be TorchAir/GE format handling
for the reduced channel vector, not a convolution weight or depthwise operation.

This standalone script has no model or project import. It compiles exactly one
small graph and compares:

1. the current two-stage reduction producing `[1,192,1,1]`;
2. a direct masked 2D reduction still producing `[1,192,1,1]`;
3. the same direct reduction kept as unambiguous `[1,192]`;
4. the current 4D GELU+broadcast result;
5. the 2D GELU result expanded only at the final broadcast.

Do not rerun any vision graph or previous ladder.

Verified 910B2 control at commit `c4f6173`, physical NPU 7:

- one cold compiled call: 11.769 s;
- steady compiled graph: 0.401 ms;
- all eager-vs-compiled lanes: bit-exact;
- eager current 4D vs direct 2D pre-GELU: max-abs `3.8147e-6`;
- eager current broadcast vs flat broadcast: max-abs `0.0004883`, cosine 1.0.

## Preconditions

Use the validated venv `python_nosym` real binary. Do not resolve a venv symlink
to the system Python. Select one free physical 310P device from 0-3. Do not edit
tracked files on the work server.

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
bash 12_unirec_0_1b_inference/run_310p_masked_global_context_probe_background.sh
```

The launcher prints absolute `RUN_ROOT`, `RUN_LOG`, and PID. Show `RUN_LOG` to
Luka immediately and follow it with `tail -f`. Expected work is exactly one
small graph. The log marks `graph_registered`, `compiled_first_begin`, and
`compiled_first` so a delay is attributable.

If a phase is unchanged for 30 seconds, report that phase, elapsed wall time,
cache directory, and `npu-smi info`. Do not delete the cache or launch another
lane.

## Required report

Paste every line beginning with:

```text
UNIREC_GLOBAL_CONTEXT_RUN_
UNIREC_GLOBAL_CONTEXT_PHASE
UNIREC_GLOBAL_CONTEXT_COMPARISON
UNIREC_GLOBAL_CONTEXT_RESULT
```

Also report absolute run paths, exit code, process wall time, cache inventory
change, and whether exactly one OM and one `compiled_module` were created.

The key comparison is whether `original_eager_vs_compiled` or
`direct_4d_eager_vs_compiled` fails while `direct_2d_eager_vs_compiled` and
`flat_broadcast_eager_vs_compiled` remain clean.
