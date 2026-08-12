# 310P UniRec remaining TransData pass

Pull `main` at or after `36855ab`. Do not edit tracked files. Use the same FP16
TorchAir layout command, fixed 128 pages, physical NPU exclusion, and warm-cache
procedure that produced the valid 36 ms `group16` result.

Run only these incremental lanes with fresh caches:

1. `strict_buffers`: `--depthwise-rewrite group16 --weight-format native
   --preformat-frozen-bn-buffers`
2. `internal`: `--depthwise-rewrite group16 --weight-format torchair_internal`
3. `internal_buffers`: `--depthwise-rewrite group16 --weight-format
   torchair_internal --preformat-frozen-bn-buffers`

First run 8 pages, then 128 only after the gate passes. Compare every full
`result_digest`:

- `strict_buffers` must equal the original native `group16` result.
- `internal_buffers` must equal `internal`; also report whether either equals
  native `group16`.

Do not run `--fuse-frozen-bn`, `--fuse-eval-bn`, or
`--precompute-frozen-bn-affine`. On 910B those changed real detector semantics.

Profile warmed `strict_buffers`. Profile `internal_buffers` only if it equals
`internal` on all 128 pages. Use the committed layout-only profiler with the
matching `--layout-*` flags. Compilation, setup, preprocessing, and postprocess
are excluded.

Return exactly:

```text
Forward ms: group16=36.00; strict_buffers=<mean>; internal=<mean>; internal_buffers=<mean>.
Digest matches: strict/group16=<n>/128; internal/group16=<n>/128; internal_buffers/internal=<n>/128.
Strict profile: TransData=<ms/count>; Conv2D=<ms>; top signatures=<top 10 compactly>.
Internal-buffer profile: <not run, or TransData/Conv2D/top 10 signatures>.
```

910B structural evidence only, not a 310P prediction:

- native `group16`: TransData 2.397 ms / 707 calls;
- strict buffers: 2.041 ms / 387 calls, 8/8 native digests exact;
- internal: 1.266 ms / 604 calls;
- internal buffers: 0.880 ms / 284 calls and 8/8 outputs equal internal;
- the output difference versus native came from internal-weight formatting,
  not from direct buffer preformatting.
