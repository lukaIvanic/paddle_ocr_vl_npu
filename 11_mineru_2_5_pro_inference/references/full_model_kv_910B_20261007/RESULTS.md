# Full-model MinerU KV benchmark on 910B2

The performance replacement is `bench_full_model_kv.py`, not the synthetic
isolated eager `kv_cache_probe/probe_attention.py`. The benchmark loads actual
FP16 MinerU2.5-Pro-2605-1.2B weights, reads distinct real OmniDocBench crops,
runs the complete vision/text prefill, then measures complete static TorchAir
decoder graphs with advancing real sampled tokens and KV. Prefill is outside
the decode timer and recorded separately. No end-to-end page rate is claimed.

Every measured forward contains all 24 text layers, KV writes, attention,
projections/MLPs, final norm, LM head and greedy sampling. The synchronized wall
interval includes CPU token copies and host EOS checks. Useful decode tokens
exclude the first prefill token, EOS and work on completed slots. Cold compile,
warmup, resetting/converting KV, fidelity checks and profiler passes are outside
throughput. Five repeats alternate forward/reverse variant order.

Environment: Ascend910B2, CANN/ATB 9.0, Python 3.12.13, torch 2.10.0+cpu,
torch-npu 2.10.0, Transformers 5.5.4, TorchAir bundled with torch-npu, FP16,
internal formats enabled, JIT compile off. The final run also records the
installed bundled TorchAir cache-compiler source hash.
All lanes use the same packed decoder projections and NZ decoder weights;
NZ weights and NZ KV storage are different experiments.

## Published run evidence

- `b1_control/`: source `c9adbd0d`, two real smoke crops, B1, 163 useful decode
  tokens per trial, 435.97 median tok/s; both outputs reached EOS. Level0
  profiles verified all 24 layers and LM head, without kernel format metadata.
- `invalid_shared_code_timing/`: source `c9adbd0d`, deliberately interrupted
  multi-variant run. Its timings are discarded because shared Python forward
  code objects caused repeated TorchAir specialization warnings. The exact
  log and `INVALID_TIMING.json` are retained. Fixed in `79c6539e` by distinct
  code objects per graph and a no-new-graphs/no-recompilation timing gate.
- `b4_native_control/`: source `79c6539e`, two code and two formula crops,
  417 useful decode tokens. All three native paths matched to EOS and passed
  the new timing gates. Median tok/s: IncreFA ND 1050.00; FIA ND 371.17;
  blocked FIA ND 352.03. This small cohort differs from the mixed one below.
- `failed_processor_recording/`: source `f17c2c2d`, failed before inference
  because Transformers 5 SizeDict needed explicit assignment and conversion
  to a JSON dictionary. No throughput samples. Fixed in `41f9f9cc`.
- `mixed_native_and_private_contracts/`: source `41f9f9cc`, eight crops in
  two B4 cohorts: two code, two formula, two table and two text. Exact command,
  crop hashes, model identity, all token IDs, repeat times, external cache
  descriptors and Level1 profiles are preserved. All eight native outputs
  reached EOS with no caps, producing 785 useful decode tokens and 1472 raw
  token slots per trial. IncreFA ND 858.65 median tok/s (853.35–867.18), FIA ND
  284.38 (280.44–289.54), blocked FIA ND 272.35 (269.64–277.41). All timed
  samples matched the raw-eager production reference and recorded zero new
  graphs/recompilation warnings. NZ drift was rejected by this older strict
  fidelity gate; the newer run below measures it explicitly instead.

Every Level1 native profile contains 192 attention calls across eight steps,
24 distinct attention nodes, 384 cache-write kernels, 776 matmuls and eight
ArgMax kernels. Attention input formats are ND; no TransData kernels were
recorded. The accompanying `profile_audit.json` records actual shapes/formats.
These full-path differences are not isolated attention speed differences:
FIA also changes the cache writer and logical layout. In the separate diagnostic
profiles, FIA attention kernel medians were about 18–19 us, versus 23 us for
IncreFA, despite the slower full FIA decoder. FIA attention rows also recorded
about 33–35 ms total kernel wait time per eight-step trace, versus about 0.23 ms
for IncreFA. This flags a runtime-wait investigation; it does not establish the
cause or turn isolated kernel duration into a model speed prediction.

The private ATB lanes could not compile on this installed runtime. Both failed
at `atb._npu_reshape_and_cache.default`: TorchAir could not find registered
AscendIR `NpuReshapeAndCache`. Their reader was therefore never reached in a
compiled full model, and no private-paged throughput is available. This is a
compiler-lowering result, not a 310P kernel performance result. FIA is not that
private vLLM operator.

## Token drift and visible candidate timings

Source `97d4e7e9` separates performance gates from output fidelity. Candidate
rates are measured even with differing greedy sequences, with decoded outputs,
common-prefix lengths, token edit distance, EOS/cap counts and repeat stability
shown alongside. Runtime errors, changed storage descriptors and compilation
inside the timer still invalidate timing samples. The native control must match.
Large changes in generation length describe a different workload, so a tok/s
ratio alone is not an equal-work speedup.

The first mixed cohort's older NZ checks diverged after 2–9 tokens. Native
sequences including EOS had lengths 107,160,83,75; both NZ paths produced
1024,593,1024,501, with two hitting the cap. This is reported quantitatively,
not treated as interchangeable with a one-token numerical drift. See the newer
fully timed run and decoded outputs below for the actual performance and fidelity.

Processed kernel/operator traces and logs are gzip-compressed without changing
content. Compile caches and raw profiler device dumps are omitted from Git;
the run directories retain them on the validation machine. No inference result
in this directory validates 310P support or performance.

## Final timed mixed run with fidelity shown separately

`mixed_all_compiled_with_drift/` is source **97d4e7e9**, physical card **2**, two
B4 cohorts, KV4096, cap1024, five AB/BA repeats, eight real crops. It exited zero:
all five lanes passed performance gates. This status does not mean every lane
matched the reference. Every repeated sequence was stable within its lane.

| Decoder path | Median useful decode tok/s | Min–max | Useful tokens/trial | EOS / cap hits |
| --- | ---: | ---: | ---: | ---: |
| Dense IncreFA ND | 882.04 | 879.78–882.69 | 785 | 8 / 0 |
| Ordinary paged FIA ND | 294.94 | 294.67–300.02 | 785 | 8 / 0 |
| Blocked FIA ND | 285.96 | 283.84–288.61 | 785 | 8 / 0 |
| Dense IncreFA NZ storage | 898.66 | 893.37–899.80 | 5497 | 4 / 4 |
| Blocked FIA NZ storage | 330.58 | 328.61–334.11 | 5497 | 4 / 4 |

The NZ rows are measured rates, **not an equal-output speedup**. They generate
7.003 times as many non-EOS decode tokens, with 8184 raw slots instead of 1472.
Seven of eight outputs differ substantially from the reference: normalized
sequence token edit distances are 93.7–97.8% on those seven, with repetitive
text and four cap hits. One very short text output matches. See `decoded_text`
in each batch reference and variant validation, plus the exact token IDs and
per-item comparison. Token edit distance is a drift diagnostic, not an OCR
quality score. No candidate timing was suppressed merely because tokens differ.

All 50 timed samples have zero new graphs and zero recompilation warnings;
native descriptors remain 2 and NZ descriptors remain 29. Real prefill format
conversion preserved logical values before generation. Each of the ten final
profiles covers eight full decoder steps with 192 attention calls, 24 distinct
attention nodes, 384 cache writes, 776 matmuls and eight ArgMax operations.

The **compiled attention kernels consume ND in every final lane**. The two NZ
storage lanes each add 384 TransData kernels per eight-step profile: 48 NZ→ND
conversions per decoder step, one per K/V cache per layer. Scatter writers also
operate on ND inside those graphs. No ND→NZ TransData kernel appears in these
traces. External NZ retention therefore does not establish a native NZ-reading
attention kernel or correct persistence of converted cache writes. The exact
cause of the repetitive NZ generation has not been isolated. This is not a
verified vLLM-Ascend 310P native-NZ hot path.

The actual private ATB APIs remain a separate compiler compatibility result in
`mixed_native_and_private_contracts/`; they were not replaced with FIA and
called a private-op benchmark. The [310P handoff](../../FULL_MODEL_KV_310P_HANDOFF.md)
requires rediscovering that runtime and recording its actual input formats,
conversions, generated outputs and full-model timings.
