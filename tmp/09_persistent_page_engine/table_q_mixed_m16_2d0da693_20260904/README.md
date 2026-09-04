# Mixed M16 verifier and draft benchmark

This run tests one compiled transformer pass with 16 real token positions:

- One B1Q8 verifier branch with KV4096 and manual grouped attention.
- One B8Q1 draft branch with KV768 and IncreFA.
- One shared M16 RMSNorm, QKV projection, output projection, MLP, and compact
  LM head.

All runs used separate processes and model loads on an otherwise idle Ascend
910B2, physical NPU 6. Each process ran 10 real-input warmups followed by 50
measured calls. The mixed outputs matched the isolated verifier and draft
anchors for all 16 native token IDs.

| Measurement | Median call |
|---|---:|
| Control A, B8Q1 | 1.131 ms |
| Control A, B1Q8 | 1.328 ms |
| Control A sum | 2.458 ms |
| Mixed A | 2.400 ms |
| Mixed B | 2.473 ms |
| Control B, B1Q8 | 1.322 ms |
| Control B, B8Q1 | 1.122 ms |
| Control B sum | 2.444 ms |

The two control sums average 2.451 ms. The two mixed runs average 2.436 ms.
The nominal mixed speedup is 1.006x, or a 0.6% latency reduction. This is
smaller than the 73 microsecond spread between the two mixed processes, so the
result is practical parity rather than a demonstrated speedup.

An earlier profile, before the final zero-copy Q1 layout change, measured these
kernel groups for one mixed call:

| Kernel group | Device time |
|---|---:|
| Matmuls, including the LM head | 652 us |
| B8Q1 IncreFA | 345 us |
| Q8 QK and PV batch matmuls | 326 us |
| Q8 scaled masked softmax | 146 us |
| KV scatter | 158 us |
| Transpose, slice, concatenate, and split | 380 us |
| AddRMSNorm | 133 us |
| RoPE | 100 us |

The final zero-copy change removed the draft branch's packed concatenation,
transpose, and split. It reduced ordinary mixed latency from about 2.55 ms to
the 2.40-2.47 ms range above. The remaining split-attention implementation does
not yet produce a material advantage over separate B8Q1 and B1Q8 calls.
