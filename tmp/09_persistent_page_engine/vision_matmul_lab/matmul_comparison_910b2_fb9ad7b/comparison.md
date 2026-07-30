# 910B2 vision MatMul comparison

| Shape/config | Full stage ms | Physical tok/s | MatMul ms | MatMul-only TFLOP/s |
|---|---:|---:|---:|---:|
| B1xS512 native 4304 ND | 15.757428 | 32492.6 | 4.959353 | 84.877 |
| B1xS512 padded 4352 NZ | 13.604366 | 37635.0 | 2.858633 | 148.320 |
| B4xS512 native 4304 ND | 27.347794 | 74887.2 | 8.004093 | 210.360 |
| B4xS512 padded 4352 NZ | 26.655551 | 76832.0 | 7.366140 | 230.239 |
| B1xS2048 native 4304 ND | 31.554370 | 64903.8 | 7.983893 | 210.893 |
| B1xS2048 padded 4352 NZ | 30.571022 | 66991.5 | 7.339400 | 231.078 |

## Padded 4352 NZ versus native 4304 ND

- B1xS512: full stage -13.664%, physical tok/s +15.826%, MatMul time -42.359%, MatMul TFLOP/s +74.747%
- B4xS512: full stage -2.531%, physical tok/s +2.597%, MatMul time -7.970%, MatMul TFLOP/s +9.450%
- B1xS2048: full stage -3.116%, physical tok/s +3.217%, MatMul time -8.072%, MatMul TFLOP/s +9.571%
