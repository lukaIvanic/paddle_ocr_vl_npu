# Table matcher strategy lab

All matchers use target-prefix tokens only, except the marked oracle upper bound.

## all (665 tables)

| matcher | accepted/call | coverage | target calls | decode speedup | CPU ms/table | CPU us/call |
|---|---:|---:|---:|---:|---:|---:|
| ecel_normalized_column_patch_w0.25 | 6.488 | 0.8504 | 40,002 | 6.146x | 5.906 | 98.178 |
| ecel_normalized | 6.370 | 0.8483 | 40,566 | 6.060x | 5.899 | 96.707 |

## latency_gt_1s (143 tables)

| matcher | accepted/call | coverage | target calls | decode speedup | CPU ms/table | CPU us/call |
|---|---:|---:|---:|---:|---:|---:|
| ecel_normalized_column_patch_w0.25 | 6.868 | 0.8642 | 23,186 | 6.737x | 17.485 | 107.841 |
| ecel_normalized | 6.664 | 0.8608 | 23,757 | 6.574x | 17.338 | 104.363 |

## latency_p75_plus (167 tables)

| matcher | accepted/call | coverage | target calls | decode speedup | CPU ms/table | CPU us/call |
|---|---:|---:|---:|---:|---:|---:|
| ecel_normalized_column_patch_w0.25 | 6.704 | 0.8599 | 25,532 | 6.539x | 15.958 | 104.380 |
| ecel_normalized | 6.505 | 0.8565 | 26,151 | 6.383x | 15.814 | 100.987 |
