# Table matcher strategy lab

All matchers use target-prefix tokens only, except the marked oracle upper bound.

## all (665 tables)

| matcher | accepted/call | coverage | target calls | decode speedup | CPU ms/table | CPU us/call |
|---|---:|---:|---:|---:|---:|---:|
| ecel_normalized_no_cursor_column_patch_w0.25 | 6.074 | 0.8428 | 42,054 | 5.843x | 6.165 | 97.480 |
| no_cursor_column_patch_w0.25 | 6.007 | 0.8418 | 42,314 | 5.806x | 6.355 | 99.875 |

## latency_gt_1s (143 tables)

| matcher | accepted/call | coverage | target calls | decode speedup | CPU ms/table | CPU us/call |
|---|---:|---:|---:|---:|---:|---:|
| ecel_normalized_no_cursor_column_patch_w0.25 | 6.411 | 0.8564 | 24,509 | 6.371x | 18.476 | 107.802 |
| no_cursor_column_patch_w0.25 | 6.325 | 0.8550 | 24,749 | 6.308x | 19.248 | 111.213 |

## latency_p75_plus (167 tables)

| matcher | accepted/call | coverage | target calls | decode speedup | CPU ms/table | CPU us/call |
|---|---:|---:|---:|---:|---:|---:|
| ecel_normalized_no_cursor_column_patch_w0.25 | 6.267 | 0.8522 | 26,938 | 6.195x | 16.823 | 104.291 |
| no_cursor_column_patch_w0.25 | 6.188 | 0.8509 | 27,178 | 6.139x | 17.326 | 106.461 |
