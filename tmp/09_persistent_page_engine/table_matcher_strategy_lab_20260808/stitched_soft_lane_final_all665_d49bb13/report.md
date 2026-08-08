# Table matcher strategy lab

All matchers use target-prefix tokens only, except the marked oracle upper bound.

## all (665 tables)

| matcher | accepted/call | coverage | target calls | decode speedup | CPU ms/table | CPU us/call |
|---|---:|---:|---:|---:|---:|---:|
| ecel_normalized_no_cursor_stitched_lane_prior_column_patch_w0.25 | 6.184 | 0.8450 | 41,443 | 5.930x | 6.347 | 101.843 |
| ecel_normalized_no_cursor_stitched_column_patch_w0.25 | 6.086 | 0.8432 | 41,950 | 5.857x | 6.225 | 98.675 |
| ecel_normalized_no_cursor_column_patch_w0.25 | 6.083 | 0.8431 | 41,963 | 5.855x | 8.818 | 139.734 |

## latency_gt_1s (143 tables)

| matcher | accepted/call | coverage | target calls | decode speedup | CPU ms/table | CPU us/call |
|---|---:|---:|---:|---:|---:|---:|
| ecel_normalized_no_cursor_stitched_lane_prior_column_patch_w0.25 | 6.540 | 0.8587 | 24,113 | 6.476x | 18.806 | 111.526 |
| ecel_normalized_no_cursor_stitched_column_patch_w0.25 | 6.416 | 0.8565 | 24,489 | 6.376x | 18.650 | 108.904 |
| ecel_normalized_no_cursor_column_patch_w0.25 | 6.414 | 0.8565 | 24,494 | 6.374x | 22.014 | 128.523 |

## latency_p75_plus (167 tables)

| matcher | accepted/call | coverage | target calls | decode speedup | CPU ms/table | CPU us/call |
|---|---:|---:|---:|---:|---:|---:|
| ecel_normalized_no_cursor_stitched_lane_prior_column_patch_w0.25 | 6.406 | 0.8548 | 26,465 | 6.307x | 17.114 | 107.991 |
| ecel_normalized_no_cursor_stitched_column_patch_w0.25 | 6.271 | 0.8524 | 26,915 | 6.200x | 16.979 | 105.347 |
| ecel_normalized_no_cursor_column_patch_w0.25 | 6.270 | 0.8523 | 26,919 | 6.200x | 20.337 | 126.166 |
