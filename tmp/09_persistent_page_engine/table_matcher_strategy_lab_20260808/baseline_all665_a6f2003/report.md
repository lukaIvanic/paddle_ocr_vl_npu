# Table matcher strategy lab

All matchers use target-prefix tokens only, except the marked oracle upper bound.

## all (665 tables)

| matcher | accepted/call | coverage | target calls | decode speedup | CPU ms/table | CPU us/call |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 10.210 | 0.9042 | 25,616 | 9.556x | 0.565 | 14.671 |
| column_patch_w0.25 | 6.429 | 0.8498 | 40,185 | 6.117x | 6.059 | 100.261 |
| start_prior | 6.326 | 0.8480 | 40,667 | 6.043x | 4.768 | 77.964 |
| reversible | 6.317 | 0.8477 | 40,729 | 6.034x | 4.800 | 78.379 |
| current | 5.934 | 0.8403 | 42,708 | 5.752x | 5.139 | 80.012 |

## latency_gt_1s (143 tables)

| matcher | accepted/call | coverage | target calls | decode speedup | CPU ms/table | CPU us/call |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 11.080 | 0.9144 | 14,617 | 10.649x | 1.706 | 16.691 |
| column_patch_w0.25 | 6.778 | 0.8629 | 23,407 | 6.672x | 18.202 | 111.198 |
| start_prior | 6.572 | 0.8594 | 23,998 | 6.506x | 14.501 | 86.409 |
| reversible | 6.567 | 0.8593 | 24,011 | 6.503x | 14.557 | 86.694 |
| current | 6.056 | 0.8499 | 25,630 | 6.089x | 15.752 | 87.885 |

## latency_p75_plus (167 tables)

| matcher | accepted/call | coverage | target calls | decode speedup | CPU ms/table | CPU us/call |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 10.871 | 0.9118 | 16,089 | 10.344x | 1.551 | 16.103 |
| column_patch_w0.25 | 6.621 | 0.8587 | 25,754 | 6.481x | 16.414 | 106.435 |
| start_prior | 6.433 | 0.8554 | 26,351 | 6.334x | 13.079 | 82.886 |
| reversible | 6.429 | 0.8554 | 26,366 | 6.330x | 13.132 | 83.174 |
| current | 5.939 | 0.8460 | 28,077 | 5.942x | 14.182 | 84.353 |
