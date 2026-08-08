# Table matcher failure analysis

Matcher: `ecel_normalized_no_cursor`. Cohort: 143 tables with measured B1 latency > 1.000 s. K=16.

## Headline

- Practical calls analyzed: 24,509
- Calls where practical equals the local oracle: 49.4%
- Calls with an oracle gap: 50.6%
- Lost opportunity tokens with no matching prefix anchor: 46.1%
- Lost opportunity tokens despite a legal prefix anchor: 53.9%
- Gap calls with ambiguous anchors: 68.4%
- Gap calls selecting a different draft row than the oracle: 77.0%

## First divergence category

| category | gap events | oracle-opportunity tokens lost |
|---|---:|---:|
| numeric | 6,102 | 44,750 |
| text | 2,024 | 12,301 |
| table_structure | 1,483 | 10,808 |
| cjk | 954 | 9,009 |
| math_or_latex | 1,191 | 4,410 |
| punctuation | 554 | 3,408 |
| whitespace | 103 | 567 |

## Worst tables

| request | latency s | target | draft | gap calls | lost opportunity tokens |
|---|---:|---:|---:|---:|---:|
| `page_000271_table_box_id_1` | 4.822 | 3,094 | 3,673 | 669 | 4,232 |
| `page_001276_table_2` | 3.461 | 2,202 | 1,989 | 821 | 3,765 |
| `page_000279_table_box_id_0` | 5.331 | 3,060 | 3,963 | 687 | 3,616 |
| `page_000292_table_box_id_1` | 2.689 | 1,682 | 2,347 | 323 | 3,205 |
| `page_000279_table_box-fy04hrwa` | 3.374 | 2,157 | 2,857 | 489 | 2,647 |
| `page_000255_table_box_id_2` | 2.172 | 1,297 | 1,365 | 294 | 2,343 |
| `page_000263_table_box_id_7` | 2.447 | 1,508 | 1,411 | 668 | 2,289 |
| `page_000626_table_box_id_4` | 2.336 | 1,468 | 1,258 | 580 | 2,214 |
| `page_000277_table_box_id_1` | 4.377 | 2,804 | 2,782 | 324 | 2,191 |
| `page_000216_table_box_id_0` | 2.918 | 1,518 | 1,421 | 244 | 2,112 |
| `page_000273_table_box_id_1` | 2.953 | 1,837 | 1,756 | 256 | 2,105 |
| `page_001367_table_60` | 2.574 | 1,632 | 1,715 | 243 | 2,097 |
| `page_000290_table_box_id_1` | 4.848 | 3,112 | 4,192 | 207 | 1,967 |
| `page_000282_table_box_id_1` | 4.787 | 3,092 | 3,094 | 225 | 1,876 |
| `page_001375_table_4` | 1.823 | 1,185 | 1,180 | 295 | 1,743 |
| `page_000283_table_box_id_1` | 4.823 | 3,098 | 3,217 | 139 | 1,526 |
| `page_000295_table_box_id_1` | 4.685 | 3,018 | 3,103 | 128 | 1,502 |
| `page_000280_table_box_id_1` | 2.487 | 1,519 | 1,525 | 241 | 1,296 |
| `page_000199_table_box_id_0` | 1.494 | 855 | 891 | 149 | 1,217 |
| `page_000270_table_box_id_0` | 2.251 | 1,339 | 1,273 | 135 | 1,203 |
| `page_000276_table_box_id_5` | 3.855 | 2,446 | 2,333 | 105 | 1,041 |
| `page_000511_table_1` | 1.575 | 904 | 892 | 183 | 1,033 |
| `page_000446_table_3` | 2.437 | 1,510 | 1,534 | 79 | 995 |
| `page_001273_table_8` | 1.526 | 894 | 872 | 90 | 985 |
| `page_001188_table_2` | 3.067 | 1,900 | 1,943 | 72 | 962 |
