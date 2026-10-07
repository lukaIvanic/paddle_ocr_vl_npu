# Historical host context availability — 2026-10-07

These fields were not recorded per lane. They cannot be recovered from current
host state. The original archive contains only run-boundary `occupancy_before.txt`,
`occupancy_after.txt` and padding-matrix occupancy snapshots. No current snapshot
is being passed off as historical evidence.

| Lane | Host load / CPU count before & after | Other jobs before & after | Device clock / power mode per lane |
|---|---|---|---|
| baseline / crop_0_bucket_768 | Unavailable | Unavailable; run-boundary snapshots only | Unavailable |
| baseline / crop_1_bucket_3072 | Unavailable | Unavailable; run-boundary snapshots only | Unavailable |
| eager_pfa / crop_0_bucket_768 | Unavailable | Unavailable; run-boundary snapshots only | Unavailable |
| eager_pfa / crop_1_bucket_3072 | Unavailable | Unavailable; run-boundary snapshots only | Unavailable |
| pfa_nz_weights / crop_0_bucket_768 | Unavailable | Unavailable; run-boundary snapshots only | Unavailable |
| pfa_nz_weights / crop_1_bucket_3072 | Unavailable | Unavailable; run-boundary snapshots only | Unavailable |
| unpad_d128 / crop_0_bucket_768 | Unavailable | Unavailable; run-boundary snapshots only | Unavailable |
| unpad_d128 / crop_1_bucket_3072 | Unavailable | Unavailable; run-boundary snapshots only | Unavailable |
| unpad_d128_nz_weights / crop_0_bucket_768 | Unavailable | Unavailable; run-boundary snapshots only | Unavailable |
| unpad_d128_nz_weights / crop_1_bucket_3072 | Unavailable | Unavailable; run-boundary snapshots only | Unavailable |
| pfa_d128 / crop_0_bucket_768 | Unavailable | Unavailable; run-boundary snapshots only | Unavailable |
| pfa_d128 / crop_1_bucket_3072 | Unavailable | Unavailable; run-boundary snapshots only | Unavailable |

The available initial occupancy snapshot showed other PIDs on cards 0, 1, 2 and
4–7; card 3 was free when selected. The run used card 3. This does not establish
those jobs or host load stayed constant across individual lanes. Consult the
raw snapshots for the recorded PIDs and memory, without inferring job ownership.
