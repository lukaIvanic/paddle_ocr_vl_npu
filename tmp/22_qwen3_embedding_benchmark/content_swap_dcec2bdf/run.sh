cd /workspace/repos/qwen-order-ed892111 || exit 1
source npu-setup
# No unoccupied device: physical 6 is healthy with >50 GiB free.
# A small BF16 accuracy probe can coexist with its resident TP worker.
export ASCEND_RT_VISIBLE_DEVICES=6 TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1
out=/workspace/results/qwen-content-swap-dcec2bdf
mkdir -p "$out"
{
 git rev-parse HEAD
 hostname
 printf '%s\n' 'Container: research_vllm_ascend_023_external_workspace' 'Chip: Ascend 910B2; physical NPU: 6; shared with resident TP worker; accuracy only' 'source npu-setup attempted; idle selector found no free device; explicit device/environment override follows' 'ASCEND_RT_VISIBLE_DEVICES=6 TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1' '/usr/local/python3.12.13/bin/python3 -u 22_qwen3_embedding_benchmark/probe_document_first.py --model /workspace/models/Qwen3-Reranker-4B --fixture /workspace/results/clef_touche_full/fixture.json --output /workspace/results/qwen-content-swap-dcec2bdf/result.json'
} > "$out/command.txt"
/usr/local/python3.12.13/bin/python3 -u 22_qwen3_embedding_benchmark/probe_document_first.py --model /workspace/models/Qwen3-Reranker-4B --fixture /workspace/results/clef_touche_full/fixture.json --output "$out/result.json" > "$out/run.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$out/exit_code.txt"
tail -55 "$out/run.log"
exit "$rc"
