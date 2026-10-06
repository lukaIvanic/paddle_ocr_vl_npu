cd /workspace/repos/qwen-order-ed892111 || exit 1
source npu-setup
export ASCEND_RT_VISIBLE_DEVICES=2 TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1
out=/workspace/results/qwen-positional-5e28f6a8
mkdir -p "$out"
{
 git rev-parse HEAD
 hostname
 printf '%s\n' 'Container: research_vllm_ascend_023_external_workspace' 'Chip: Ascend 910B2; NPU 2; shared with resident TP worker; not a latency benchmark' 'source npu-setup attempted; no idle device; override ASCEND_RT_VISIBLE_DEVICES=2 TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1' '/usr/local/python3.12.13/bin/python3 -u 22_qwen3_embedding_benchmark/investigate_prompt_roles.py --model /workspace/models/Qwen3-Reranker-4B --prior-result tmp/22_qwen3_embedding_benchmark/content_swap_dcec2bdf/result.json --fixture /workspace/results/clef_touche_full/fixture.json --phase holdout --positional-instructions --max-document-tokens 400 --output /workspace/results/qwen-positional-5e28f6a8/result.json'
} > "$out/command.txt"
/usr/local/python3.12.13/bin/python3 -u 22_qwen3_embedding_benchmark/investigate_prompt_roles.py --model /workspace/models/Qwen3-Reranker-4B --prior-result tmp/22_qwen3_embedding_benchmark/content_swap_dcec2bdf/result.json --fixture /workspace/results/clef_touche_full/fixture.json --phase holdout --positional-instructions --max-document-tokens 400 --output "$out/result.json" > "$out/run.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$out/exit_code.txt"
tail -8 "$out/run.log"
exit "$rc"
