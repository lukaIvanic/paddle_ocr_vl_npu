#!/bin/bash
source npu-setup
# Card 2 was verified free/healthy before launch; do not migrate mid-experiment.
export ASCEND_RT_VISIBLE_DEVICES=2
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /workspace/repos/paddle_ocr_vl_npu_vision_diagnostics_retry || exit 1
R=$(cat /tmp/mineru_vision_nz_pairs_active_run.txt)
OLD=/workspace/repos/paddle_ocr_vl_npu_mineru_kv_probe/tmp/11_mineru_2_5_pro_inference/vision_crop_contracts_910B_20261007T115828Z_82575a91
/usr/local/python3.12.13/bin/python3 -u 11_mineru_2_5_pro_inference/run_production_attention_matrix.py \
 --capture-dir "$OLD/capture" --cache-root "$OLD/vision_cache" \
 --config-json "$R/control_config.json" --output-dir "$R/matrix" \
 --routes crop_0_bucket_768,crop_1_bucket_3072 \
 --variants baseline,pfa_nz_weights,grouped_qkv,grouped_qkv_nz_weights,grouped_qkv_mlp_fc1,grouped_qkv_mlp_fc1_nz_weights,pfa_d128,pfa_d128_nz_weights,eager_pfa,eager_pfa_nz_weights,unpad_d128,unpad_d128_nz_weights \
 --steps 30 --profile --timeout-s 1800
status=$?
printf '%s\n' "$status" > "$R/exit_code.txt"
exit "$status"
