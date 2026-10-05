# Source provenance

The local implementation adapts these Apache-2.0 sources. The license is included
as `LICENSE.apache-2.0`.

- `modeling_backbone.py`: Transformers **5.17.0**, `models/qwen3_5/modeling_qwen3_5.py`.
  Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
  Inspected file SHA256: `762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b`.
  Changed to a text-only, B1, no-cache eager forward; removed the Transformers
  framework, optional kernel dispatch, export path and generation/vision code.
  The chunk scan, normalization and attention retain reference arithmetic.
- `modeling_head.py` and `text_inputs.py`: Cloudflare **clef-flash**,
  `joint_schema_model.py`, revision `17f0b0ad64efb65d273590632833508766b2aae6`.
  Inspected file SHA256: `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`.
  The head computation is unchanged. Text encoding uses `tokenizers` directly,
  rejects media and overlength inputs, and omits padding and the serving wrapper.

Sources were read from the exact environment/release used for the saved 910B
baseline. The local runtime never imports either upstream Python file. Weights
and tokenizer remain in the separately downloaded, digest-verified release.
