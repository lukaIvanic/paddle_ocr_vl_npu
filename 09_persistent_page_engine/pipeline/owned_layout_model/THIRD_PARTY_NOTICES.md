# Third-party notice

The eager PP-DocLayoutV3 and HGNetV2 implementation in this directory is an
inference-focused adaptation of the corresponding implementations distributed
in Hugging Face Transformers 5.5.4.

The PP-DocLayoutV3 source is copyright the PaddlePaddle Team and the
HuggingFace Inc. team. The HGNetV2 source is copyright Baidu Inc. and the
HuggingFace Inc. team. Both upstream implementations are licensed under the
Apache License, Version 2.0:

https://www.apache.org/licenses/LICENSE-2.0

The adapted Python source files retain their upstream copyright and license
headers. This project removes the Transformers framework integration and
training surfaces, retains the checkpoint-compatible module topology and eager
inference math, and adds a strict local safetensors loader.
