"""Load Clef-flash's text backbone, output embeddings and head from safetensors.

No Transformers, remote Python, vision model, or vocabulary-logit computation.
The untied lm_head weights are still needed by the head's lexical option prior.
"""
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

from safetensors import safe_open
from safetensors.torch import load_file
import torch
from torch import nn

from download_model import verify
from modeling_backbone import TextBackbone
from modeling_head import JointSchemaHead


class ClefTextModel(nn.Module):
    def __init__(self, config, head_config):
        super().__init__()
        self.backbone = TextBackbone(config)
        self.lm_head = nn.Embedding(config.vocab_size, config.hidden_size)
        self.head = JointSchemaHead(**head_config)

    def forward(self, input_ids, record):
        if input_ids.shape != (1, len(record.input_ids)):
            raise ValueError("Expected one unpadded sequence matching the encoded record")
        hidden = self.backbone(input_ids)
        return self.head(hidden, input_ids, torch.ones_like(input_ids), [record], self.lm_head.weight)[0]


def load_model(directory, device, progress=lambda name: None):
    """Load only the pinned BF16 text checkpoint; verify before using its contents."""
    directory = Path(directory)
    release = json.loads(Path(__file__).with_name("release.json").read_text())
    index_name = "model.safetensors.index.json"
    needed = ["config.json", "joint_head_config.json", index_name, "joint_head.safetensors", "tokenizer.json"]
    needed += [name for name in release["files"] if name.startswith("model-")]
    for name in needed:
        if not verify(directory / name, release["files"][name]):
            raise ValueError(f"Pinned release verification failed: {name}")
        progress("verified:" + name)
    config = SimpleNamespace(**json.loads((directory / "config.json").read_text())["text_config"])
    head_config = json.loads((directory / "joint_head_config.json").read_text())
    # The verified config fixes the architecture; there is no general-model API.
    with torch.device("meta"):
        model = ClefTextModel(config, head_config)
    weight_map = json.loads((directory / index_name).read_text())["weight_map"]
    expected = {name: parameter for name, parameter in model.named_parameters() if not name.startswith("head.")}
    selected = {}
    for source in weight_map:
        if source.startswith("model.language_model."):
            selected["backbone." + source.removeprefix("model.language_model.")] = source
        elif source == "lm_head.weight":
            selected[source] = source
        elif not source.startswith("model.visual."):
            raise ValueError(f"Unexpected checkpoint tensor: {source}")
    if selected.keys() != expected.keys():
        raise ValueError(f"Checkpoint keys differ: {selected.keys() ^ expected.keys()}")
    shards = defaultdict(list)
    for name, source in selected.items():
        shards[weight_map[source]].append((name, source))
    for shard, entries in shards.items():
        state = {}
        with safe_open(directory / shard, framework="pt", device="cpu") as tensors:
            for name, source in entries:
                value = tensors.get_tensor(source)
                if value.shape != expected[name].shape or value.dtype != torch.bfloat16:
                    raise ValueError(f"Unexpected shape/dtype for {source}: {value.shape}, {value.dtype}")
                state[name] = value.to(device)
        model.load_state_dict(state, strict=False, assign=True)
        progress("loaded:" + shard)
    head_state = {k: v.to(device=device, dtype=torch.bfloat16)
                  for k, v in load_file(directory / "joint_head.safetensors").items()}
    model.head.load_state_dict(head_state, strict=True, assign=True)
    model.backbone.inv_freq = model.backbone.inv_freq.to(device)
    if any(p.is_meta or p.dtype != torch.bfloat16 or p.device != torch.device(device)
           for p in model.parameters()):
        raise RuntimeError("Incomplete BF16 checkpoint load")
    return model.eval()
