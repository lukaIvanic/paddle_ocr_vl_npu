#!/usr/bin/env python3

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


DEFAULT_TASK = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = (
    "<|im_start|>system\n"
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Transformers Qwen3 reranker reference.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--model-dir")
    parser.add_argument("--query", default="What is the capital of China?")
    parser.add_argument(
        "--documents",
        nargs="+",
        default=[
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other.",
        ],
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def parse_device(device_name: str) -> torch.device:
    if device_name == "auto":
        try:
            import torch_npu  # noqa: F401
        except ModuleNotFoundError:
            return torch.device("cpu")
        if torch.npu.is_available():
            return torch.device("npu:0")
        return torch.device("cpu")
    if device_name.startswith("npu"):
        import torch_npu  # noqa: F401
    return torch.device(device_name)


def format_instruction(task: str, query: str, document: str) -> str:
    return f"<Instruct>: {task}\n<Query>: {query}\n<Document>: {document}"


def build_inputs(
    tokenizer,
    pairs: list[str],
    *,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
    body_max_length = max_length - len(prefix_tokens) - len(suffix_tokens)
    if body_max_length <= 0:
        raise ValueError("max_length is too small for the reranker prompt wrapper")

    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=body_max_length,
    )
    input_ids = [prefix_tokens + item + suffix_tokens for item in inputs["input_ids"]]
    padded = tokenizer.pad(
        {"input_ids": input_ids},
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
        max_length=max_length,
    )
    return {key: value.to(device) for key, value in padded.items()}


@torch.inference_mode()
def compute_scores(
    model,
    inputs: dict[str, torch.Tensor],
    *,
    token_false_id: int,
    token_true_id: int,
) -> torch.Tensor:
    logits = model(**inputs).logits[:, -1, :]
    false_logits = logits[:, token_false_id]
    true_logits = logits[:, token_true_id]
    yes_no_logits = torch.stack((false_logits, true_logits), dim=1)
    return F.log_softmax(yes_no_logits, dim=1)[:, 1].exp()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError("transformers is required to run this reference script") from exc

    model_name_or_path = args.model_dir or args.model_id
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = parse_device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).eval()
    model.to(device)

    batch_size = args.batch_size or len(args.documents)
    if batch_size != len(args.documents):
        raise ValueError("--batch-size must match len(--documents) for this static reference script")

    pairs = [format_instruction(args.task, args.query, document) for document in args.documents]
    inputs = build_inputs(tokenizer, pairs, max_length=args.max_length, device=device)
    scores = compute_scores(
        model,
        inputs,
        token_false_id=tokenizer.convert_tokens_to_ids("no"),
        token_true_id=tokenizer.convert_tokens_to_ids("yes"),
    )
    ranked = sorted(enumerate(scores.detach().float().cpu().tolist()), key=lambda item: item[1], reverse=True)

    print(f"shape={tuple(scores.shape)}")
    print(f"dtype={scores.dtype}")
    print(f"device={scores.device}")
    print(f"scores={scores.detach().float().cpu().tolist()}")
    print(f"ranked={ranked}")


if __name__ == "__main__":
    main()
