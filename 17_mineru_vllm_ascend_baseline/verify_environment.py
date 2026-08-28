#!/usr/bin/env python3
"""Verify the exact stock vLLM-Ascend environment required by experiment 17."""

from __future__ import annotations

import argparse
import inspect
import json
from importlib.metadata import version
from pathlib import Path


EXPECTED = {
    "vllm": "0.21.0+empty",
    "vllm-ascend": "0.21.0rc1",
    "torch": "2.10.0+cpu",
    "torch-npu": "2.10.0",
    "transformers": "5.5.4",
    "mineru-vl-utils": "1.0.5",
    "httpx-retries": "0.6.0",
}
VLLM_PACKAGES = {"vllm", "vllm-ascend"}


def version_mismatches(
    actual: dict[str, str],
    *,
    allow_vllm_version_drift: bool,
) -> dict[str, dict[str, str]]:
    checked = (
        set(EXPECTED) - VLLM_PACKAGES
        if allow_vllm_version_drift
        else set(EXPECTED)
    )
    return {
        name: {"expected": EXPECTED[name], "actual": actual[name]}
        for name in sorted(checked)
        if actual[name] != EXPECTED[name]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--allow-vllm-version-drift",
        action="store_true",
        help=(
            "Record the installed vLLM and vLLM-Ascend versions without "
            "requiring the 910B reference versions. Required APIs and every "
            "other package version remain strict."
        ),
    )
    args = parser.parse_args()

    from mineru_vl_utils import MinerUClient
    import numpy
    import vllm
    import vllm_ascend
    from vllm import AsyncEngineArgs

    actual = {name: version(name) for name in EXPECTED}
    mismatches = version_mismatches(
        actual,
        allow_vllm_version_drift=args.allow_vllm_version_drift,
    )
    if mismatches:
        raise RuntimeError(f"environment mismatch: {mismatches}")

    engine_parameters = inspect.signature(AsyncEngineArgs).parameters
    required_engine_parameters = {
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "block_size",
        "max_num_seqs",
        "max_num_batched_tokens",
        "additional_config",
        "compilation_config",
    }
    missing_engine = sorted(required_engine_parameters - set(engine_parameters))
    if missing_engine:
        raise RuntimeError(f"AsyncEngineArgs missing parameters: {missing_engine}")
    if not hasattr(MinerUClient, "concurrent_two_step_extract"):
        raise RuntimeError("MinerUClient has no concurrent_two_step_extract")

    result = {
        "status": "ENVIRONMENT_READY",
        "packages": actual,
        "version_policy": (
            "record_vllm_versions_require_other_versions_exact"
            if args.allow_vllm_version_drift
            else "require_all_reference_versions_exact"
        ),
        "vllm_source": vllm.__file__,
        "vllm_ascend_source": vllm_ascend.__file__,
        "numpy": numpy.__version__,
        "async_engine_required_parameters": sorted(required_engine_parameters),
        "mineru_concurrent_two_step_extract": True,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
