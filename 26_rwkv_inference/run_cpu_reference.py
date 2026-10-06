"""Prepare the pinned C FP32 embedding runtime, or save a small CPU smoke anchor.

All inference, tokenization and preprocessing run in upstream C. PyTorch is used
only by `prepare` to read/export the checkpoint. No CUDA/NPU runtime is imported
by `smoke`. Upstream: howard-hou/EmbeddingRWKV, Apache-2.0.
"""
import argparse
import ast
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import time

UPSTREAM_COMMIT = "3c306736c58550f4be6d384be068512ba9bfbd72"
MODEL_REVISION = "d6bfff190b6ce4fb6bbd9c574e7e26df3df64075"
CHECKPOINT_SHA256 = "9033eec92f163d1a710474977fa3fb68b7ee04697e0e961d83434743bd256a15"
C_SHA256 = "50ef7f733b35ec6fb4fbdeab14b8aba3361db50ebf54e615cea117ec9516a3a1"
VOCAB_SHA256 = "e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89"
INSTRUCTION = "Instruct: Given a query, retrieve documents that answer the query\nQuery: {query}"
CONTEXT = 2048
EOS_CHUNK = 512
EOS = 65535


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def host_info():
    try:
        commit = subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parent),
                                          "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"hostname": platform.node(), "platform": platform.platform(),
            "machine": platform.machine(), "python": sys.version,
            "script_sha256": sha256(__file__), "source_commit": commit,
            "command": sys.argv}


def prepare(args):
    """Export exactly as upstream export_weights.py; compile unmodified C."""
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    torch.set_num_threads(4)

    checkpoint = args.checkpoint.resolve()
    source = args.upstream.resolve() / "rwkv-emb.c/rwkv_emb.c"
    vocabulary = args.upstream.resolve() / "embedding/eval/tokenizer/rwkv_vocab_v20230424.txt"
    license_path = args.upstream.resolve() / "rwkv-emb.c/LICENSE"
    require_hash(checkpoint, CHECKPOINT_SHA256)
    require_hash(source, C_SHA256)
    require_hash(vocabulary, VOCAB_SHA256)
    license_text = license_path.read_text()
    if "Apache License" not in license_text:
        raise ValueError("Missing upstream Apache license")
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    channels = state["rwkv.emb.weight"].shape[1]
    layers = max(int(k.split(".")[2]) for k in state if k.startswith("rwkv.blocks.")) + 1
    if channels != 768 or layers != 12 or state["rwkv.emb.weight"].shape[0] != 65536:
        raise ValueError("Expected the released 12-layer, 768-channel 0.1B checkpoint")

    runtime = args.runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, runtime / "rwkv_emb.c")
    shutil.copyfile(license_path, runtime / "LICENSE")
    compiler = shutil.which(args.compiler)
    if not compiler:
        raise ValueError(f"Compiler not found: {args.compiler}")
    command = [compiler, "-O3", "-fopenmp", "-shared", "-fPIC", "-DRWKV_NO_MAIN",
               str(runtime / "rwkv_emb.c"), "-lm", "-o", str(runtime / "librwkv_emb.so")]
    built = subprocess.run(command, text=True, capture_output=True)
    (runtime / "build.log").write_text(built.stdout + built.stderr)
    if built.returncode:
        raise RuntimeError(f"C build failed; see {runtime / 'build.log'}")

    shapes = {}
    with (runtime / "embedding-fp32.bin").open("wb") as stream:
        keys = [k for k in state if k.startswith(("rwkv.", "head.")) and k != "rwkv.head.weight"]
        if len(keys) != 410:
            raise ValueError(f"Expected 410 exported tensors; found {len(keys)}")
        stream.write(struct.pack("<8I", 0x52454D42, 1, layers, channels, 65536, 64, len(keys), 0))
        for key in keys:
            value = state[key].float()
            if value.device.type != "cpu" or value.ndim > 4:
                raise ValueError(f"Unexpected export tensor: {key}")
            if value.ndim == 2 and key.endswith(".weight") and key != "rwkv.emb.weight":
                value = value.T
            array = value.contiguous().numpy().astype("<f4", copy=False)
            name = key.encode("ascii")
            if len(name) >= 128:
                raise ValueError(f"Tensor name too long: {key}")
            shapes[key] = list(array.shape)
            stream.write(name.ljust(128, b"\0"))
            stream.write(struct.pack("<5IQ", array.ndim, *(list(array.shape) + [1] * 4)[:4], array.size))
            stream.write(array.tobytes())

    vocab = []
    seen = set()
    for line in vocabulary.read_text().splitlines():
        start, end = line.index(" "), line.rindex(" ")
        index = int(line[:start])
        value = ast.literal_eval(line[start:end])
        if isinstance(value, str):
            value = value.encode("utf-8")
        if not isinstance(value, bytes) or len(value) != int(line[end:]) or index in seen:
            raise ValueError("Malformed tokenizer vocabulary")
        seen.add(index)
        vocab.append((index, value))
    with (runtime / "tokenizer.bin").open("wb") as stream:
        stream.write(struct.pack("<2I", 0x52564F43, len(vocab)))
        for index, value in vocab:
            stream.write(struct.pack("<2I", index, len(value)))
            stream.write(value)
    files = {name: sha256(runtime / name) for name in
             ["embedding-fp32.bin", "tokenizer.bin", "librwkv_emb.so", "rwkv_emb.c", "LICENSE"]}
    manifest = {"schema_version": 1, "checkpoint": str(checkpoint),
                "checkpoint_sha256": CHECKPOINT_SHA256, "model_revision": MODEL_REVISION,
                "upstream_commit": UPSTREAM_COMMIT, "vocab_sha256": VOCAB_SHA256,
                "layers": layers, "channels": channels, "dtype": "float32",
                "matrix_layout": "input,output", "tensor_shapes": shapes,
                "files_sha256": files, "compiler_command": command,
                "compiler_version": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
                "torch": torch.__version__, "inference_executed": False, **host_info()}
    write_json(runtime / "manifest.json", manifest)
    print(json.dumps({"runtime": str(runtime), "exported_tensors": len(shapes),
                      "inference_executed": False}, indent=2), flush=True)


class CReference:
    """Path-configurable subset of upstream rwkv_c.py's ctypes transport."""

    def __init__(self, runtime, threads):
        import numpy as np
        self.np = np
        self.handle = None
        self.manifest = json.loads((runtime / "manifest.json").read_text())
        if (self.manifest["checkpoint_sha256"] != CHECKPOINT_SHA256 or
                self.manifest["upstream_commit"] != UPSTREAM_COMMIT or
                self.manifest["files_sha256"]["rwkv_emb.c"] != C_SHA256):
            raise ValueError("Runtime provenance does not match the pinned reference")
        for name, expected in self.manifest["files_sha256"].items():
            if Path(name).name != name:
                raise ValueError("Invalid runtime manifest filename")
            require_hash(runtime / name, expected)
        self.lib = ct.CDLL(str(runtime / "librwkv_emb.so"))
        self.F = ct.POINTER(ct.c_float)
        self.I = ct.POINTER(ct.c_int32)
        self.U = ct.POINTER(ct.c_uint8)
        signatures = {
            "rwkv_threads": ([ct.c_int], None),
            "rwkv_load": ([ct.c_char_p], ct.c_void_p),
            "rwkv_free": ([ct.c_void_p], None),
            "rwkv_channels": ([ct.c_void_p], ct.c_int),
            "rwkv_layers": ([ct.c_void_p], ct.c_int),
            "rwkv_load_tokenizer": ([ct.c_void_p, ct.c_char_p], ct.c_int),
            "rwkv_tokenize": ([ct.c_void_p, ct.c_char_p, self.I, ct.c_int], ct.c_int),
            "rwkv_prepare": ([ct.c_void_p, ct.POINTER(ct.c_char_p), ct.c_int, ct.c_int,
                              ct.c_int, ct.POINTER(self.I), ct.POINTER(self.U)], ct.c_int),
            "rwkv_release": ([ct.c_void_p], None),
            "rwkv_encode_tokens": ([ct.c_void_p, self.I, self.U, ct.c_int, ct.c_int,
                                    ct.c_int, self.F, self.F], ct.c_int),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = argtypes, restype
        self.lib.rwkv_threads(threads)
        self.handle = self.lib.rwkv_load(os.fsencode(runtime / "embedding-fp32.bin"))
        if not self.handle:
            raise RuntimeError("C checkpoint loading failed")
        try:
            if self.lib.rwkv_load_tokenizer(self.handle, os.fsencode(runtime / "tokenizer.bin")):
                raise RuntimeError("C tokenizer loading failed")
            self.channels = self.lib.rwkv_channels(self.handle)
            self.layers = self.lib.rwkv_layers(self.handle)
            if (self.layers, self.channels) != (12, 768):
                raise ValueError("Loaded runtime has unexpected dimensions")
        except Exception:
            self.close()
            raise

    def close(self):
        if self.handle:
            self.lib.rwkv_free(self.handle)
            self.handle = None

    def tokenize(self, text):
        data = text.encode("utf-8")
        out = self.np.empty(len(data) + 1, self.np.int32)
        count = self.lib.rwkv_tokenize(self.handle, data, out.ctypes.data_as(self.I), len(out))
        if count < 0:
            raise RuntimeError("C tokenization failed")
        return out[:count].copy()

    def prepare_batch(self, texts):
        array = (ct.c_char_p * len(texts))(*(text.encode("utf-8") for text in texts))
        ids, mask = self.I(), self.U()
        length = self.lib.rwkv_prepare(self.handle, array, len(texts), CONTEXT, EOS_CHUNK,
                                       ct.byref(ids), ct.byref(mask))
        if length < 0:
            raise RuntimeError("C preprocessing failed")
        try:
            shape = (len(texts), length)
            tokens = self.np.ctypeslib.as_array(ids, (len(texts) * length,)).copy().reshape(shape)
            eos_mask = self.np.ctypeslib.as_array(mask, (len(texts) * length,)).copy().reshape(shape)
        finally:
            self.lib.rwkv_release(ids)
            self.lib.rwkv_release(mask)
        if length % 16 or (eos_mask.sum(1) == 0).any() or (tokens[eos_mask != 0] != EOS).any():
            raise ValueError("Unexpected EOS mask/alignment")
        return tokens, eos_mask

    def encode(self, ids, mask, trace):
        np = self.np
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        if ids.ndim != 2 or ids.shape != mask.shape:
            raise ValueError("Token IDs and EOS mask must have the same [B,T] shape")
        batch, length = ids.shape
        out = np.empty((batch, self.channels), np.float32)
        traces = np.empty((self.layers + 2, batch, length, self.channels), np.float32) if trace else None
        status = self.lib.rwkv_encode_tokens(self.handle, ids.ctypes.data_as(self.I),
                                            mask.ctypes.data_as(self.U), batch, length, 2,
                                            out.ctypes.data_as(self.F),
                                            traces.ctypes.data_as(self.F) if trace else None)
        if status:
            raise RuntimeError(f"C inference failed with status {status}")
        if not np.isfinite(out).all() or (traces is not None and not np.isfinite(traces).all()):
            raise ValueError("Nonfinite reference output")
        if not np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-4, rtol=0):
            raise ValueError("Reference embeddings are not normalized")
        return out, traces


def load_cases(path):
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError("Expected smoke case schema version 1")
    cases, seen = [], set()
    for row in document["cases"]:
        key, role, text = row["id"], row["role"], row["text"]
        repeat = row.get("repeat", 1)
        minimum = row.get("minimum_raw_tokens", 0)
        if (not isinstance(key, str) or not key or key in seen or
                role not in ("query", "document") or not isinstance(text, str) or
                "\0" in text or type(repeat) is not int or not 1 <= repeat <= 1024 or
                type(minimum) is not int or not 0 <= minimum <= CONTEXT * 8):
            raise ValueError("Invalid or duplicate smoke case")
        seen.add(key)
        text *= repeat
        encoded_text = INSTRUCTION.format(query=text) if role == "query" else text
        if len(encoded_text.encode("utf-8")) > 100_000:
            raise ValueError("Smoke input is too large")
        cases.append({"id": key, "role": role, "text": text, "encoded_text": encoded_text,
                      "minimum_raw_tokens": minimum})
    if len(cases) < 2:
        raise ValueError("At least two cases are required for repeat-call isolation")
    return cases


def smoke(args):
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import numpy as np

    cases = load_cases(args.cases.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model = None
    report = {"schema_version": 1, "backend": "upstream C FP32 CPU", "dtype": "float32",
              "device": "cpu", "threads": args.threads, "batch_size": args.batch_size,
              "context": CONTEXT, "eos_chunk_size": EOS_CHUNK, "head": "RETR",
              "query_instruction": INSTRUCTION, "cases_sha256": sha256(args.cases),
              "numpy": np.__version__, "trace_kind": "layer hidden outputs, not recurrent matrices",
              "trace_labels": ["embedding", *[f"block_{i}" for i in range(12)], "ln_out"],
              "batches": [], "all_checks_passed": False, **host_info()}
    try:
        model = CReference(args.runtime.resolve(), args.threads)
        report["setup_seconds"] = time.perf_counter() - started
        report["runtime_manifest"] = model.manifest
        first = None
        for offset in range(0, len(cases), args.batch_size):
            batch = cases[offset:offset + args.batch_size]
            raw = [model.tokenize(row["encoded_text"]) for row in batch]
            for row, tokens in zip(batch, raw):
                if len(tokens) < row["minimum_raw_tokens"] or len(tokens) > CONTEXT * 8:
                    raise ValueError(f"Unexpected raw token count for {row['id']}: {len(tokens)}")
            ids, mask = model.prepare_batch([row["encoded_text"] for row in batch])
            trace_bytes = (model.layers + 2) * ids.size * model.channels * 4
            if trace_bytes > args.max_trace_mib * 1024 ** 2:
                raise ValueError("Trace exceeds memory limit; reduce batch size")
            before = time.perf_counter()
            embeddings, traces = model.encode(ids, mask, trace=True)
            seconds = time.perf_counter() - before
            filename = f"batch_{offset // args.batch_size:03d}.npz"
            arrays = {"input_ids": ids, "eos_mask": mask, "embeddings": embeddings, "layer_outputs": traces}
            arrays.update({f"raw_tokens_{i}": value for i, value in enumerate(raw)})
            np.savez_compressed(output / filename, **arrays)
            report["batches"].append({"cases": batch, "raw_token_counts": [len(x) for x in raw],
                                      "input_shape": list(ids.shape), "selected_eos_counts": mask.sum(1).tolist(),
                                      "embedding_norms": np.linalg.norm(embeddings, axis=1).tolist(),
                                      "model_seconds": seconds, "artifact": filename,
                                      "artifact_sha256": sha256(output / filename)})
            if first is None:
                first = (ids.copy(), mask.copy(), embeddings.copy())
            print(f"Saved {filename}: {[row['id'] for row in batch]}, {seconds:.3f}s", flush=True)
        before = time.perf_counter()
        repeated, _ = model.encode(first[0], first[1], trace=False)
        report["repeat_check_seconds"] = time.perf_counter() - before
        report["repeat_call_bitwise_equal"] = bool(np.array_equal(repeated, first[2]))
        if not report["repeat_call_bitwise_equal"]:
            raise ValueError("First batch changed after intervening calls")
        report["all_checks_passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if model is not None:
            model.close()
        report["total_seconds"] = time.perf_counter() - started
        write_json(output / "result.json", report)
    print(json.dumps({"output": str(output), "all_checks_passed": True,
                      "batches": len(report["batches"])}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare", help="Export weights and build the C library; no inference")
    setup.add_argument("--upstream", type=Path, required=True)
    setup.add_argument("--checkpoint", type=Path, required=True)
    setup.add_argument("--runtime", type=Path, required=True, help="New directory; refuses overwrite")
    setup.add_argument("--compiler", default="gcc")
    setup.set_defaults(function=prepare)
    run = commands.add_parser("smoke", help="Save CPU FP32 comparison artifacts for fixed cases")
    run.add_argument("--runtime", type=Path, required=True)
    run.add_argument("--cases", type=Path, default=Path(__file__).parent / "data/smoke_cases.json")
    run.add_argument("--output", type=Path, required=True, help="New directory; refuses overwrite")
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--max-trace-mib", type=int, default=128)
    run.set_defaults(function=smoke)
    args = parser.parse_args()
    if args.command == "smoke" and not (1 <= args.threads <= 64 and 1 <= args.batch_size <= 4 and
                                        1 <= args.max_trace_mib <= 1024):
        parser.error("Use threads 1..64, batch size 1..4 and max trace MiB 1..1024")
    args.function(args)


if __name__ == "__main__":
    main()
