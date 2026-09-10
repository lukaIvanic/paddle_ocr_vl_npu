"""Step-2 mechanical relocation receipt; CPU-only, never imports model code.

Normally verifies the copy against the pinned Git source. --materialize performs
the one-time bulk copy/import rewrite. This is migration tooling, not serving.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "19_table_ocr_serving"
PIN = "be691de190ae099d1a9b0ba80865006b122ecc00"
PREFIX = "09_persistent_page_engine/"
CORE = {
    "scripts.serve_crop_ocr_api": "serve",
    "paddleocr_vl.serving.engine": "serving_runtime",
    "paddleocr_vl.model.modeling": "paddle_ocr_vl_1_6_modeling",
    "paddleocr_vl.model.vision_prefill": "vision_prefill",
    "paddleocr_vl.model.text_decode": "text_prefill_and_decode",
    "paddleocr_vl.model.text_prefill": "text_prefill_and_decode",
    "paddleocr_vl.model.preprocessing": "crop_processing",
}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def module_path(module):
    if module in CORE:
        return CORE[module]
    if module == "paddleocr_vl" or module.startswith("paddleocr_vl."):
        return "_support" + module[len("paddleocr_vl"):]
    if module.split(".")[0] in {"utils", "pipeline"}:
        return "_support." + module
    return module


def absolute_import(node, module, is_package):
    if node.level:
        package = module if is_package else module.rpartition(".")[0]
        return importlib.util.resolve_name("." * node.level + (node.module or ""), package)
    return node.module or ""


def inventory():
    files = git("ls-tree", "-r", "--name-only", PIN, PREFIX).decode().splitlines()
    modules = {}
    for path in files:
        if not path.endswith(".py"):
            continue
        relative = path[len(PREFIX):-3]
        package = relative.endswith("/__init__")
        name = relative[:-9] if package else relative
        modules[name.replace("/", ".")] = (path, package)
    return modules


def sources():
    modules = inventory()
    pending = ["scripts.serve_crop_ocr_api"]
    found = {}
    while pending:
        name = pending.pop()
        if name in found or name not in modules:
            continue
        path, package = modules[name]
        source = git("show", f"{PIN}:{path}").decode()
        found[name] = (path, package, source)
        # Preserve package initialization as well as all statically imported
        # alternatives, even when they are not selected in the flagship lane.
        parts = name.split(".")
        pending.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                base = absolute_import(node, name, package)
                pending.append(base)
                pending.extend(base + "." + item.name for item in node.names)
            elif isinstance(node, ast.Import):
                pending.extend(item.name for item in node.names)
    return found


def relocate(name, package, source):
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    changes = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.ImportFrom, ast.Import)):
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        if isinstance(node, ast.ImportFrom):
            base = absolute_import(node, name, package)
            target = module_path(base)
            if target == module_path(name):
                replacement = "pass  # Definition is now in this same module."
            elif node.module == "__future__" and name.endswith(".text_prefill"):
                replacement = "# Future annotations are enabled at the combined module's start."
            else:
                items = []
                for item in node.names:
                    old = item.name
                    new = "_prefill_linear_tokenwise" if base.endswith(".text_prefill") and old == "_linear_tokenwise" else old
                    alias = item.asname or (old if new != old else None)
                    items.append(new + (" as " + alias if alias else ""))
                replacement = "from " + target + " import " + ", ".join(items)
        else:
            # Preserve binding names for 'import x as y'. No unaliased dotted
            # project-module import may be rewritten by guessing its usage.
            items = []
            for item in node.names:
                target = module_path(item.name)
                if target != item.name and "." in item.name and not item.asname:
                    raise AssertionError((name, "unaliased dotted import", item.name))
                alias = item.asname or (item.name if target != item.name else None)
                items.append(target + (" as " + alias if alias else ""))
            replacement = "import " + ", ".join(items)
        changes.append((start, end, replacement))
    for start, end, replacement in sorted(changes, reverse=True):
        source = source[:start] + replacement + source[end:]
    if name.endswith(".text_prefill"):
        source = re.sub(r"\b_linear_tokenwise\b", "_prefill_linear_tokenwise", source)
    if name == "scripts.serve_crop_ocr_api":
        source = source.replace("EXPERIMENT_ROOT = HERE.parent", "EXPERIMENT_ROOT = HERE")
        # Storage namespaces only; the benchmark will explicitly pass these.
        source = source.replace(".runtime_cache/09_persistent_page_engine", ".runtime_cache/19_table_ocr_serving")
    if name.endswith(".text_decode"):
        # The compiler's source fingerprint must hash the relocated files;
        # do not spoof old cache keys to avoid compilation.
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decode_source_hash")
        segment = ast.get_source_segment(source, fn)
        replacement = segment
        for item in ast.walk(fn):
            if isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value.endswith(".py"):
                target = "text_prefill_and_decode.py" if item.value in {"text_decode.py", "text_prefill.py"} else "_support/model/" + item.value
                replacement = replacement.replace('"' + item.value + '"', '"' + target + '"')
        source = source.replace(segment, replacement)
    return source


def expected():
    found = sources()
    output = {}
    receipt = []
    # Decode definitions must precede the prefill classes that reference them.
    order = sorted(found, key=lambda n: (n == "paddleocr_vl.model.text_prefill", n))
    for name in order:
        path, package, source = found[name]
        new = module_path(name).replace(".", "/") + ("/__init__.py" if package else ".py")
        content = relocate(name, package, source)
        if new in output:
            content = output[new] + "\n\n# ---- Relocated text-prefill implementation (unchanged computation) ----\n\n" + content
        output[new] = content
        receipt.append({"source": path, "source_blob": git("rev-parse", f"{PIN}:{path}").decode().strip(), "destination": new})
    # Package parent for the separately owned pipeline and timing modules.
    output.setdefault("_support/__init__.py", '"""Temporary mechanical support files retained for step-2 parity."""\n')
    for path in ["presets/table_compact_vocab/b1_verifier_topfreq_16384.json", "utils/timeline_viewer.html"]:
        new = path if path.startswith("presets/") else "_support/" + path
        output[new] = git("show", f"{PIN}:{PREFIX}{path}").decode()
    receipt_data = {"source_commit": PIN, "phase": "mechanical relocation, before cleanup", "sources": receipt,
                    "destination_sha256": {p: hashlib.sha256(s.encode()).hexdigest() for p,s in sorted(output.items())}}
    output["relocation.json"] = json.dumps(receipt_data, indent=2) + "\n"
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    output = expected()
    for relative, content in output.items():
        path = DEST / relative
        if relative.endswith(".py"):
            ast.parse(content, filename=relative)
        if args.materialize:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        else:
            assert path.read_text() == content, f"Non-mechanical difference: {relative}"
    # Independently compare each function/class body, ignoring only import
    # locations, the prefill helper rename, and relocated cache-hash paths.
    class Normalize(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return ast.Pass()

        def visit_Name(self, node):
            if node.id == "_prefill_linear_tokenwise":
                node.id = "_linear_tokenwise"
            return node

        def visit_FunctionDef(self, node):
            if node.name == "_prefill_linear_tokenwise":
                node.name = "_linear_tokenwise"
            return self.generic_visit(node)

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = node.value.replace(".runtime_cache/19_table_ocr_serving", ".runtime_cache/09_persistent_page_engine")
            return node

    checked = 0
    for name, (_, package, source) in sources().items():
        dest = module_path(name).replace(".", "/") + ("/__init__.py" if package else ".py")
        relocated = ast.parse(output[dest])
        definitions = {n.name: n for n in relocated.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for original in ast.parse(source).body:
            if not isinstance(original, (ast.FunctionDef, ast.ClassDef)):
                continue
            if original.name == "decode_source_hash":
                continue  # Exact approved path rewrite checked above.
            lookup = "_prefill_linear_tokenwise" if name.endswith(".text_prefill") and original.name == "_linear_tokenwise" else original.name
            copied = definitions[lookup]
            assert ast.dump(Normalize().visit(original)) == ast.dump(Normalize().visit(copied)), (name, lookup, "executable body changed")
            checked += 1
    print(f"PASS: {checked} function/class bodies unchanged apart from imports and approved naming/storage changes.")
    print(f"PASS: {len(output)} files match pinned-source mechanical relocation; all Python parses.")


if __name__ == "__main__":
    main()
