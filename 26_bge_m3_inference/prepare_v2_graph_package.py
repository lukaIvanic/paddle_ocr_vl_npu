"""Private graph-test package: V2 dispatch only, plus missing static inference metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--vendor", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann"))
a = p.parse_args()
shutil.copytree(a.vendor, a.output, symlinks=True)  # Fails if destination exists.
registrations = {}
for config in (a.output / "op_impl/ai_core/tbe/config").glob("*/*.json"):
    data = json.loads(config.read_text())
    registrations[str(config.relative_to(a.output))] = list(data)
    config.write_text(json.dumps({k: v for k, v in data.items() if k == "AddLayerNormQuantV2"}, indent=2))
# Keep headers/dependencies and their libraries, but do not override stock kernel dispatch.
source = Path(__file__).with_name("v2_graph_infer.cpp")
lib = a.output / "libbge_v2_graph_infer.so"
subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2", str(source),
                "-I" + str(a.cann / "include"), "-L" + str(a.cann / "lib64"),
                "-lexe_graph", "-lregister", "-lopp_registry", "-Wl,--no-undefined", "-o", str(lib)], check=True)
manifest = {"upstream_vendor": str(a.vendor), "original_dispatch_registrations": registrations,
            "enabled_dispatch": ["AddLayerNormQuantV2"],
            "metadata_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "op_api_sha256": hashlib.sha256((a.output / "op_api/lib/libcust_opapi.so").read_bytes()).hexdigest()}
(a.output / "bge_graph_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest))
