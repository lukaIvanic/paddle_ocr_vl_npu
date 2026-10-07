"""Read-only local environment/asset inventory; no imports of torch or downloads.

Uses standard-library Python. Candidate interpreters report installed package
metadata in short subprocesses. Nothing activates an environment or selects a
device automatically. Explicit --root arguments replace the default search roots.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


PACKAGES = ('torch','torch-npu','torchair','transformers','tokenizers',
            'huggingface-hub','pyarrow','numpy','Pillow','safetensors','torchvision',
            'pytrec-eval','pytrec-eval-terrier')


def probe():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.executable, prefix=sys.prefix, version=sys.version,
                packages=versions, torchair_top_level=importlib.util.find_spec('torchair') is not None)


def inventory(roots, max_seconds=45, max_dirs=20000):
    result = dict(search_roots=[str(p) for p in roots], python_candidates=[],
                  model_candidates=[], dataset_candidates=[], setup_scripts=[],
                  search_errors=[], truncated=False)
    interpreters = {shutil.which(name) for name in ('python3','python','python3.12','python3.11','python3.10')}
    interpreters.discard(None)
    seen = set()
    started = time.monotonic()
    count = 0
    skip = {'.git','node_modules','__pycache__','site-packages','dist-packages',
            '.runtime_cache','proc','sys','dev'}
    for root in roots:
        if not root.is_dir():
            continue
        def onerror(error):
            if len(result['search_errors'])<30:
                result['search_errors'].append(str(error))
        for directory, dirs, files in os.walk(root, onerror=onerror, followlinks=False):
            path = Path(directory)
            key = str(path.absolute())
            if key in seen:
                dirs[:]=[]
                continue
            seen.add(key)
            count += 1
            if count>max_dirs or time.monotonic()-started>max_seconds:
                result['truncated']=True
                break
            depth = len(path.relative_to(root).parts)
            dirs[:]=[d for d in dirs if d not in skip] if depth<10 else []
            if path.name=='bin':
                for name in ('python','python3','python3.12','python3.11','python3.10'):
                    candidate=path/name
                    if candidate.is_file() and os.access(candidate,os.X_OK):
                        # Preserve the venv path; resolve() would erase it.
                        interpreters.add(str(candidate.absolute()))
            for name in ('npu-setup','set_env.sh'):
                if name in files:
                    result['setup_scripts'].append(str(path/name))
            if 'config.json' in files:
                try:
                    config_path=path/'config.json'
                    if config_path.stat().st_size<100000:
                        config=json.loads(config_path.read_text())
                        if isinstance(config,dict) and config.get('model_type')=='ops_colqwen3':
                            result['model_candidates'].append(str(path))
                except (OSError, ValueError, UnicodeError):
                    pass
            if path.name=='english-corpus' and 'test-00000-of-00001.parquet' in files:
                result['dataset_candidates'].append(str(path.parent))
        if result['truncated']:
            break
    npu_setup=shutil.which('npu-setup')
    if npu_setup:
        result['setup_scripts'].append(npu_setup)
    result['python_candidates']=sorted(interpreters)
    result['setup_scripts']=sorted(set(result['setup_scripts']))
    result['directories_scanned']=count
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,action='append')
    parser.add_argument('--probe-python',action='store_true')
    args=parser.parse_args()
    if args.probe_python:
        print(json.dumps(probe()))
        return
    roots=args.root or [Path.cwd(),Path.home(),Path('/workspace'),Path('/opt'),
                       Path('/data'),Path('/models'),Path('/datasets'),Path('/usr/local')]
    result=inventory(roots)
    result.update(host=platform.node(),platform=platform.platform(),
                  environment={k:os.environ.get(k) for k in ('VIRTUAL_ENV','CONDA_PREFIX',
                    'ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME','ASCEND_RT_VISIBLE_DEVICES')},
                  interpreters=[])
    prefixes=set()
    if len(result['python_candidates'])>60:
        result['interpreter_probe_truncated']=True
    for candidate in result['python_candidates'][:60]:
        try:
            process=subprocess.run([candidate,str(Path(__file__).absolute()),'--probe-python'],
                capture_output=True,text=True,timeout=15)
            if process.returncode:
                result['interpreters'].append(dict(candidate=candidate,error=process.stderr[-2000:]))
                continue
            row=json.loads(process.stdout)
            identity=(row['prefix'],row['version'])
            if identity in prefixes:
                continue
            prefixes.add(identity)
            row['candidate']=candidate
            result['interpreters'].append(row)
        except (OSError,ValueError,subprocess.TimeoutExpired) as error:
            result['interpreters'].append(dict(candidate=candidate,error=str(error)))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
