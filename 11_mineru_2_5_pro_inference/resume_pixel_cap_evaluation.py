"""Resume scoring completed paired predictions after a runtime-preflight failure."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from run_pixel_cap_ablation import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)
    root = args.run_root.resolve()
    selected = json.loads((root / 'selection.json').read_text())['page_count']
    for label in ['original', 'capped']:
        run = root / label
        assert (run / 'exit_code.txt').read_text().strip() == '0'
        s = json.loads((run / 'output/run_summary_shard_00.json').read_text())
        assert s['completed'] == selected and s['failed'] == 0
    (root / 'evaluation_resume_commit.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True))
    code = 1
    try:
        for label in ['original', 'capped']:
            run = root / label
            evaluation = run / 'evaluation'
            if evaluation.exists():
                # Only the preflight-failure case is supported: no scoring started.
                if (evaluation / 'run.log').exists() or (evaluation / 'exit_code.txt').exists():
                    raise ValueError('existing evaluator execution; inspect before resuming')
                for name in ['evaluation', 'evaluation_input_adapter', 'evaluation_launcher.log',
                             'evaluation_launcher.log.command', 'evaluation_launcher.log.exit_code',
                             'evaluation_launcher.log.wall_s']:
                    old = run / name
                    if old.exists():
                        preserved = run / (name + '.failed_preflight')
                        if preserved.exists():
                            raise FileExistsError(preserved)
                        old.rename(preserved)
            os.environ.update(RUN_ROOT=str(run), DATASET_JSON=str(root / 'OmniDocBench_affected.json'), LIMIT=str(selected))
            print(f'EVALUATION_START {label}', flush=True)
            execute(['bash', '11_mineru_2_5_pro_inference/run_serving_accuracy.sh'], run, 'evaluation_launcher.log')
            print(f'EVALUATION_FINISH {label}', flush=True)
        code = 0
    finally:
        (root / 'evaluation_resume_exit_code.txt').write_text(str(code) + '\n')
        if code == 0:
            # Preserve the failed chain's status before marking the resumed chain complete.
            previous = root / 'exit_code.txt'
            previous.rename(root / 'initial_chain_exit_code.txt')
            previous.write_text('0\n')


if __name__ == '__main__':
    main()
