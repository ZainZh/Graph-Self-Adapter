"""Script-only scheduling and result-free per-task freezes; no live learner agent."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tools.fast_adapter.p28_common import R, TASKS, SEEDS, task_root, save_json, digest
from tools.fast_adapter.p28_eval import EVAL_STRATA


def jobs_run(jobs):
    pending = list(jobs); active = {}; outcomes = []
    while pending or active:
        for gpu in range(3):
            if gpu in active or not pending:
                continue
            name, module, args = pending.pop(0)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                       PYTHONPATH='data/runtime/fast-adapter-p28-deps:.',
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                       HF_HUB_DISABLE_PROGRESS_BARS='1', TRANSFORMERS_NO_ADVISORY_WARNINGS='1')
            log = (R/'logs'/f'{name}.log').open('x')
            process = subprocess.Popen([sys.executable, '-u', '-m', module, *args],
                        stdout=log, stderr=subprocess.STDOUT, env=env)
            active[gpu] = (name, process, log)
            print('START', name, 'gpu', gpu, 'pid', process.pid, flush=True)
        for gpu, (name, process, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close(); del active[gpu]
            outcomes.append({'job': name, 'gpu': gpu, 'returncode': code})
            print('END', name, code, flush=True)
            if code:
                # Let existing jobs close cleanly; never kill an unrelated service.
                for _, other, stream in active.values():
                    other.wait(); stream.close()
                save_json(R/'FAILED_JOBS.json', {'outcomes': outcomes})
                raise RuntimeError((name, code))
        if pending or active:
            time.sleep(3)
    return outcomes


def main():
    began = time.time()
    for b in json.loads((R/'EVALUATION_SOURCE_FREEZE.json').read_text())['bindings']:
        assert digest(b['path']) == b['sha256'], b['path']
    preparation = json.loads((R/'PREPARATION_LAUNCH.json').read_text())['rows']
    while not all((task_root(t)/'BASE_READY.json').exists() for t in TASKS):
        for row in preparation:
            if (task_root(row['task'])/'BASE_READY.json').exists():
                continue
            try:
                os.kill(row['pid'], 0)
            except ProcessLookupError:
                raise RuntimeError('Preparation exited without readiness: '+row['task'])
        if time.time()-began > 6*3600:
            raise TimeoutError('Preparation wait exceeded6h')
        time.sleep(5)
    jobs_run([(f'base_qualify_{t}', 'tools.fast_adapter.p28_eval', ['qualify', '--task', t]) for t in TASKS])
    qualified = [t for t in TASKS if json.loads((task_root(t)/'qualification/BASE.json').read_text())['passed']]
    failed = [t for t in TASKS if t not in qualified]
    save_json(R/'BASE_QUALIFICATION_GATE.json', {'qualified': qualified, 'unready': failed,
              'rule': 'Fixed final base at least8/16 native successes; unready tasks retained in report'})
    jobs_run([(f'meta_data_{t}', 'tools.fast_adapter.p28_data',
               ['collect', '--stage', 'meta', '--task', t]) for t in qualified])
    jobs_run([(f'train_{t}_seed{s}', 'tools.fast_adapter.p28_train',
               ['adapt', '--task', t, '--seed', str(s)]) for t in qualified for s in SEEDS])
    for t in qualified:
        root = task_root(t)
        paths = [root/'models/base_final.pt', root/'data/PCA.pt', R/'PROTOCOL.json',
                 R/'EVALUATION_AMENDMENT.json', R/'EVALUATION_SOURCE_FREEZE.json']
        paths += [root/'models'/f'seed{s}'/f'{variant}.pt' for s in SEEDS for variant in ('fast', 'history')]
        save_json(root/'MODEL_FREEZE.json', {'time': time.time(), 'independent_results_not_generated': True,
                  'selection': 'Fixed final checkpoint only; no best-seed/K/checkpoint selection',
                  'bindings': [{'path': str(p), 'sha256': digest(p)} for p in paths]})
    jobs_run([(f'final_{t}_{st}', 'tools.fast_adapter.p28_eval',
               ['evaluate', '--task', t, '--stratum', st]) for t in qualified for st in EVAL_STRATA])
    save_json(R/'CAMPAIGN_DONE.json', {'time': time.time(), 'seconds': time.time()-began,
              'status': 'all_qualified_tasks_complete_pending_batch_audit',
              'qualified_tasks': qualified, 'base_unready_tasks': failed,
              'query_episodes': len(qualified)*3*64*9,
              'support_episodes': len(qualified)*3*64*2,
              'query_commands': len(qualified)*3*64*9*160,
              'support_commands': len(qualified)*3*64*2*80})
    print('CAMPAIGN_COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save_json(R/'CAMPAIGN_ERROR.json', {'time': time.time(), 'error': repr(exc)})
        raise
