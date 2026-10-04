"""Finite queue for already sealed final tasks; no model selection or mutation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tools.fast_adapter.p29_common import R, TASKS, digest, save_json, task_root
from tools.fast_adapter.p29_eval import STRATA


def run(tasks, gpus, dino, tag):
    jobs = {}; ended = set(); logs = R/'logs'; began = time.time()
    receipt = logs/(tag+'_START.json')
    assert not receipt.exists()
    save_json(receipt, {'time': began, 'tasks': tasks, 'gpus': gpus, 'pid': os.getpid(), 'source_sha256': digest(__file__), 'only_existing_final_freezes': True})
    while time.time()-began < 4*3600:
        for key, value in list(jobs.items()):
            if value['p'].poll() is not None:
                code = value['p'].returncode; value['f'].close(); del jobs[key]; ended.add(key)
                save_json(logs/f'{tag}_{key}_END.json', {'time': time.time(), 'returncode': code, 'gpu': value['gpu']})
                print('END', key, code, flush=True)
                if code != 0:
                    raise RuntimeError(f'{key} failed; preserve partial artifacts, do not automatically restart')
        busy = {v['slot'] for v in jobs.values()}
        pending = []
        for task in tasks:
            freeze = task_root(task)/'FINAL_FREEZE.json'
            if not freeze.exists(): continue
            cfg = json.loads(freeze.read_text()); rnd = cfg['round']
            for st in STRATA:
                key = task+'_'+st
                if not (task_root(task)/'final'/rnd/st/'RESULT.json').exists() and key not in jobs and key not in ended:
                    assert not (task_root(task)/'final'/rnd/st).exists(), 'Unowned partial final directory; inspect'
                    pending.append((key, cfg['module_command_without_stratum']+['--stratum', st, '--dino', dino]))
        for slot, gpu in enumerate(gpus):
            if slot in busy or not pending: continue
            key, args = pending.pop(0); env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH='data/runtime/fast-adapter-p28-deps:.', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='2', HF_HUB_DISABLE_PROGRESS_BARS='1')
            f = (logs/f'{tag}_{key}.log').open('x')
            p = subprocess.Popen([sys.executable, '-m', *args], env=env, stdout=f, stderr=subprocess.STDOUT)
            jobs[key] = {'p': p, 'f': f, 'gpu': gpu, 'slot': slot}; print('START', key, gpu, p.pid, flush=True)
        complete = True
        for task in tasks:
            fp = task_root(task)/'FINAL_FREEZE.json'
            if not fp.exists(): complete=False; continue
            rnd = json.loads(fp.read_text())['round']
            complete &= all((task_root(task)/'final'/rnd/st/'RESULT.json').exists() for st in STRATA)
        if complete and not jobs:
            save_json(logs/(tag+'_DONE.json'), {'time': time.time(), 'seconds': time.time()-began, 'tasks': tasks})
            print('DONE', flush=True); return
        time.sleep(10)
    save_json(logs/(tag+'_TIMEOUT.json'), {'time': time.time(), 'active': list(jobs)})
    raise RuntimeError('Queue timeout; inspect active workers')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',nargs='+',choices=TASKS,required=True)
    p.add_argument('--gpus',nargs='+',type=int,required=True);p.add_argument('--dino',default='assets/dinov3');p.add_argument('--tag',required=True);a=p.parse_args()
    run(a.tasks,a.gpus,a.dino,a.tag)
