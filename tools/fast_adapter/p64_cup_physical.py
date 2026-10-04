"""Prepare cup source/model/asset closure and execute an explicitly reviewed queue.

Drafting performs no rendering, integration, inference or training. Queue mode
requires the separate exact root approval consumed by the single-cell runner.
"""
import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prove.fast_adapter.p61_data import sha256

ROOT=Path(__file__).resolve().parents[2]


def source_closure(roots):
    seen=set();pending=[ROOT/name for name in roots]
    def resolve(name):
        path=ROOT/name.replace('.','/')
        return path.with_suffix('.py') if path.with_suffix('.py').is_file() else path/'__init__.py'
    while pending:
        path=pending.pop().resolve()
        if path in seen or not path.is_file() or not path.is_relative_to(ROOT):continue
        seen.add(path)
        relative=path.relative_to(ROOT)
        # Python executes package initializers along the import path too.
        for parent in path.parents:
            if parent==ROOT:break
            init=parent/'__init__.py'
            if init.is_file():pending.append(init)
        package='.'.join(relative.parent.parts)
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[item.name for item in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    base=package.split('.')[:len(package.split('.'))-node.level+1]
                    name='.'.join(base+([node.module] if node.module else []))
                else:name=node.module or ''
                names=[name]+[name+'.'+item.name for item in node.names]
            pending.extend(resolve(name) for name in names if name.startswith(('prove.','tools.')))
    return sorted(seen)


def prepare(output,encoder,action_policy,dino_model,*,draft=True):
    from prove.fast_adapter.robot_geometry_env_p45_v02 import source_assets,panda_assets
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    roots=['tools/fast_adapter/p64_cup_evaluate.py','tools/fast_adapter/p64_cup_physical.py',
           'tools/fast_adapter/p64_cup_report.py','tests/test_p64_cup.py','tests/test_p64_cup_report.py']
    sources=source_closure(roots)
    train=Path(encoder).parent
    models=[Path(encoder),Path(action_policy),Path(dino_model)/'model.safetensors',
        Path(dino_model)/'config.json',Path(dino_model)/'preprocessor_config.json',
        Path('data/sources/lila-wam-reference-20260923/VISUAL_WEIGHTS.pt'),
        Path('data/sources/lila-wam-reference-20260923/models/vla_model_fm.py'),
        train/'TRAINING_FREEZE.json',train/'ROOT_TRAINING_REVIEW.json',
        train.parent/'cache_v01/MANIFEST.json']
    assets=source_assets();asset_root=panda_assets()
    files={str(p.relative_to(ROOT)):sha256(p) for p in sources}
    missing=[]
    for path in models:
        if path.is_file():files[str(path)]=sha256(path)
        else:missing.append(str(path))
    for name,digest in assets.items():files[str(asset_root/name)]=digest
    if missing and not draft:raise FileNotFoundError('Missing freeze assets: '+str(missing))
    cells=[dict(split=split,seed=seed,arm=arm) for split in ('id','size','position')
        for seed in range(640100,640104) for arm in ('fixed','translation_size5')]
    value=dict(schema='prove.p64-cup-physical-freeze.v1',status='draft' if draft else 'frozen',
        result_state='before_physical_evaluation',steps=240,cells=cells,automatic_retries=0,
        encoder=str(encoder),action_policy=str(action_policy),dino_model=str(dino_model),files=files,
        missing_assets=missing,source_closure_count=len(sources),panda_asset_root=str(asset_root),
        packages={name:importlib.metadata.version(name) for name in ('torch','numpy','mujoco','transformers')},
        runtime_inputs=['native448 RGB','visible robot mask','visible cup-body/support mask','visible handle-front mask',
            'known camera calibration','own EEF pose','completed public history'],
        orientation=dict(yaw_degrees=0,world_rotation='Rx90deg',rotation_latent_updates=False),
        success='original native bilateral handle contact and lift >.04m in all10 terminal states, AND mechanical validity',
        mechanical=dict(full_trajectory_penetration_m_strict_lt=.002,arm_limit_violation_rad_le=.001,
            finger_limit_violation_m_le=.0001,all_states_commands_finite=True),
        latent=dict(active_indices=[0,1,2,6,7],steps_per_feedback=3,minimum_observations=3,
            freeze_at='approach-to-close transition before object lift'),
        selected_videos=[dict(split='id',seed=640100),dict(split='size',seed=640100)])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(value,indent=2)+'\n')
    return dict(path=str(output),sha256=sha256(output),missing_assets=missing,source_closure_count=len(sources))


def queue(freeze,output,device='cuda'):
    from tools.fast_adapter.p64_cup_evaluate import validate
    contract=validate(freeze);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for index,cell in enumerate(contract['cells']):
        destination=output/f"{index:02d}_{cell['split']}_{cell['seed']}_{cell['arm']}"
        command=[sys.executable,'-m','tools.fast_adapter.p64_cup_evaluate','--freeze',str(freeze),
            '--index',str(index),'--output',str(destination),'--device',device]
        environment=dict(os.environ,MUJOCO_GL='egl',EGL_DEVICE_ID='0',OMP_NUM_THREADS='2')
        started=time.monotonic()
        with (output/f'{index:02d}.log').open('w') as log:
            completed=subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
        record=dict(index=index,cell=cell,exit_code=completed.returncode,elapsed_seconds=time.monotonic()-started,
            output=str(destination),automatic_retries=0)
        records.append(record)
        (output/'QUEUE_PROGRESS.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        if completed.returncode:
            # Stop for review; never change code or silently retry within a seal.
            raise RuntimeError('Cup cell program error; retained log '+str(output/f'{index:02d}.log'))
    (output/'QUEUE_COMPLETION.json').write_text(json.dumps(dict(completed=len(records),records=records),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    for name in ('draft','freeze'):
        p=sub.add_parser(name);p.add_argument('--output',required=True);p.add_argument('--encoder',required=True)
        p.add_argument('--action-policy',required=True);p.add_argument('--dino-model',required=True)
    q=sub.add_parser('queue');q.add_argument('--freeze',required=True);q.add_argument('--output',required=True);q.add_argument('--device',default='cuda')
    a=parser.parse_args()
    if a.command=='queue':queue(a.freeze,a.output,a.device)
    else:print(json.dumps(prepare(a.output,a.encoder,a.action_policy,a.dino_model,draft=a.command=='draft'),indent=2))
