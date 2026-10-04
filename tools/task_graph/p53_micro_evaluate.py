#!/usr/bin/env python3
"""P53 prepared 6-entry × 6-arm micro evaluation. No execution during prepare.

Old/new skills share the same qualified near-entry state and new success contract.
Failed Code entries remain in system denominator as unexecuted blocked cells.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from prove.task_graph.p52_stage_policy import Config,StagePolicy,StageRuntime
from tools.task_graph.p52_policy_pilot import Vision,validate_features,initialize
from tools.task_graph.p52_full_task import InsertJudge,structural_measurement
from tools.task_graph.collect_libero_p52 import PublicRenderer
from tools.task_graph.collect_libero_demo_gt import snapshot,dump,verify as verify_environment
from tools.task_graph.p52_pack_dataset import sha
from tools.task_graph.p52_deployment import source_closure
from tools.task_graph.p53_dataset import verify_seal,validate_entry_measurement,lines

ROOT=Path(__file__).resolve().parents[2]
ARMS=('p52_bc','p52_fixed','p52_online','p53_bc','p53_fixed','p53_online')


def load_skill(path):
    package=torch.load(path,map_location='cpu',weights_only=True)
    if package['schema'] not in ('p52_stage_skill_v01','p53_short_insert_skill_v01'):raise ValueError('Unknown frozen skill checkpoint')
    model=StagePolicy(Config(**package['config']));model.load_state_dict(package['model'],strict=True)
    return model,package


def verify_entry(entry):
    if entry.get('split')!='validation' or not isinstance(entry.get('near_entry_qualified'),bool):raise ValueError('Require fixed validation identity and Code-entry qualification')
    if sha(entry['source_result'])!=entry['source_result_sha256']:raise ValueError('Code-entry result changed')
    result=json.loads(Path(entry['source_result']).read_text())
    if bool(result.get('near_entry_qualified',False))!=entry['near_entry_qualified']:raise ValueError('Entry qualification differs from actual source result')
    root=Path(entry['source_result']).parent
    actual,source_seal=verify_seal(root)
    source_binding=json.loads((root/'BINDING.json').read_text())
    if not source_binding.get('entry_contract'):raise ValueError('Missing frozen near-entry contract')
    if not entry['near_entry_qualified']:return
    if sha(entry['source_seal'])!=entry['source_seal_sha256']:raise ValueError('Near-entry seal changed')
    seal=json.loads(Path(entry['source_seal']).read_text());root=Path(entry['source_seal']).parent
    if Path(entry['source_seal']).resolve()!=(Path(entry['source_result']).parent/'SEAL.json').resolve():raise ValueError('Entry seal/result roots differ')
    first=lines(root/'private/LABELS.jsonl')[0]
    validate_entry_measurement(first,source_binding['entry_contract'])
    if first['teacher_phase'] not in source_binding['allowed_phases'] or first.get('authority')!=source_binding['teacher_authority']:raise ValueError('Entry teacher authority/phase differs from binding')
    frame=Path(entry['frame'])
    if frame.resolve()!=(root/'private'/first['private_state']).resolve():raise ValueError('Must restore actual initial micro frame')
    relative=str(frame.resolve().relative_to(root.resolve()))
    if sha(frame)!=entry['frame_sha256'] or seal['files'].get(relative)!=entry['frame_sha256']:raise ValueError('Near-entry state does not match source seal')
    with np.load(frame,allow_pickle=False) as state:
        if not all(k in state for k in ('supervisor_sim_state','supervisor_ctrl')):raise ValueError('Real predecessor state+ctrl required')


def prepare(parent_path,entries,checkpoints,vision):
    parent=json.loads(parent_path.read_text());verify_environment(parent['environment'])
    if len(entries)!=6 or len({e['name'] for e in entries})!=6:raise ValueError('Retain all six predeclared entry identities, including Code failures')
    for e in entries:verify_entry(e)
    weights={k:dict(path=str(Path(v).resolve()),sha256=sha(v)) for k,v in checkpoints.items()}
    if set(weights)!={'p52_bc','p52_joint','p53_bc','p53_joint'}:raise ValueError('Four explicit final checkpoints required')
    packages={}
    for name,item in weights.items():_,packages[name]=load_skill(item['path'])
    bc,joint=packages['p53_bc'],packages['p53_joint']
    if bc.get('branch')!='bc' or joint.get('branch')!='joint' or bc['common_warmup_sha256']!=joint['common_warmup_sha256'] or bc['data_sha256']!=joint['data_sha256']:
        raise ValueError('P53 branches must share exact common warmup and native data')
    if any(p.get('horizon')!=24 or p.get('total_outer_updates')!=3000 for p in (bc,joint)):raise ValueError('P53 final budget differs')
    command='import json,importlib.metadata as m;print(json.dumps({n:m.version(n) for n in ("torch","transformers","numpy")}))'
    vision=dict(vision,packages=json.loads(subprocess.check_output([vision['python'] or sys.executable,'-c',command],text=True)))
    assets=[Path(vision['dino'])/p for p in ('model.safetensors','config.json','preprocessor_config.json')]+[Path(vision['lila'])/p for p in ('VISUAL_WEIGHTS.pt','models/vla_model_fm.py')]
    return dict(schema='p53.micro-eval.v1',phase='opened_development',parent=dict(path=str(parent_path.resolve()),sha256=sha(parent_path)),
        environment=parent['environment'],public=parent['public'],entries=entries,checkpoints=weights,vision=vision,
        sources={p:sha(ROOT/p) for p in source_closure(ROOT,['tools/task_graph/p53_micro_evaluate.py','tools/task_graph/p52_vision_worker.py'])},
        assets={str(p.resolve()):sha(p) for p in assets},packages={n:importlib.metadata.version(n) for n in ('torch','numpy','scipy','mujoco')},
        cells=[dict(index=i*6+j,entry=i,arm=arm) for i in range(6) for j,arm in enumerate(ARMS)],max_actions=24,
        primary_success='official native goal AND at least one finger touching book jointly hold3',
        secondary='official-only hold3 ever reached, descriptive only',
        structural_stop='wrong target immediate; neither finger touching book five consecutive observations',
        denominator='six original Code entries retained for all six arms; failed entry means system failure and no micro execution',
        comparison='same new boundary, P52 versus P53 skill; no direct cross-contract success-rate claim against old P52 records',automatic_retries=0)


def verify(binding):
    if binding['schema']!='p53.micro-eval.v1' or binding['max_actions']!=24:raise ValueError('Wrong micro scope')
    expected=[dict(index=i*6+j,entry=i,arm=arm) for i in range(6) for j,arm in enumerate(ARMS)]
    if binding['cells']!=expected or len(binding['entries'])!=6:raise ValueError('Changed paired denominator')
    if sha(binding['parent']['path'])!=binding['parent']['sha256']:raise ValueError('Parent binding changed')
    verify_environment(binding['environment'])
    for name,digest in binding['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen evaluator source changed')
    for name,digest in binding['assets'].items():
        if sha(name)!=digest:raise ValueError('Frozen visual asset changed')
    for item in binding['checkpoints'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Skill checkpoint changed')
    for n,v in binding['packages'].items():
        if importlib.metadata.version(n)!=v:raise ValueError('Runtime version changed')
    for entry in binding['entries']:verify_entry(entry)


def run(binding_path,index,output):
    binding=json.loads(binding_path.read_text());verify(binding)
    if index not in range(36):raise ValueError('Cell outside declared36')
    cell=binding['cells'][index];entry=binding['entries'][cell['entry']];output.mkdir(parents=True,exist_ok=False)
    dump(output/'BINDING.json',binding);dump(output/'CELL.json',cell)
    if not entry['near_entry_qualified']:
        dump(output/'SEAL.json',dict(binding_sha256=sha(output/'BINDING.json'),cell=cell,files={},micro_executed=False))
        result=dict(cell=cell,micro_executed=False,near_entry_qualified=False,system_success=False,micro_success=None,official_only_hold3=False,
                    reason='blocked_by_original_code_entry_failure',actions=0,seal_sha256=sha(output/'SEAL.json'))
        dump(output/'RESULT.json',result);return result
    for folder in ('public','private'):(output/folder).mkdir()
    version,arm=cell['arm'].split('_');weight_key=f'{version}_'+('bc' if arm=='bc' else 'joint')
    model,_=load_skill(binding['checkpoints'][weight_key]['path']);runtime=StageRuntime(model,online=arm=='online')
    before={k:v.clone() for k,v in model.state_dict().items()};env=vision=renderer=None;steps=0
    try:
        vision=Vision(binding['vision']);dump(output/'VISUAL_BINDING.json',vision.binding)
        env=initialize(binding['environment'],entry,output);renderer=PublicRenderer(env,binding['public']);judge=InsertJudge()
        names={kind:[getattr(env.sim.model,kind+'_id2name')(i) for i in range(getattr(env.sim.model,count))] for kind,count in [('body','nbody'),('geom','ngeom')]}
        obs=env._get_observations(force_update=True);previous=previous_action=None;official_hold=0;official_ever=False
        while True:
            path=output/'public'/f'frame_{steps:04d}.npz';np.savez_compressed(path,**renderer.observe(obs))
            features=vision.encode(path);validate_features(features,model.config);torch.save(features,output/'public'/f'features_{steps:04d}.pt')
            if previous is None:runtime.enter(features)
            before_z=runtime.z.clone();loss=None
            if previous is not None:loss=runtime.feedback(previous,previous_action,features)
            action=runtime.command(features).numpy()
            if action.shape!=(7,) or not np.isfinite(action).all() or action[-1]!=1:raise ValueError('Invalid OSC command')
            official,fingers,wrong=structural_measurement(env,names);judgment=judge.observe(official,fingers,wrong)
            official_hold=official_hold+1 if official else 0;official_ever=official_ever or official_hold>=3
            stop=judgment['stop'] or ('micro_budget24' if steps>=24 else None)
            row=dict(t=steps,observation=path.name,sha256=sha(path),z_before=before_z.tolist(),z_after=runtime.z.tolist(),inner_loss=loss,updates=runtime.updates,action=action.tolist(),executed=stop is None)
            with (output/'public/ACTIONS.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            private,truth=snapshot(env,obs);np.savez_compressed(output/'private'/f'frame_{steps:04d}.npz',**{k:v for k,v in private.items() if k.startswith('supervisor_')})
            with (output/'private/JUDGMENTS.jsonl').open('a') as f:f.write(json.dumps(dict(t=steps,official_success=official,official_only_hold=official_hold,
                  official_only_hold3_ever=official_ever,finger_contacts=fingers,wrong_target=wrong,contacts=truth['contacts'],**judgment),allow_nan=False)+'\n')
            if stop:break
            previous=features;previous_action=torch.tensor(action[:6]);obs,_,_,_=env.step(action.copy());steps+=1
        if any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):raise RuntimeError('Deployment weights mutated')
        verify(binding)
        files={str(p.relative_to(output)):sha(p) for folder in ('public','private') for p in (output/folder).iterdir()}
        dump(output/'SEAL.json',dict(binding_sha256=sha(output/'BINDING.json'),cell=cell,files=files,actions=steps,micro_executed=True))
        result=dict(cell=cell,micro_executed=True,near_entry_qualified=True,system_success=judgment['complete'],micro_success=judgment['complete'],
                    official_only_hold3=official_ever,official_success=official,actions=steps,updates=runtime.updates,reason=stop,
                    seal_sha256=sha(output/'SEAL.json'),full_task_success_evidence=False)
        dump(output/'RESULT.json',result);return result
    except Exception as e:dump(output/'ABORT.json',dict(cell=cell,reason=str(e),actions=steps,automatic_retry=False));raise
    finally:
        runtime.exit()
        if renderer is not None:renderer.close()
        if env is not None:env.close()
        if vision is not None:vision.close()


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ('parent','entries','p52-bc','p52-joint','p53-bc','p53-joint','dino','lila','output'):q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--vision-python');q.add_argument('--vision-device',default='cuda')
    q=sub.add_parser('run');q.add_argument('--binding',type=Path,required=True);q.add_argument('--index',type=int,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if a.command=='prepare':
        if a.output.exists():raise FileExistsError(a.output)
        checkpoints={k:getattr(a,k) for k in ('p52_bc','p52_joint','p53_bc','p53_joint')}
        value=prepare(a.parent,json.loads(a.entries.read_text()),checkpoints,dict(dino=str(a.dino.resolve()),lila=str(a.lila.resolve()),python=a.vision_python,device=a.vision_device))
        a.output.parent.mkdir(parents=True,exist_ok=True);dump(a.output,value)
    else:print(json.dumps(run(a.binding,a.index,a.output)))


if __name__=='__main__':main()
