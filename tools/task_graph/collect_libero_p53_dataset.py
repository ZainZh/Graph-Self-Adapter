#!/usr/bin/env python3
"""24 opened P52 real entries -> new physical Code+micro trajectories; no retries."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,argparse,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.task_graph.collect_libero_p53_v03 import (run_code,record_private,validate_graph,verify as verify_qualification,PublicRenderer,MicroTeacher,private_measurement)
from tools.task_graph.p52_policy_pilot import initialize as initialize_entry
from tools.task_graph.p52_deployment import source_closure
from tools.task_graph.collect_libero_demo_gt import dump,sha,verify as verify_environment


def validate_conditions(entries):
    if len(entries)!=24 or [e['index'] for e in entries]!=list(range(24)) or [e['job_id'] for e in entries]!=[f'job_{i:02d}' for i in range(24)]:raise ValueError('Exactly24 original numbered conditions required')
    for entry in entries:
        c=entry['source_condition'];expected='validation' if c['quadrant_index']==c['grasp_index']%4 else 'train'
        if entry['split']!=expected or entry['split']!=c['split'] or entry['job_id']!=c['job_id']:raise ValueError('Original split/identity changed')
    if sum(e['split']=='train' for e in entries)!=18:raise ValueError('Original18/6 split required')


def prepare(qualification_binding,qualification_results,episodes_root):
    qualification_binding=Path(qualification_binding).resolve();qualification_results=Path(qualification_results).resolve();episodes_root=Path(episodes_root).resolve()
    parent=json.loads(qualification_binding.read_text());verify_qualification(parent)
    provenance={str(qualification_binding):sha(qualification_binding)};qual=[]
    for resultpath in sorted(qualification_results.rglob('RESULT.json')):
        r=json.loads(resultpath.read_text())
        if r.get('schema')!='prove.p53-code-micro-result.v3':continue
        folder=resultpath.parent;seal=json.loads((folder/'SEAL.json').read_text())
        if not r['near_entry_qualified'] or not r['teacher_qualified'] or not r['official_success']:raise ValueError('v03 four qualification gate failed')
        if sha(folder/'SEAL.json')!=r['seal_sha256'] or sha(folder/'BINDING.json')!=sha(qualification_binding):raise ValueError('Qualification provenance mismatch')
        qual.append(r['entry']['index'])
        for name in ('RESULT.json','SEAL.json','BINDING.json'):provenance[str(folder/name)]=sha(folder/name)
    if sorted(qual)!=list(range(4)):raise ValueError('All4 exact v03 qualified results required before collection freeze')
    episodes=[]
    for p in episodes_root.glob('queue_*/worker_*/job_*/episode/RESULT.json'):
        folder=p.parent;r=json.loads(p.read_text());c=json.loads((folder/'ENTRY.json').read_text());seal=json.loads((folder/'SEAL.json').read_text())
        if r.get('schema')!='prove.p52-training-teacher-result.v2':continue
        if not r['prefix_qualified'] or not r['teacher_qualified'] or sha(folder/'SEAL.json')!=r['seal_sha256']:raise ValueError('Original source trajectory not qualified/sealed')
        state=folder/'private/frame_0000.npz';public=folder/'public/frame_0000.npz'
        for name in ('private/frame_0000.npz','public/frame_0000.npz','ENTRY.json','PREFIX_RESULT.json','BINDING.json'):
            if sha(folder/name)!=seal['files'][name]:raise ValueError('Source entry/provenance changed')
        if sha(folder/'BINDING.json')!=seal['binding_sha256']:raise ValueError('Source binding mismatch')
        with np.load(state) as z:
            if 'supervisor_sim_state' not in z or 'supervisor_ctrl' not in z:raise ValueError('Full physical state+ctrl required')
        for name in ('RESULT.json','ENTRY.json','PREFIX_RESULT.json','SEAL.json','BINDING.json','private/frame_0000.npz','public/frame_0000.npz'):provenance[str(folder/name)]=sha(folder/name)
        episodes.append(dict(index=c['index'],job_id=c['job_id'],condition_seed=c['condition_seed'],split=c['split'],source_condition=c,
            frame=str(state),frame_sha256=sha(state),source_public_frame=str(public),source_public_frame_sha256=sha(public),
            source_seal=str(folder/'SEAL.json'),source_seal_sha256=sha(folder/'SEAL.json'),source_result=str(p),source_result_sha256=sha(p),
            source_sealed_files=seal['files'],source_kind='opened_P52_actual_prefix_and_quadrant_physical_entry'))
    episodes.sort(key=lambda e:e['index']);validate_conditions(episodes)
    keys=('environment','public','prefix','relation','task_graph','reference_poses','reference_authority','near_reference_z_offset','entry_contract','allowed_phases','teacher_authority','micro_domain','code_position_tolerance','code_rotation_tolerance','prefix_max_actions')
    binding={k:copy.deepcopy(parent[k]) for k in keys}
    binding.update(schema='prove.p53-restored-real-entry-collection.v1',stage_scope='near_contact_insert',conditions=episodes,
        qualification_binding=dict(path=str(qualification_binding),sha256=sha(qualification_binding)),inputs=provenance,
        sources={p:sha(ROOT/p) for p in source_closure(ROOT,['tools/task_graph/collect_libero_p53_dataset.py'])},
        initialization='Restore each opened real P52 prefix+quadrant entry state/ctrl once; restore gripper command integrator and OSC goal; then actual new Code/micro physics without later state restore',
        automatic_retries=0,max_episodes=24,entry_restores=1,later_restores=0,code_budget=120,micro_budget=24,
        scope='Opened development short-skill training/regression, not fresh held-out or full-task execution',
        training_rule='Only predeclared train + near_entry_qualified + teacher_qualified full short trajectories; all24 failures retained; no replacements')
    return binding


def verify(binding):
    if binding['schema']!='prove.p53-restored-real-entry-collection.v1' or binding['automatic_retries']!=0 or binding['micro_budget']!=24 or binding['code_budget']!=120:raise ValueError('Changed collection scope')
    validate_conditions(binding['conditions']);validate_graph(binding['task_graph']);verify_environment(binding['environment'])
    for p,h in binding['inputs'].items():
        if sha(p)!=h:raise ValueError('Frozen source/provenance changed:'+p)
    for p,h in binding['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen implementation changed:'+p)
    parent=json.loads(Path(binding['qualification_binding']['path']).read_text());verify_qualification(parent)
    for k in ('task_graph','reference_poses','entry_contract','allowed_phases','teacher_authority','micro_domain','code_position_tolerance','code_rotation_tolerance','environment','public','relation'):
        if binding[k]!=parent[k]:raise ValueError('Qualified v03 contract changed:'+k)


def collect(binding_path,index,output):
    binding=json.loads(binding_path.read_text());verify(binding)
    if index not in range(24):raise ValueError('Outside24collection scope')
    entry=binding['conditions'][index];output.mkdir(parents=True,exist_ok=False)
    for name in ('prefix','code','public','private'):(output/name).mkdir()
    dump(output/'BINDING.json',binding);dump(output/'ENTRY.json',entry);dump(output/'TASK_GRAPH.json',binding['task_graph'])
    env=renderer=None;actions=0;labels=[]
    np.random.seed(entry['condition_seed'])
    try:
        env=initialize_entry(binding['environment'],entry,output)
        names={kind:[getattr(env.sim.model,kind+'_id2name')(i) for i in range(getattr(env.sim.model,count))] for kind,count in [('body','nbody'),('site','nsite'),('geom','ngeom')]}
        dump(output/'private/NAMES.json',names)
        prefix=dict(qualified=True,actions=0,source_kind='restored_existing_real_P52_prefix_entry',source_frame=entry['frame'],source_frame_sha256=entry['frame_sha256'],source_job_id=entry['job_id']);dump(output/'PREFIX_RESULT.json',prefix)
        code=run_code(env,binding,output/'code',names) if prefix['qualified'] else dict(qualified=False,actions=0,stop_reason='prefix_failed',near_entry_qualified=False)
        dump(output/'CODE_RESULT.json',code)
        if code['qualified']:
            if code['next_stage']!='s9':raise ValueError('Graph must enter microInsert')
            node=next(n for n in binding['task_graph']['nodes'] if n['id']==code['next_stage'])
            renderer=PublicRenderer(env,binding['public']);teacher=MicroTeacher(binding['relation']);obs=env._get_observations(force_update=True)
            while True:
                public_path=output/'public'/f'frame_{actions:04d}.npz';np.savez_compressed(public_path,**renderer.observe(obs))
                measured=private_measurement(env,obs,names,binding['relation']);action,label=teacher.command(measured)
                stop=label['stop_reason'] or ('micro_budget' if actions>=node['max_actions'] else None)
                private_path=output/'private'/f'frame_{actions:04d}.npz';digest=record_private(env,obs,private_path)
                label.update(t=actions,stage_id=node['id'],finger_contacts=measured['finger_contacts'],contacts=measured['contacts'],rim_height=measured['rim_height'],stop_reason=stop,
                    private_state=private_path.name,private_state_sha256=digest,public_observation=public_path.name,public_observation_sha256=sha(public_path))
                with (output/'private/LABELS.jsonl').open('a') as f:f.write(json.dumps(label,allow_nan=False)+'\n')
                labels.append(label)
                if actions:
                    prev=labels[-2];tr=dict(episode_id=entry['job_id'],stage_id=node['id'],t=actions-1,observation=prev['public_observation'],observation_sha256=prev['public_observation_sha256'],executed_action=prev['expert_action'],next_observation=public_path.name,next_observation_sha256=sha(public_path))
                    with (output/'public/TRANSITIONS.jsonl').open('a') as f:f.write(json.dumps(tr,allow_nan=False)+'\n')
                if stop:break
                obs,_,_,_=env.step(action.copy());actions+=1
        verify(binding)
        files={str(p.relative_to(output)):sha(p) for name in ('prefix','code','public','private') for p in sorted((output/name).iterdir()) if p.is_file()}
        files.update({name:sha(output/name) for name in ('BINDING.json','ENTRY.json','TASK_GRAPH.json','PREFIX_RESULT.json','CODE_RESULT.json')})
        dump(output/'SEAL.json',dict(files=files,binding_sha256=sha(output/'BINDING.json'),entry_restores=1,original_initial_restores=0,later_restores=0))
        result=dict(schema='prove.p53-short-collection-result.v1',entry=entry,job_id=entry['job_id'],split=entry['split'],prefix_qualified=prefix['qualified'],near_entry_qualified=code['qualified'],
            prefix_actions=prefix['actions'],code_actions=code['actions'],actions=actions,frames=len(labels),teacher_qualified=bool(labels and labels[-1]['stop_reason']=='qualified_goal_contact_hold'),
            official_success=bool(env._check_success()),stop_reason=labels[-1]['stop_reason'] if labels else code['stop_reason'],seal_sha256=sha(output/'SEAL.json'),
            code_completed_goal=code['stop_reason']=='code_completed_goal',automatic_retry=False,learned_policy_success=False,training_dataset_qualified=bool(entry['split']=='train' and labels and labels[-1]['stop_reason']=='qualified_goal_contact_hold'),full_task_success_evidence=False)
        dump(output/'RESULT.json',result);return result
    except Exception as exc:
        dump(output/'ABORT.json',dict(error=type(exc).__name__,reason=str(exc),micro_actions=actions,automatic_retry=False));raise
    finally:
        if renderer is not None:renderer.close()
        if env is not None:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for name in ('qualification-binding','qualification-results','episodes-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('collect');p.add_argument('--binding',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='prepare':
        if a.output.exists():raise FileExistsError(a.output)
        b=prepare(a.qualification_binding,a.qualification_results,a.episodes_root);verify(b);dump(a.output,b)
    else:print(json.dumps(collect(a.binding,a.index,a.output),indent=2))
