#!/usr/bin/env python3
"""P53 v03: near-contact local alignment plus insertion within unchanged H24."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,argparse
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from prove.task_graph.p53_micro_insert_v03 import PublicPosePath,MicroTeacher,near_supervisor,geometry
from tools.task_graph.collect_libero_p52_variants import VariantController,verify as verify_collection
from tools.task_graph.collect_libero_p52_v02 import PublicRenderer,private_measurement
from tools.task_graph.p52_full_task import initialize,TaskGraph,MixedRouter,GraphExecutor,LiberoSupervisor,grounding,insert_entry_domain,snapshot
from tools.task_graph.p52_deployment import source_closure
from tools.task_graph.collect_libero_demo_gt import dump,sha,verify as verify_environment


def prepare(collection_binding,reference):
    collection_binding=Path(collection_binding).resolve();reference=Path(reference).resolve()
    parent=json.loads(collection_binding.read_text());verify_collection(parent)
    seal=json.loads((reference/'SEAL.json').read_text());result=json.loads((reference/'RESULT.json').read_text())
    if not result['graph_complete'] or not result['official_success'] or sha(reference/'SEAL.json')!=result['seal_sha256']:raise ValueError('Successful opened reference required')
    refs=[];inputs={str(collection_binding):sha(collection_binding),str(reference/'SEAL.json'):sha(reference/'SEAL.json'),str(reference/'RESULT.json'):sha(reference/'RESULT.json')}
    for i in range(58):
        path=reference/'public'/f'frame_{i:04d}.npz'
        if sha(path)!=seal['files'][f'public/{path.name}']:raise ValueError('Reference changed')
        with np.load(path) as z:refs.append(dict(position=z['eef_pos'].tolist(),quaternion_xyzw=z['eef_quat_xyzw'].tolist()))
        inputs[str(path)]=sha(path)
    tol=min(parent['relation']['book_collision_halfsize'])/4;rot=float(np.arctan((min(parent['relation']['book_collision_halfsize'])/16)/max(parent['relation']['book_collision_halfsize'])))
    refs[57]['position'][2]+=tol
    conditions=[dict(index=0,job_id='qualification_00',condition_seed=530000,grasp_index=None,grasp_world_offset=[0.,0.,0.])]
    for g in (0,3,5):
        item=next(c for c in parent['conditions'] if c['grasp_index']==g);i=len(conditions)
        conditions.append(dict(index=i,job_id=f'qualification_{i:02d}',condition_seed=530000+i,grasp_index=g,grasp_world_offset=item['grasp_world_offset']))
    graph=dict(schema='prove.p53-short-taskgraph.v3',graph_id='libero_book_caddy_p53_micro_v03',
      prefix_nodes=parent['prefix']['runtime_graph']['nodes'][:6],
      nodes=[dict(id='s7',skill='Align',backend='public_code_pose_path',reference_start=0,reference_end=1,target_rule='raise_current_xy_orientation_to_public_max_z_then_reference41',next='s8',termination='public_endpoint_reached'),
             dict(id='s8',skill='Approach',backend='public_code_pose_path',reference_start=2,reference_end=2,target_rule='public_reference57_plus_one_position_tolerance_z',next='s9',termination='public_endpoint_reached_and_near_ready'),
             dict(id='s9',skill='Insert',backend='private_training_micro_teacher',max_actions=24,next=None,termination='original_goal_and_contact_hold3')],
      shared_code_budget=120,prefix_budget=400,public_action_inputs=['robot0_eef_pos','robot0_eef_quat'],
      supervisor_routing_only=['near_ready','structural_stop','code_completed_goal'])
    return dict(schema='prove.p53-code-micro-qualification.v3',environment=parent['environment'],public=parent['public'],prefix=parent['prefix'],relation=parent['relation'],
      parent=dict(path=str(collection_binding),sha256=sha(collection_binding)),task_graph=graph,reference_poses=refs,reference_authority='opened P52 successful BC public EEF only; static original scene',
      near_reference_z_offset=tol,entry_contract=dict(bottom_gap_min=0.,bottom_gap_max=2*tol,footprint_slack_min=[-tol,-tol],require_bilateral_contact=True,require_official_false=True),allowed_phases=['local_align','insert'],teacher_authority='private_training_micro_teacher_gt_v03',micro_domain=dict(max_gap=3*tol,min_footprint_slack=-tol,unsafe_below_rim_realign='stop',phases=['local_align','insert']),code_position_tolerance=tol,code_rotation_tolerance=rot,prefix_max_actions=400,conditions=conditions,
      inputs=inputs,sources={p:sha(ROOT/p) for p in source_closure(ROOT,['tools/task_graph/collect_libero_p53_v03.py'])},
      automatic_retries=0,original_initial_restores=1,later_state_restores=0,training_dataset_qualified=False,
      scope='4 Code/micro teacher engineering qualifications, not learned policy success or generalized Code skill')


def validate_graph(graph):
    if graph['schema']!='prove.p53-short-taskgraph.v3' or [n['id'] for n in graph['prefix_nodes']]!=[f's{i}' for i in range(1,7)]:raise ValueError('Prefix graph mismatch')
    nodes=graph['nodes']
    if [(n['id'],n['skill'],n['backend'],n['next']) for n in nodes[:2]]!=[('s7','Align','public_code_pose_path','s8'),('s8','Approach','public_code_pose_path','s9')]:raise ValueError('Code graph routing mismatch')
    if len(nodes)!=3 or (nodes[2]['id'],nodes[2]['skill'],nodes[2]['next'])!=('s9','Insert',None) or nodes[2]['backend'] not in ('private_training_micro_teacher','frozen_stage_policy'):raise ValueError('Insert graph routing mismatch')
    if [(n['reference_start'],n['reference_end']) for n in nodes[:2]]!=[(0,1),(2,2)] or graph['shared_code_budget']!=120 or nodes[2]['max_actions']!=24:raise ValueError('Graph bounds changed')
    if [n['target_rule'] for n in nodes[:2]]!=['raise_current_xy_orientation_to_public_max_z_then_reference41','public_reference57_plus_one_position_tolerance_z']:raise ValueError('Code target rule changed')
    if [n['termination'] for n in nodes]!=['public_endpoint_reached','public_endpoint_reached_and_near_ready','original_goal_and_contact_hold3']:raise ValueError('Graph boundary changed')


def verify(binding):
    if binding['schema']!='prove.p53-code-micro-qualification.v3' or len(binding['conditions'])!=4 or binding['automatic_retries']!=0:raise ValueError('Qualification scope changed')
    validate_graph(binding['task_graph']);verify_environment(binding['environment'])
    parent=json.loads(Path(binding['parent']['path']).read_text());verify_collection(parent)
    if parent['prefix']!=binding['prefix'] or parent['relation']!=binding['relation']:raise ValueError('Parent prefix/relation changed')
    for p,h in binding['inputs'].items():
        if sha(p)!=h:raise ValueError('Frozen input changed:'+p)
    for p,h in binding['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen source changed:'+p)
    expected=prepare(binding['parent']['path'],Path(next(p for p in binding['inputs'] if p.endswith('/public/frame_0000.npz'))).parents[1])
    if expected!=binding:raise ValueError('Freeze differs from source-derived configuration')


def run_prefix(env,binding,output,instance,entry):
    prefix=binding['prefix'];graph=TaskGraph.from_dict(prefix['runtime_graph'])
    if [s.id for s in graph.nodes]!=[f's{i}' for i in range(1,8)]:raise ValueError('Unexpected original graph')
    skill=VariantController(binding['environment']['source'],graph,prefix['checkpoint'],entry['grasp_world_offset'])
    router=MixedRouter()
    for name in {s.skill for s in graph.nodes}:
        router.register(name,kind=router.required_kind(name),revision=skill.revision,call=skill,compatible=lambda g:g.stage_id in skill.segments)
    executor=GraphExecutor(graph,router,instance_id=instance)
    supervisor=LiberoSupervisor(env,json.loads(Path(prefix['calibration']).read_text()),instance_id=instance)
    obs=env._get_observations(force_update=True);active=None;holds=0;actions=0
    try:
        while True:
            skill.set_observation(obs);action=None;stop=None;ready=False;events=[]
            for _ in range(8):
                stage=graph.stage(executor.current_stage)
                if stage.id=='s7':raise RuntimeError('Old s7 controller must never execute')
                if active!=stage.id:
                    if active is not None:skill.on_exit()
                    skill.on_enter(stage.id);active=stage.id
                report=supervisor.observe(obs,stage.id);facts=dict(report.facts)
                entry_ok,entry_info=insert_entry_domain(skill);facts['public_insert_entry_domain']=entry_ok
                if actions>=binding['prefix_max_actions']:stop='prefix_budget';break
                result=executor.tick(grounding(graph,stage,instance,actions),facts,progress=report.progress)
                event=asdict(result);event['action']=None if result.action is None else np.asarray(result.action).tolist()
                events.append(dict(judge=event,supervisor=asdict(report),public_entry=dict(qualified=entry_ok,**entry_info),controller=dict(skill.diagnostics)))
                if result.state=='complete':
                    if stage.id=='s6':ready=True;break
                    continue
                action=event['action']
                if action is None:
                    unknown=result.reason in ('invariant_unknown','precondition_unknown')
                    if unknown and result.state!='structural_failure' and not any(facts.get(k) is True for k in ('object_lost','wrong_target')) and holds<3:
                        holds+=1;action=[0.,0.,0.,0.,0.,0.,1.]
                    else:stop='prefix_'+str(result.reason)
                break
            else:raise RuntimeError('Unbounded same-observation prefix transition')
            arrays,_=snapshot(env,obs);arrays={k:v for k,v in arrays.items() if k.startswith('supervisor_')}
            arrays['action_next']=np.asarray([] if action is None else action)
            path=output/f'frame_{actions:04d}.npz';np.savez_compressed(path,**arrays)
            with (output/'TRACE.jsonl').open('a') as f:f.write(json.dumps(dict(t=actions,file=path.name,sha256=sha(path),events=events,action=action,ready=ready,stop=stop),allow_nan=False)+'\n')
            if stop or ready:return dict(qualified=ready,stop_reason=stop,actions=actions,completed_s6=ready,public_entry_qualified=bool(entry_ok))
            obs,_,_,_=env.step(np.asarray(action));actions+=1
    finally:skill.on_exit()


def record_private(env,obs,path):
    arrays,_=snapshot(env,obs);arrays={k:v for k,v in arrays.items() if k.startswith('supervisor_')}
    np.savez_compressed(path,**arrays);return sha(path)


def run_code(env,binding,output,names):
    graph=binding['task_graph'];validate_graph(graph)
    nodes={n['id']:n for n in graph['nodes']};stage='s7';active=nodes[stage]
    actions=0;obs=env._get_observations(force_update=True)
    initial=np.asarray(obs['robot0_eef_pos']).copy();initial[2]=max(p['position'][2] for p in binding['reference_poses'])
    targets=[dict(position=initial.tolist(),quaternion_xyzw=np.asarray(obs['robot0_eef_quat']).tolist()),binding['reference_poses'][41],binding['reference_poses'][57]]
    control=PublicPosePath(targets,binding['code_position_tolerance'],binding['code_rotation_tolerance'])
    dump(output/'PUBLIC_TARGETS.json',dict(targets=targets,origin='public current robot XY/orientation + frozen public reference maximum Z, ref41, ref57'))
    while True:
        measured=private_measurement(env,obs,names,binding['relation']);gate=near_supervisor(binding['relation'],measured)
        # Gate is used only for stop/handoff. The controller receives robot state only.
        action,reached,diag=control.command(obs['robot0_eef_pos'],obs['robot0_eef_quat'],active['reference_start'],active['reference_end'])
        transitions=[]
        if reached and stage=='s7':
            transitions.append(dict(from_stage='s7',to_stage=active['next']));stage=active['next'];active=nodes[stage]
            action,reached,diag=control.command(obs['robot0_eef_pos'],obs['robot0_eef_quat'],active['reference_start'],active['reference_end'])
        ready=bool(stage=='s8' and reached and gate['near_ready'])
        stop='code_completed_goal' if gate['code_completed_goal'] else 'code_structural_stop' if gate['structural_stop'] else 'near_ready' if ready else 'code_budget' if actions>=graph['shared_code_budget'] else None
        path=output/f'frame_{actions:04d}.npz';digest=record_private(env,obs,path)
        row=dict(t=actions,stage_id=stage,file=path.name,sha256=digest,action=None if stop else action.tolist(),
                 public_controller=diag,supervisor_gate=gate,private_geometry=geometry(binding['relation'],measured),transitions=transitions,stop_reason=stop)
        with (output/'TRACE.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if stop:return dict(qualified=ready,actions=actions,stop_reason=stop,next_stage=active['next'] if ready else None,near_entry_qualified=ready)
        obs,_,_,_=env.step(action.copy());actions+=1


def collect(binding_path,index,output):
    binding=json.loads(binding_path.read_text());verify(binding)
    if index not in range(4):raise ValueError('Outside4qualification scope')
    entry=binding['conditions'][index];output.mkdir(parents=True,exist_ok=False)
    for name in ('prefix','code','public','private'):(output/name).mkdir()
    dump(output/'BINDING.json',binding);dump(output/'ENTRY.json',entry);dump(output/'TASK_GRAPH.json',binding['task_graph'])
    env=renderer=None;actions=0;labels=[]
    np.random.seed(entry['condition_seed'])
    try:
        env=initialize(binding,output)
        names={kind:[getattr(env.sim.model,kind+'_id2name')(i) for i in range(getattr(env.sim.model,count))] for kind,count in [('body','nbody'),('site','nsite'),('geom','ngeom')]}
        dump(output/'private/NAMES.json',names)
        prefix=run_prefix(env,binding,output/'prefix',entry['job_id'],entry);dump(output/'PREFIX_RESULT.json',prefix)
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
        dump(output/'SEAL.json',dict(files=files,binding_sha256=sha(output/'BINDING.json'),original_restores=1,later_restores=0))
        result=dict(schema='prove.p53-code-micro-result.v3',entry=entry,job_id=entry['job_id'],prefix_qualified=prefix['qualified'],near_entry_qualified=code['qualified'],
            prefix_actions=prefix['actions'],code_actions=code['actions'],actions=actions,frames=len(labels),teacher_qualified=bool(labels and labels[-1]['stop_reason']=='qualified_goal_contact_hold'),
            official_success=bool(env._check_success()),stop_reason=labels[-1]['stop_reason'] if labels else code['stop_reason'],seal_sha256=sha(output/'SEAL.json'),
            code_completed_goal=code['stop_reason']=='code_completed_goal',automatic_retry=False,learned_policy_success=False,training_dataset_qualified=False)
        dump(output/'RESULT.json',result);return result
    except Exception as exc:
        dump(output/'ABORT.json',dict(error=type(exc).__name__,reason=str(exc),micro_actions=actions,automatic_retry=False));raise
    finally:
        if renderer is not None:renderer.close()
        if env is not None:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for name in ('collection-binding','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('collect');p.add_argument('--binding',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='prepare':
        if a.output.exists():raise FileExistsError(a.output)
        binding=prepare(a.collection_binding,a.reference);verify(binding);dump(a.output,binding)
        graphpath=a.output.with_name('TASK_GRAPH_v03.json')
        if graphpath.exists():raise FileExistsError(graphpath)
        dump(graphpath,binding['task_graph'])
    else:print(json.dumps(collect(a.binding,a.index,a.output),indent=2))
