#!/usr/bin/env python3
"""Offline RGB handoff only: frozen scene reconstruction, never physical stepping."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from prove.task_graph.agent_stage_judge import digest,validate_request,REQUEST_SCHEMA


def checked(binding):
    path=Path(binding['path']).resolve()
    if digest(path)!=binding['sha256']:raise ValueError('input hash mismatch: '+str(path))
    return path


def verify_scope(scope_path,expected_sha):
    if digest(scope_path)!=expected_sha:raise ValueError('scope hash mismatch')
    scope=json.loads(Path(scope_path).read_text())
    if scope.get('schema')!='graph-self-adapter.agent-stage-pilot-scope.v1' or scope.get('physical_episode_limit')!=0:raise ValueError('offline scope required')
    paths={key:checked(binding) for key,binding in scope['inputs'].items()}
    for binding in scope['source_bindings'].values():checked(binding)
    freeze=json.loads(Path(scope['source_bindings']['closed_v04_execution_freeze']['path']).read_text())
    for binding in freeze['bindings']:checked(binding)
    output=Path(scope['output']).resolve()
    if not output.is_relative_to((REPO/'evaluations/agent_stage_v05').resolve()):raise ValueError('new v05 output required')
    if output.exists():raise FileExistsError(output)
    return scope,paths,output


def sample_indices(times,scope):
    import numpy as np
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('invalid causal trace clock')
    available=np.flatnonzero((times>=scope['earliest_time_sec'])&(times<=scope['cutoff_time_sec']))
    if not len(available):raise ValueError('no scoped RGB times')
    return sorted({int(available[np.argmin(abs(times[available]-target))]) for target in scope['sample_times_sec']})


def public_proprio(archive,observed):
    """Read only robot arrays, never archived role geometry/predictions/labels."""
    import numpy as np
    candidates=[(float(archive[key]),key[:-len('timestamp')]) for key in archive.files if key.endswith('_timestamp')]
    candidates=[row for row in candidates if row[0]<=observed]
    if not candidates: return {}
    time,prefix=max(candidates)
    if observed-time>.25:return {}
    arms={}
    for arm in ('left','right'):
        row={key:np.asarray(archive[prefix+arm+'_'+key]).tolist() for key in
             ('q','qdot','tcp_position_m','tcp_rotation','finger_encoders','actual_aperture_m','commanded_aperture_m')}
        row['robot_timestamp_sec']=float(archive[prefix+arm+'_timestamp_sec'])
        row.update(measurement_source='simulator_actual_joint_encoder',
            calibration_source='sealed_v04_encoder_equivalent_proxy_not_physical_inner_jaw_width')
        arms[arm]=row
    return arms


def generate(scope,paths,output,scope_path,scope_sha):
    import numpy as np
    from PIL import Image,ImageDraw
    spec=importlib.util.spec_from_file_location('_agent_rgb_renderer',paths['renderer'])
    render=importlib.util.module_from_spec(spec);sys.modules[spec.name]=render;spec.loader.exec_module(render)
    # This source is for initialization/binding audit only. No result/trace facts enter request.
    full=json.loads(paths['report'].read_text())
    report={key:full[key] for key in ('schema','task','candidate_id','condition','bindings')}
    del full
    scene=paths['scene'];candidate=paths['candidate_freeze']
    for key in ('scene','candidate_freeze','visual_trace'):
        binding=report['bindings'].get(key)
        expected=(binding.get('sha256') if isinstance(binding,dict) else report['bindings'].get(key+'_sha256'))
        if digest(paths[key])!=expected:raise ValueError('sealed source '+key+' hash mismatch')
        raw_path=binding.get('path') if isinstance(binding,dict) else report['bindings'].get(key+'_path')
        if raw_path is not None:
            bound=render._resolve_bound_path(raw_path,paths['report'])
            if bound!=paths[key].resolve():raise ValueError('sealed source '+key+' path mismatch')
    freeze=json.loads(candidate.read_text())
    physics=paths['frozen_physics']
    if digest(physics)!=freeze['code']['prove/task_graph/twin_physics.py']:raise ValueError('frozen physics mismatch')
    graph=json.loads(paths['graph'].read_text());stage=next(s for s in graph['stages'] if s['stage_id']==scope['stage_id'])
    if (stage['object'],stage['target'],stage['arm'])!=(scope['object'],scope['target'],scope['actor']):raise ValueError('stage intent mismatch')
    with np.load(paths['visual_trace'],allow_pickle=False) as trace:
        times=trace['timestamps'].copy();indices=sample_indices(times,scope);qpos=trace['qpos'][indices].copy()
    world,model,data,initial=render.frozen_initial_state(report,paths['report'],scene,physics,candidate)
    if initial['mj_step_executed'] is not False:raise ValueError('offline initialization stepped physics')
    renderer=None
    try:
        available=[model.camera(i).name for i in range(model.ncam)]
        views=[v for v in scope['views'] if v in available]
        if not views:raise ValueError('no requested RGB camera exists')
        if len(indices)*len(views)>scope['max_frames']:raise ValueError('frame cap exceeded')
        output.mkdir(parents=True);renderer=render.mujoco.Renderer(model,width=640,height=360)
        frames=[];pictures=[]
        for index,state in zip(indices,qpos):
            data.qpos[:]=state;data.qvel[:]=0.;data.time=float(times[index]);render.mujoco.mj_forward(model,data)
            for view in views:
                renderer.update_scene(data,camera=view);rgb=renderer.render()
                frame_id=f'f{len(frames):02d}';filename=frame_id+'_'+view+'.png';Image.fromarray(rgb).save(output/filename)
                frame={'frame_id':frame_id,'path':filename,'sha256':digest(output/filename),
                       'physical_timestamp_sec':float(times[index]),'view':view}
                frames.append(frame)
                tile=Image.new('RGB',(640,390),'white');tile.paste(Image.fromarray(rgb),(0,30))
                ImageDraw.Draw(tile).text((8,8),f"{scope['task']} | {scope['stage_id']} | t={times[index]:.3f}s | {view}",fill='black');pictures.append(tile)
        observed=max(f['physical_timestamp_sec'] for f in frames)
        with np.load(paths['public_packets'],allow_pickle=False) as archive:prop=public_proprio(archive,observed)
        request={'schema':REQUEST_SCHEMA,'request_id':'agent-stage-v05-knife00-align','task':scope['task'],'stage_id':scope['stage_id'],
            'graph_sha256':digest(paths['graph']),'observed_physical_timestamp_sec':observed,
            'expires_physical_timestamp_sec':observed+.25,
            'stage_contract':{'skill':stage['phase'],'actor':stage['actor'],'object':stage['object'],'target':stage['target'],
                'goal':stage['goal_relation'],'termination':'Declared Align finishes when reference progression and robot command target are reached while held-or-supported relation is maintained; evaluate visible placement semantics, without certifying hidden contact.'},
            'frames':frames,'public_proprio':prop,'requested_fact_ids':['visible_target_relation','placement_visually_stable','gripper_visibly_released'],
            'input_provenance':'RGB_only_plus_optional_public_robot_feedback; simulator_proxy_not_hardware'}
        validate_request(request,now_sec=observed,task=scope['task'],stage_id=scope['stage_id'],graph_sha256=request['graph_sha256'],frame_root=output)
        (output/'request.json').write_text(json.dumps(request,indent=2,sort_keys=True)+'\n')
        panel=Image.new('RGB',(1280,390*((len(pictures)+1)//2)),'white')
        for i,picture in enumerate(pictures):panel.paste(picture,((i%2)*640,(i//2)*390))
        panel.save(output/'panel.png')
        for binding in (*scope['source_bindings'].values(),*scope['inputs'].values()):checked(binding)
        if digest(scope_path)!=scope_sha:raise ValueError('scope changed during generation')
        manifest={'schema':'graph-self-adapter.agent-stage-evidence.v1','scope':{'path':str(scope_path),'sha256':scope_sha},
            'source_bindings':scope['source_bindings'],'inputs':scope['inputs'],'actual_code':{str(REPO/p):digest(REPO/p) for p in
                ('prove/task_graph/agent_stage_judge.py','tools/evidence/build_agent_stage_evidence.py')},
            'request':{'path':str(output/'request.json'),'sha256':digest(output/'request.json')},
            'panel':{'path':str(output/'panel.png'),'sha256':digest(output/'panel.png')},'frames':frames,
            'available_views':views,'missing_requested_views':[v for v in scope['views'] if v not in views],
            'mj_step_executed':False,'new_physical_episodes':0,'GT_labels_in_request':False,
            'renderer_only_initialization':initial,'reviewer_not_blinded':scope['reviewer_not_blinded']}
        (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        return manifest
    finally:
        if renderer is not None:renderer.close()
        world.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scope',type=Path,required=True)
    p.add_argument('--scope-sha256',required=True);p.add_argument('--execute',action='store_true');args=p.parse_args(argv)
    scope,paths,output=verify_scope(args.scope,args.scope_sha256)
    if not args.execute:print(json.dumps({'dry_run':True,'scope_sha256':args.scope_sha256,'output':str(output),'physical_episodes':0}));return
    if scope.get('execution_authorized') is not True:raise ValueError('scope not authorized; nothing generated')
    manifest=generate(scope,paths,output,args.scope,args.scope_sha256)
    print(json.dumps({'request':manifest['request'],'panel':manifest['panel'],'frames':len(manifest['frames']),'new_physical_episodes':0}))


if __name__=='__main__':main()
