#!/usr/bin/env python3
"""Fresh scoped all-task workflow; shared real/sim interfaces, API grounding, continuous preview."""
import argparse,copy,hashlib,json,os,sys,time,uuid
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
for p in (HERE,REPO):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from night_route_v57 import PROFILES,compile_task
from prove.xtrainer.multitask_bundle import load_bundle
from prove.xtrainer.agent import digest,write
from prove.xtrainer.io import SafetyError,load_calibration
import direct_white_entry as backend
SOURCES=[Path(__file__),HERE/'night_route_v57.py',HERE/'night_runtime_v57.py',HERE/'night_vision_v54.py',HERE/'api_stage_judge_v57.py',HERE/'api_grounding_v28.py',HERE/'multitask_special_runtime.py',HERE/'object_centered_runtime.py',HERE/'ordinary_v25_repair_sim_io.py',HERE/'all_code_stage_trajectory.py',HERE/'stage_trajectory.py',HERE/'all_code_stage_runtime.py',HERE/'public_grasp_geometry_v55.py',HERE/'sim_preview_v33.py',HERE/'task_stream_view_v35.py',REPO/'prove/xtrainer/runtime.py',REPO/'prove/xtrainer/agent.py',REPO/'prove/xtrainer/perception.py']
PROFILE=None;CANDIDATE=None

def runtime_hash():return hashlib.sha256(''.join(digest(p) for p in SOURCES).encode()).hexdigest()
def bindings(path,cfg,mode):
    scene=cfg['task'];b=load_bundle(scene);expected=compile_task(b['graph'],PROFILE,b);route=Path(cfg['special_route_graph'])
    if json.loads(route.read_text())!=expected:raise SafetyError('declared task variant graph differs')
    b['graph']=expected;b['paths']['graph']=route;runtime=copy.deepcopy(cfg['runtime']);camera,right=load_calibration(runtime['left_calibration'],runtime['right_calibration']);runtime.update(T_left_from_camera=camera.tolist(),T_left_from_right=right.tolist())
    files=SOURCES+[path,route,Path(cfg['formal_run_reservation']),Path(cfg['sim']['physics_source']),Path(runtime['sdk_file']),Path(runtime['left_calibration']),Path(runtime['right_calibration']),Path(runtime['vision']['sam_utility']),Path(runtime['vision']['sam_checkpoint'])]
    if runtime.get('paired_SAM_reference'):files.append(Path(runtime['paired_SAM_reference']))
    files.append(Path(runtime['robot_pad_prior']))
    scene_root=Path(cfg['sim']['scene_run']);files+=list(scene_root.glob('*.xml'))
    catalog=REPO/'evaluations/twin_rsi_v01/EXTERNAL_ASSET_BINDINGS.json';files.append(catalog);assets=json.loads(catalog.read_text())
    for p,h in assets['task_assets'][scene]['external_asset_hashes'].items():
        if digest(p)!=h:raise SafetyError('standard scene asset changed')
        files.append(Path(p))
    for row in assets.get('external_evaluator_sources',[]):
        if digest(row['path'])!=row['sha256']:raise SafetyError('evaluator code changed')
        files.append(Path(row['path']))
    bound=dict(b['bindings']);bound.update({str(p.resolve()):digest(p) for p in files});return {'dependencies':{}},runtime,b,bound

def descriptions(scene,graph):
    roles=sorted(({s['object'] for s in graph['stages']}|{s['target'] for s in graph['stages']})-{'table_handover_region'})
    labels={'drawer':'moving drawer body including its handle, exclude stationary outer shell','drawer_shell':'stationary outer drawer cabinet shell','mug':'whole mug including handle','rack':'whole mug rack/peg target','yellow_cup':'whole yellow cup','blue_cup':'whole blue cup target','yellow_cube':'whole yellow cube','tray':'whole tray receptacle','red_knife':'whole red handled knife, exclude holder and other knife','board':'whole cutting board','broom':'whole broom including handle and brush','dustpan':'whole dustpan including handle','bowl':'whole receiving bowl'}
    out={r:labels.get(r,'whole '+r.replace('_',' ')) for r in roles}
    if scene in ('collect_cubes','sorting_cube_bimanual'):out['support_plane_region']='small visible flat tabletop patch with no objects, robot, edges or shadows; background ROI only'
    if scene=='put_knife_board_new':out['red_knife_grasp_part']='ONLY exposed red grip/handle of red knife; exclude blade, knife holder and blue knife'
    return out

def main(argv=None):
    global PROFILE,CANDIDATE
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=tuple(PROFILES),required=True);p.add_argument('--backend',choices=('sim','real'),default='sim');p.add_argument('--check',action='store_true');p.add_argument('--candidate',default='c09');p.add_argument('--live-port',type=int);a=p.parse_args(argv);PROFILE=a.task;CANDIDATE=a.candidate;scene=PROFILES[PROFILE][0]
    root=REPO/'evaluations/night_tasks_v49'/PROFILE;source=json.loads((root/'CONFIG_SOURCE.json').read_text());cfg=copy.deepcopy(json.loads(Path(source['path']).read_text()));cfg['task']=scene
    paired=root/'PAIRED_SAM_REFERENCE_V52.json'
    if paired.exists():cfg['runtime'].update(paired_SAM_reference=str(paired.resolve()),paired_SAM_reference_sha256=digest(paired))
    b=load_bundle(scene);graph=compile_task(b['graph'],PROFILE,b);cfg['runtime']['stage_judge'].update(model='openai/gpt-6-astra',backend='OpenRouter',maximum_calls=12)
    cfg['runtime'].setdefault('max_root_requests_per_task',16)
    cfg['runtime']['maximum_node_control_updates']=300
    cfg['runtime']['robot_pad_prior']=str(REPO/'evaluations/three_task_migration_v25/collect_cubes/grasp_lift_observer_v01/ROBOT_DISTAL_PAD_PRIOR.json')
    if a.backend=='real':
        if not a.check:p.error('offline real interface checks only; no hardware connection')
        from three_task_v26_public_io import offline_contract_check,real_constructor_check
        r=offline_contract_check(scene);r['real_constructor']=real_constructor_check(cfg['runtime']);print(json.dumps(r,indent=2));return r
    if a.check:
        r={'logical_task':PROFILE,'scene':scene,'nodes':len(graph['stages']),'roles':descriptions(scene,graph),'phases':[s['phase'] for s in graph['stages']],'public_interfaces':['capture','samples','command','fk','stop'],'device_connections':0,'episodes_started':0};print(json.dumps(r,indent=2));return r
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    owner=root/CANDIDATE;owner.mkdir(exist_ok=True);scope=owner/'TRIAL_SCOPE.json'
    if not scope.exists():write(scope,{'schema':'graph-self-adapter.simulator-candidate-validation.v1','task':scene,'logical_task':PROFILE,'mode':'sim','execution_authorized':True,'hardware_motion_jobs':0,'maximum_full_simulator_episodes':1,'maximum_paid_API_calls':13,'model':'openai/gpt-6-astra','objective':json.loads((REPO/'evaluations/night_tasks_v49/PLAN.json').read_text())['tasks'][PROFILE]['core'],'initial_conditions':'Opened nominal native scene, one continuous initialization','forbidden_runtime_inputs':['object body pose','contact truth','goal labels'],'physics_continues_during_API':True,'seal_before_private_evaluation':True,'user_authorization':'Continue and optimize all remaining tasks; explicitly simplified core goals allowed'})
    stamp=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8];output=owner/'runs'/stamp;control=owner/'control'/stamp;control.mkdir(parents=True)
    from macro_one_command import reserve
    claim=reserve(scope,owner/'FORMAL_RUN_LEDGER.json',output,control);write(control/'FORMAL_RUN_RESERVATION.json',claim);route=control/'TASK_GRAPH.json';write(route,graph)
    cfg.update(special_route_graph=str(route),formal_run_reservation=str(control/'FORMAL_RUN_RESERVATION.json'))
    cfg['runtime']['vision'].update(role_descriptions=descriptions(scene,graph),role_boxes={},prompt_frame_sha256=None,prompt_frame_output=str(output/'INITIAL_RGB.png'),prompt_reply_path=str(output/'INITIAL_BOXES.json'))
    config=control/'CONFIG.json';write(config,cfg);_,_,_,bound=bindings(config,cfg,'sim');bound[str(scope)]=digest(scope)
    freeze={'schema':'graph-self-adapter.shared-cup-freeze.v1','status':'result_free_frozen','scope_sha256':digest(scope),'runtime_sha256':runtime_hash(),'bindings':[{'path':p,'sha256':h} for p,h in sorted(bound.items())],'jobs':[{'task':scene,'mode':'sim','output':str(output)}],'formal_reservation':str(control/'FORMAL_RUN_RESERVATION.json'),'hardware_motion_jobs':0}
    freeze_path=control/'EXECUTION_FREEZE.json';write(freeze_path,freeze)
    oldload=backend.load
    def load(path,name):
        if Path(path).name=='sim_robot_io.py':
            import ordinary_v25_repair_sim_io as io
            return io
        if Path(path).name=='direct_white_runtime.py':
            import night_runtime_v57 as runtime
            return runtime
        return oldload(path,name)
    from api_stage_judge_v57 import APIStageJudge
    backend.TASK=scene;backend.VARIANT='night_all_task_common_io_v49';backend.load=load;backend.bindings=bindings;backend.runtime_hash=runtime_hash;backend.RealRootJudge=lambda *args,**kw:APIStageJudge(*args,**kw,maximum_calls=12)
    from task_stream_view_v33 import install
    install()
    from task_stream_view_v35 import install as labels
    labels()
    import task_live_view_v27
    port=a.live_port or 18800+list(PROFILES).index(PROFILE);view=task_live_view_v27.TaskLiveView(PROFILE,port);view.set_owner(owner);print(json.dumps({'task':PROFILE,'live_view':view.start(),'output':str(output)}),flush=True)
    try:
        result=backend.main(['sim','--config',str(config),'--scope',str(scope),'--execution-freeze',str(freeze_path),'--output',str(output)])
        if isinstance(result,dict):
            checks=result['post_seal_private_evaluation']['task_outcome']['checks'];safe=result['post_seal_private_evaluation']['numerical_safety']['passed']
            names={'open_drawer_only':['drawer_opened'],'nest_two_cups':['yellow_cup_grasped','yellow_inside_blue'],'sort_one_yellow_cube':['yellow_cube_left_grasp','yellow_cube_right_grasp','yellow_cube_inside_tray'],'collect_single_cube':['yellow_cube_grasped','yellow_cube_inside_bowl'],'sweep_with_stationary_pan':['has_sweep_block','yellow_cube_inside_pan','brush_force_grip','brush_held_at_end']}.get(PROFILE)
            goal=result['post_seal_private_evaluation']['task_outcome']['passed'] if names is None else all(checks.get(k) is True for k in names)
            report={'logical_task':PROFILE,'scene_task':scene,'graph_completed':result['graph_completed'],'failure':result['failure'],'nodes':len(graph['stages']),'completed_nodes':len(result['stage_trace']),'task_goal_passed':bool(result['graph_completed'] and goal and safe),'private_goal_checks':checks,'numerical_safety_passed':safe,'evaluation_after_seal':True,'hardware_motion_commands':result['hardware_motion_commands']};write(output/'TASK_VARIANT_RESULT.json',report);print(json.dumps(report,indent=2),flush=True)
            status_path=REPO/'evaluations/night_tasks_v49/STATUS.json';status=json.loads(status_path.read_text());status['tasks'][PROFILE].update(state='pass' if report['task_goal_passed'] else 'negative',qualified=report['task_goal_passed'],latest_result=str(output/'TASK_VARIANT_RESULT.json'));write(status_path,status)
        return result
    finally:view.finished=True;view.refresh();view.close()
if __name__=='__main__':main()
