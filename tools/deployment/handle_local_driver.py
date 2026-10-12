#!/usr/bin/env python3
"""Frozen simulator-only handle candidate, default file check; Root runs --execute."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE))
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import RealRootJudge,write,digest
from PIL import Image
from prove.xtrainer.perception import LiveVision
from shared_cup_entry import load,bindings
from shared_cup_runtime import SharedCupExecutor
from handle_local_candidate import observed_patch_cloud,tube_relation,pad_relation_from_robot_mesh,candidate_goal,lift_goal_from_visible_height

VARIANT='handle_local_Code_candidate_1_v1'


class HandleVision(LiveVision):
    def __init__(self,camera,*args,**kwargs):
        owner=self
        class Tap:
            def capture(self):
                frame=camera.capture();owner.latest_depth=frame[2].copy();owner.latest_K=frame[3].copy();return frame
            def close(self):camera.close()
        self.evidence_root=kwargs.pop('evidence_root',None);self.mask_records=[];self.frame_count=0
        super().__init__(Tap(),*args,**kwargs);self.raw_camera=camera;self.latest_masks=None
    def _tracker(self):
        tracker=super()._tracker();owner=self
        class Tap:
            def initialize(self,*args):
                masks=tracker.initialize(*args);owner.latest_masks={k:np.asarray(v,bool).copy() for k,v in masks.items()};return masks
            def track(self,*args):
                masks=tracker.track(*args);owner.latest_masks={k:np.asarray(v,bool).copy() for k,v in masks.items()};return masks
        return Tap()
    def capture(self):
        obs=super().capture()
        if self.evidence_root is not None and (self.frame_count==0 or self.frame_count%10==0):
            root=Path(self.evidence_root);root.mkdir(parents=True,exist_ok=True)
            rgb=root/f'RGB_{self.frame_count:06d}.png';Image.fromarray(obs.rgb).save(rgb)
            masks=root/f'MASKS_{self.frame_count:06d}.npz';np.savez_compressed(masks,**self.latest_masks)
            data=root/f'RGBD_{self.frame_count:06d}.npz';np.savez_compressed(data,depth_m=self.latest_depth,K=self.latest_K,T=self.T)
            self.mask_records.append({'frame_index':self.frame_count,'stamp':obs.stamp,'physical_time_sec':self.raw_camera.last_physical_time,
                'role_points':{k:None if v is None else np.asarray(v).tolist() for k,v in obs.roles.items()},
                'files':[{'path':str(p),'sha256':digest(p)} for p in (rgb,masks,data)],'mask_source':'actual_SAM2','GT_fallback':False})
            write(root/'MASK_EVIDENCE.json',{'frames':self.mask_records,'actual_SAM':True,'functional_parts_are_mask_intersect_Root_patch':True})
        self.frame_count+=1
        # Functional patches are local planning inputs, not rigid-object drift roles.
        obs.roles={k:v for k,v in obs.roles.items() if k in ('white_mug','red_mug','box')}
        return obs
    def cloud(self,role):
        boxes=self.config['role_boxes']
        if role not in boxes or self.latest_masks is None or role not in self.latest_masks:raise SafetyError('current Root functional patch missing: '+role)
        box=boxes[role] if role in ('white_handle','table_patch') else [0,0,self.latest_depth.shape[1],self.latest_depth.shape[0]]
        return observed_patch_cloud(self.latest_depth,self.latest_masks[role],self.latest_K,self.T,box)


class LocalExecutor(SharedCupExecutor):
    def _reached(self,goal,pose):
        error=self._error(goal,pose)
        return bool(np.linalg.norm(error[:3])<=getattr(self,'entry_position_tolerance_m',.001) and np.linalg.norm(error[3:])<=.02)
    def refresh_lift_attachment(self,*args):pass # no reference-trajectory attachment mapping in this local primitive
    def node(self,phase):
        self.stage=copy.deepcopy(next(s for s in self.b['graph']['stages'] if s['cycle_id']=='c03' and s['phase']==('Lift' if phase=='Hold' else phase)))
        self.stage.update(stage_id='handle01.'+phase.lower(),phase=phase,holding_contract='none' if phase=='Approach' else 'bilateral_contact')
        self.stage_started=self.clock();write(self.output/'NODE_STATUS.json',{'stage_id':self.stage['stage_id'],'phase':phase,'candidate':1,'local_only':True})
    def confirm(self,facts):
        after=self.semantics(self.stage,facts,self.capture())
        if any(self.visual_facts.get(k) is not True for k in facts):raise SafetyError('Root local node not confirmed: '+str(facts))
        return after
    def dispatch(self,goal,g):
        self.guard_limits();samples,obs=self.capture();actual=np.stack([s.q for s in samples]);target,pose,error=self._step_ik(actual,g,'left',goal)
        fresh,frame=self.capture();self.safety.validate(target,g,fresh,.1,self.robot,self.T)
        final=self.robot.samples();self.safety.samples(final)
        if self.clock()-frame.stamp>self.cfg['safety']['max_age_sec']:raise SafetyError('fresh handle observation expired before command')
        if np.max(abs(target-np.stack([s.q for s in final])))>.01+1e-9:raise SafetyError('actual pose changed during local IK')
        self._hardware_attempts+=1;intent={'attempt_count':self._hardware_attempts,'acknowledged_count':self._hardware_acknowledged,'stage_id':self.stage['stage_id'],'q_target_rad':target.tolist(),'gripper_command':g.tolist(),'flags':[True,False],'partial_motion_possible_without_ack':True}
        write(self.output/'COMMAND_INTENT.json',intent);self.robot.command(target,g,[True,False]);self._hardware_acknowledged+=1
        self.safety.last_target=target.copy();self.command_g=g.copy();self.command_index+=1;self.sleep(.1)
        self.trace.append({'stage_id':self.stage['stage_id'],'goal_TCP_m':goal[0].tolist(),'goal_rotation':goal[1].tolist(),'actual_q':actual.tolist(),'g':g.tolist(),'error_before_command':error.tolist(),'command_count':self.command_index})
    def reach(self,goal,g):
        for _ in range(self.cfg.get('maximum_node_control_updates',1000)):
            samples,obs=self.capture();q=np.stack([s.q for s in samples])
            if self._reached(goal,self._shared_pose(q,self.command_g,'left')):return samples,obs
            self.dispatch(goal,g)
        raise SafetyError('local Cartesian control budget exhausted')
    def local_run(self):
        failure=None;completed=[];hold_elapsed=0.
        try:
            self.start();self.node('Approach');samples,obs=self.capture()
            if not {'white_handle','table_patch'}<=set(self.vision.config['role_boxes']):raise SafetyError('Root handle/table fresh patch grounding required')
            handle=self.vision.cloud('white_handle');table=self.vision.cloud('table_patch');relation=tube_relation(handle,self.vision.T[:3,3])
            if relation['radial_extent_m']<.0002:raise SafetyError('observed tube radius unresolved')
            self.entry_position_tolerance_m=min(.001,relation['radial_extent_m']/4.)
            q=np.stack([s.q for s in samples])
            self.robot.fk(q,self.command_g,'left')
            with self.robot.fk_lock:pads=pad_relation_from_robot_mesh(self.robot.world.model,self.robot.scratch,'left')
            goal=candidate_goal(relation,pads)
            table_z=float(np.median(table[:,2]));pad_bottom=float(np.min((pads['pad_vertices_TCP_local_m']@goal[1].T+goal[0])[:,2]))
            if pad_bottom<=table_z:raise SafetyError('robot pad candidate reaches observed table plane')
            serial=lambda v:v.tolist() if isinstance(v,np.ndarray) else v
            write(self.output/'CANDIDATE_PLAN.json',{'candidate':1,'relation':{k:serial(v) for k,v in relation.items()},'pads':{k:serial(v) for k,v in pads.items()},'TCP_goal':goal[0].tolist(),'R_goal':goal[1].tolist(),'observed_table_z_m':table_z,'entry_position_tolerance_m':self.entry_position_tolerance_m,'no_hidden_object_geometry':True})
            open_g=self.command_g.copy();open_g[0]=max(open_g[0],self.b['initial_g'][0])
            pre=(goal[0]-relation['tangent_down']*relation['span_m'],goal[1]);self.reach(pre,open_g);self.reach(goal,open_g)
            self.confirm(['align_ready']) # local handle-entry alignment only, not placement
            completed.append('Approach')
            self.node('Grasp');g=self.command_g.copy();g[0]=0.
            self.dispatch(goal,g) # unchanged standard full-close command; wait for actual finger settling
            self.sleep(.5);self.confirm(['holding']);completed.append('Grasp')
            self.node('Lift');samples,obs=self.capture();actual=(samples[0].tcp.copy(),samples[0].rotation.copy())
            lift,measurement=lift_goal_from_visible_height(actual,self.vision.cloud('white_mug'),self.vision.cloud('table_patch'))
            write(self.output/'LIFT_MEASUREMENT.json',measurement);self.reach(lift,g);self.confirm(['holding','lifted']);completed.append('Lift')
            self.node('Hold');started=self.clock()
            while self.clock()-started<60.:
                samples,obs=self.capture()
                if obs.roles.get('white_mug') is None:raise SafetyError('held object visual observation missing')
                if np.max(abs(samples[0].q-self.safety.last_target[0]))>self.cfg['safety']['max_tracking_rad']:raise SafetyError('actual robot moved during continuous hold')
                self.sleep(.2);hold_elapsed=self.clock()-started
            self.confirm(['holding','lifted']);completed.append('Hold')
        except Exception as exc:failure=str(exc)
        stop_results=[]
        for stop in self.stops:
            try:stop.stop();stop_results.append({'stopped':True})
            except Exception as exc:stop_results.append({'error':str(exc)})
        return {'variant':VARIANT,'candidate':1,'local_completed':completed==['Approach','Grasp','Lift','Hold'],'graph_completed':False,'whole_task_success_claim':False,'failure':failure,'completed_local_nodes':completed,'continuous_hold_observed_wall_sec':hold_elapsed,'simulation_command_attempts':self._hardware_attempts,'simulation_command_acknowledgments':self._hardware_acknowledged,'hardware_motion_commands':0,'trace':self.trace,'Root_records':self.judge.records,'stop_results':stop_results,'mask_source':'actual_SAM2_intersect_Root_visible_patch','g_close_command':0.,'object_CAD_or_GT_control':False}


def qualify_after_seal(robot,trace,result):
    # Called only after PREDICTIONS_SEALED and physics-thread termination.
    model=robot.world.model;mj=robot.mj;data=mj.MjData(model);end=float(trace[-1]['physical_time_sec'])
    selected=[x for x in trace if x['physical_time_sec']>=end-60.1];positions=[];rotations=[];heights=[];bilateral=[]
    initial=np.asarray(trace[0]['qpos']);white=model.body('object_white_mug').id
    for row in selected:
        data.qpos[:]=row['qpos'];data.qvel[:]=row['qvel'];data.ctrl[:]=row['ctrl'];mj.mj_forward(model,data)
        tcp=data.site('left_tcp');R=tcp.xmat.reshape(3,3);body=data.body('object_white_mug')
        positions.append(R.T@(body.xpos-tcp.xpos));rotations.append(body.xquat.copy());heights.append(float(body.xpos[2]-initial[model.joint('object_white_mug_free').qposadr[0]+2]))
        fingers=set()
        for contact in data.contact:
            gids=[int(contact.geom1),int(contact.geom2)];bodies=[int(model.geom_bodyid[g]) for g in gids]
            if white not in bodies:continue
            other=gids[1-bodies.index(white)];name=model.geom(other).name
            if name.startswith('left_J1_7_'):fingers.add(1)
            if name.startswith('left_J1_8_'):fingers.add(2)
        bilateral.append(len(fingers)==2)
    coverage=float(selected[-1]['physical_time_sec']-selected[0]['physical_time_sec'])
    drift=float(np.max(np.linalg.norm(np.asarray(positions)-positions[0],axis=1)))
    final=robot.world.contact_evidence()['roles']['white_mug']
    passed=bool(coverage>=59.8 and all(bilateral) and min(heights)>.005 and drift<=.005 and final['held_by']['left'] and not final['object_table_support_contact'])
    return {'passed':passed,'private_post_seal_only':True,'control_input':False,'physical_window_sec':coverage,'sample_count':len(selected),
            'geometric_bilateral_contact_all_samples':all(bilateral),'geometric_bilateral_sample_fraction':float(np.mean(bilateral)),
            'final_original_private_contact_evidence':final,'minimum_body_height_above_native_initial_m':min(heights),'maximum_body_relative_TCP_drift_m':drift,
            'contract':'last60s sampled unpaused physics; original final held flag plus bilateral-contact geometry every sample, >5mm body rise, <=5mm body/TCP drift; not full-task goal',
            'sampling_limitation':'contact geometry sampled at100ms, original force-based held evidence at final state; no claim of measured force at every sample'}


def prepared(cfg_path):
    cfg=json.loads(cfg_path.read_text());report,effective,bundle,bound=bindings(cfg_path,cfg,'sim')
    for p in [Path(__file__),HERE/'handle_local_candidate.py',REPO/'tests/test_handle_local_candidate.py'] :bound[str(p.resolve())]=digest(p)
    return cfg,effective,bundle,bound


def guard(scope_path,freeze_path,bound,output):
    if not scope_path or not freeze_path:raise SafetyError('local scope and freeze required')
    scope=json.loads(scope_path.read_text());freeze=json.loads(freeze_path.read_text());job={'task':'put_mug_on_box','kind':'handle_local','candidate':1,'mode':'sim','output':str(output.resolve())}
    if scope.get('execution_authorized') is not True or scope.get('mode')!='sim' or scope.get('hardware_motion_jobs')!=0 or scope.get('maximum_local_candidate_episodes',0)<1:raise SafetyError('local simulator candidate scope absent')
    if freeze.get('schema')!='graph-self-adapter.handle-local-freeze.v1' or freeze.get('status')!='result_free_frozen' or freeze.get('jobs')!=[job] or freeze.get('scope_sha256')!=digest(scope_path):raise SafetyError('exact local candidate result-free freeze absent')
    actual={str(Path(x['path']).resolve()):x['sha256'] for x in freeze['bindings']};required={**bound,str(scope_path.resolve()):digest(scope_path)}
    if len(actual)!=len(freeze['bindings']) or not required.items()<=actual.items() or any(digest(p)!=sha for p,sha in actual.items()):raise SafetyError('local candidate source/evidence binding changed')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--scope',type=Path);p.add_argument('--execution-freeze',type=Path);p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args(argv)
    cfg,runtime,bundle,bound=prepared(a.config)
    if not a.execute:
        ready={'variant':VARIANT,'hardware_connections':0,'episode_started':False,'required_bindings':[{'path':p,'sha256':sha} for p,sha in sorted(bound.items())],'freeze_job':{'task':'put_mug_on_box','kind':'handle_local','candidate':1,'mode':'sim','output':'ABSOLUTE_FRESH_OUTPUT'}};print(json.dumps(ready,indent=2));return ready
    if not a.output or a.output.exists():raise SafetyError('fresh explicit local output required')
    guard(a.scope,a.execution_freeze,bound,a.output);os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    sim=load(HERE/'sim_robot_io.py','_handle_sim_io');warm=load(HERE/'operator_launcher.py','_handle_startup').startup_vision_class(HandleVision,SafetyError)
    robot=sim.SimRobot(scene_run=cfg['sim']['scene_run'],physics_source=cfg['sim']['physics_source'],bundle=bundle,T_left_from_right=runtime['T_left_from_right'])
    camera=sim.SimRGBDCamera(robot);vision=None
    try:
        camera.connect();robot.start();runtime.update(mode='sim',known_initial_gripper_commands=bundle['initial_g'].tolist())
        vision=warm(camera,camera.T,runtime['vision'],evidence_root=a.output/'SAM_MASK_EVIDENCE');judge=RealRootJudge(a.output/'REAL_AGENT_HANDOFFS',task='put_mug_on_box',graph_sha=digest(bundle['paths']['graph']),identity=runtime['root_identity'])
        ex=LocalExecutor(bundle,robot,vision,judge,[sim.SimStop(robot,arm) for arm in ('left','right')],runtime,a.output);result=ex.local_run()
        write(a.output/'PREDICTIONS_SEALED.json',{'result':result,'sealed_before_private_evaluator':True});robot.stop_event.set();robot.thread.join(timeout=3)
        trace=robot.world.evaluator_trace();np.savez_compressed(a.output/'STATE_TRACE.npz',timestamps=[x['physical_time_sec'] for x in trace],qpos=[x['qpos'] for x in trace],qvel=[x['qvel'] for x in trace],ctrl=[x['ctrl'] for x in trace])
        qualification=qualify_after_seal(robot,trace,result)
        write(a.output/'LOCAL_PRIVATE_QUALIFICATION.json',qualification)
        result['post_seal_local_qualification']=qualification
        result['local_qualified']=bool(result['local_completed'] and qualification['passed'])
        result.update(physics_integrity=robot.integrity(),simulation_physics_continues_during_Root=True,post_seal_state_trace_sha256=digest(a.output/'STATE_TRACE.npz'))
        write(a.output/'RESULT.json',result);return result
    finally:
        if vision:vision.close()
        elif camera.renderer:camera.close()
        robot.close()

if __name__=='__main__':main()
