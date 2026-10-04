#!/usr/bin/env python3
"""Optional, result-free P62 bounded Panda development validation.

Preparing a freeze does not authorize simulation. run-cell additionally requires
root's post-offline authorization bound to that exact freeze. No automatic retry.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from prove.fast_adapter.p61_data import sha256
from tools.task_graph.train_precision_action_policy import dump_new

ROOT=Path(__file__).resolve().parents[2]
STEPS=240
SCHEMA='prove.p62-physical-development-freeze.v1'
APPROVAL_SCHEMA='prove.p62-root-physical-review.v1'
SPLITS=(('id',611100),('geometry',611200))
ENCODERS=('task_trained_lila',)
ARMS=('fixed','translation_size5')

def cells():
    return [dict(task='insert',split=split,condition_seed=base+i,encoder=encoder,arm=arm,video=i==0)
            for split,base in SPLITS for i in range(4) for encoder in ENCODERS for arm in ARMS]

def cell_name(index,cell):
    return f"{index:02d}_{cell['split']}_{cell['condition_seed']}_{cell['encoder']}_{cell['arm']}"

def binding(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=sha256(p))

def neural_state_sha256(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        value=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape,dtype=np.int64).tobytes())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def require_egl():
    for key,value in dict(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0').items():
        if os.environ.get(key)!=value:raise RuntimeError(f'{key}={value} must be set before Python starts')

def source_closure():
    pending=['tools/task_graph/p62_visual_validation.py','prove/fast_adapter/p62_runtime.py','tests/test_p62_visual_validation.py'];found={}
    while pending:
        relative=pending.pop()
        if relative in found:continue
        path=ROOT/relative
        if not path.is_file():continue
        found[relative]=sha256(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom):
                package=list(Path(relative).with_suffix('').parts[:-1])
                if node.level:
                    if node.level>len(package):raise ValueError('relative import escapes repository package')
                    base=package[:len(package)-node.level+1]
                    module='.'.join(base+(node.module.split('.') if node.module else []))
                else:module=node.module or ''
                modules=[module]+[module+'.'+alias.name for alias in node.names]
            for module in modules:
                if not module.startswith(('prove.','tools.')):continue
                parts=module.split('.')
                for i in range(1,len(parts)+1):
                    prefix='/'.join(parts[:i]);pending.extend([prefix+'.py',prefix+'/__init__.py'])
    return dict(sorted(found.items()))

def selected_candidate_configuration():
    from prove.fast_adapter.p62_adaptation import candidate_configuration
    return candidate_configuration('translation_size5')

def validate_qualification(path):
    value=json.loads(Path(path).read_text())
    if (value.get('passed') is not True or value.get('primary_model')!='task_trained_lila'
        or value.get('selected_candidate')!='translation_size5'):
        raise ValueError('P62 qualified primary candidate required')
    selected=next((r for r in value['candidate_checks'] if r['candidate']=='translation_size5'),None)
    if selected is None or selected['passed'] is not True or {r['distribution'] for r in selected['strata']}!={'id','geometry'} or not all(r['passed'] and all(r['checks'].values()) for r in selected['strata']):
        raise ValueError('P62 candidate failed a prescribed stratum gate')
    return value

def prepare(output,checkpoint,head,dino_dir,dataset_manifest,qualification,plan):
    require_egl()
    from prove.fast_adapter.geometry3d_env import configuration
    from prove.fast_adapter.robot_geometry_env_p45_v02 import source_assets
    from prove.task_graph.p59.core import FrozenP50Action
    from tools.task_graph.p50_precision_pilot import package_versions,BACKEND
    FrozenP50Action(head)  # Canonical frozen action checkpoint digest/schema.
    validate_qualification(qualification)
    if Path(plan).resolve()!=(ROOT/'evaluations/fast_adapter_p62_observable_update_v01/PLAN.md').resolve():raise ValueError('must bind adopted P62 plan')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    files=dict(action_head=binding(head),plan=binding(plan),dataset_manifest=binding(dataset_manifest),qualification=binding(qualification))
    qualification_dir=Path(qualification).parent
    report_path=qualification_dir/'REPORT.json';seal_path=qualification_dir/'PREDICTION_SEAL.json'
    report=json.loads(report_path.read_text());seal=json.loads(seal_path.read_text())
    if report['qualification']!=json.loads(Path(qualification).read_text()) or report['prediction_seal']['sha256']!=sha256(seal_path):raise ValueError('qualification/report seal mismatch')
    if seal['checkpoints']['task_trained_lila']['sha256']!=sha256(checkpoint) or seal['dataset_manifest']['sha256']!=sha256(dataset_manifest):raise ValueError('physical model/data differs from qualified offline candidate')
    offline_freeze=Path(seal['freeze']['path'])
    if sha256(offline_freeze)!=seal['freeze']['sha256']:raise ValueError('offline freeze binding changed')
    offline=json.loads(offline_freeze.read_text())
    for relative in ('prove/fast_adapter/p62_adaptation.py','prove/fast_adapter/p61_adaptation.py','prove/fast_adapter/p61_encoder.py','prove/task_graph/precision_action_policy.py'):
        if sha256(ROOT/relative)!=offline['source_bindings'][relative]:raise ValueError('qualified solver/model source changed')
    files.update(offline_report=binding(report_path),offline_prediction_seal=binding(seal_path),offline_freeze=binding(offline_freeze))
    for arm,path in zip(ENCODERS,[checkpoint]):
        payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if payload.get('schema')!='prove.p61-relation-checkpoint.v1' or payload['training']['arm']!=arm:
            raise ValueError('wrong final checkpoint identity')
        files[arm]=binding(path)
        for key in ('source','checkpoint'):
            name='lila_'+key;value=binding(payload['lila_binding'][key])
            if name in files and files[name]!=value:raise ValueError('different pretrained LiLa binding')
            files[name]=value
    for name in ('model.safetensors','config.json','preprocessor_config.json'):
        files['dino_'+name]=binding(Path(dino_dir)/name)
    manifest=json.loads(Path(dataset_manifest).read_text())
    if files['dino_model.safetensors']['sha256']!=manifest['dino_weights_sha256']:
        raise ValueError('runtime DINO differs from training cache')
    freeze=dict(schema=SCHEMA,status='frozen_before_results',phase='new_bounded_development',fresh_test_claim=False,
        cells=cells(),steps_per_episode=STEPS,automatic_retries=0,physics_workers_maximum=2,model_threads=1,renderer_gpu_ids=[0],
        model_device='cuda:0',files=files,sources=source_closure(),robot_assets=source_assets(),packages=package_versions(),backend=BACKEND,
        candidate_configuration=selected_candidate_configuration(),
        dino_dir=str(Path(dino_dir).resolve()),conditions=[dict(split=s,condition_seed=b+i,evaluator_configuration=configuration('insert',b+i,s)) for s,b in SPLITS for i in range(4)],
        runtime_contract=dict(inputs='448 RGB, visible role masks, calibration, own pose; no corner/PnP/target truth',
            held_length='env.calibrate_held_geometry public visible-mask+own-pose calibration',
            estimate='opening center3/rotation/halfsize2; outer contour offset only inside residual',
            anchor='first EEF pose fixed for episode',history='first12 plus latest20 admitted contours',
            online='persistent translation/size z5 in z8; rotation z strictly zero; 3 GN steps per valid frame after >=3 admitted contours',
            correction='center_anchor += .02*z_xyz; R_anchor = R_pred@Exp(.25*z_rot); loghalfsize += .25*z_size',
            deployment_difference='training updates only at t16,t24 with sparse support; physical candidate updates every valid frame with first12+latest20; bounded deployment extrapolation',fixed='zero z; reencode current observation each frame',
            prefix_steps=12,frozen='all DINO, LiLa, relation/readout and P50 weights'),
        evaluation_contract=dict(method='unchanged P47 evaluate_sealed after public predictions/trajectory seal',
            strict_native_success='last10 native successes and penetration<2mm',mechanical='P47 original arm<=1mrad/finger<=0.1mm/penetration<2mm finite',
            denominator='all16 prespecified cells including aborts',videos='offset0 in both splits; both arms',no_force_or_contact_change=True))
    dump_new(output/'PHYSICAL_FREEZE.json',freeze)
    for name in freeze['sources']:
        target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
        if sha256(target)!=freeze['sources'][name]:raise ValueError('source changed during snapshot')
    return freeze

def validate_freeze(path,approval=None,check_runtime=True):
    frozen=json.loads(Path(path).read_text())
    if (frozen['schema']!=SCHEMA or frozen['status']!='frozen_before_results' or frozen['cells']!=cells()
        or frozen['steps_per_episode']!=240 or frozen['automatic_retries']!=0 or frozen['physics_workers_maximum']!=2 or frozen['model_threads']!=1
        or frozen['candidate_configuration']!=selected_candidate_configuration()):
        raise ValueError('physical scope drift')
    validate_qualification(frozen['files']['qualification']['path'])
    if approval is not None:
        reviewed=json.loads(Path(approval).read_text())
        if (reviewed.get('schema')!=APPROVAL_SCHEMA or reviewed.get('freeze_sha256')!=sha256(path)
            or reviewed.get('physical_execution_authorized') is not True or reviewed.get('root_reviewed') is not True):
            raise ValueError('missing exact-freeze post-offline root authorization')
    if check_runtime:
        require_egl()
        from prove.fast_adapter.robot_geometry_env_p45_v02 import source_assets
        from tools.task_graph.p50_precision_pilot import package_versions
        if source_closure()!=frozen['sources'] or source_assets()!=frozen['robot_assets'] or package_versions()!=frozen['packages']:
            raise ValueError('source/assets/packages changed')
        for key,item in frozen['files'].items():
            if sha256(item['path'])!=item['sha256']:raise ValueError(f'changed frozen file: {key}')
    return frozen

def run_cell(freeze_path,approval_path,index):
    require_egl()
    if approval_path is None:raise ValueError('post-offline root authorization required before simulator construction')
    frozen=validate_freeze(freeze_path,approval_path)
    if not 0<=index<len(cells()):raise ValueError('cell outside bounded scope')
    torch.set_num_threads(1)
    from prove.fast_adapter.p62_runtime import P62Runtime,PUBLIC_POSE_KEYS
    from prove.fast_adapter.geometry3d_env import configuration
    from prove.fast_adapter.robot_geometry_env_p45_v02 import RobotGeometrySkill
    from prove.task_graph.precision_action_policy import PrecisionActionController,load_frozen_head,encode_state
    from tools.fast_adapter.p45_common import observe_roles
    from tools.fast_adapter.p47_evaluate import snapshot,evaluate_sealed
    from tools.task_graph.p50_precision_pilot import controller_state,domain_excursions
    cell=frozen['cells'][index];output=Path(freeze_path).parent/'cells'/cell_name(index,cell)
    output.mkdir(parents=True,exist_ok=False)
    dump_new(output/'PRE_RUN.json',dict(schema='prove.p62-physical-pre-run.v1',cell=cell,cell_index=index,
        freeze=binding(freeze_path),approval=binding(approval_path),steps=STEPS,automatic_retry=False))
    env=None;traces=[];observations=[];rgb_frames=[];mask_frames=[];videos=[];details=[];started=time.monotonic()
    try:
        runtime=P62Runtime(frozen['files'][cell['encoder']]['path'],frozen['dino_dir'],cell['arm'],device='cuda:0')
        head=load_frozen_head(frozen['files']['action_head']['path'])
        models=dict(encoder=runtime.encoder,dino=runtime.vision.model,action_head=head)
        before={k:neural_state_sha256(v) for k,v in models.items()}
        cfg=configuration('insert',cell['condition_seed'],cell['split'])
        expected=next(c['evaluator_configuration'] for c in frozen['conditions'] if c['condition_seed']==cell['condition_seed'])
        if cfg!=expected:raise ValueError('configuration drift')
        env=RobotGeometrySkill(cfg,render=True)
        if env.model.nmocap!=0 or env.model.nu<7:raise ValueError('articulated Panda required')
        calibration=env.calibrate_held_geometry()
        if calibration.get('source')!='RGB visible-mask bbox plus known own-tool pose':raise ValueError('public held-length calibration required')
        length=float(calibration['length']);controller=PrecisionActionController(head,env.start_pos,env.start_R,length=length)
        p,R=env.proprio();previous=np.r_[p,R.ravel(),1.]
        for t in range(STEPS):
            observed,rgb,masks=observe_roles(env)
            # Old backend may compute corners internally; explicitly discard them.
            public={key:copy.deepcopy(observed[key]) for key in PUBLIC_POSE_KEYS}
            admitted=runtime.update(public,rgb,masks,t)
            estimate=runtime.estimate();proprio=env.proprio()
            observations.append(dict(public,t=t,previous_action=previous.copy(),contour_admitted=admitted))
            if cell['video'] or t==0:rgb_frames.append(rgb.copy());mask_frames.append(masks.copy())
            pre_state=controller_state(controller)
            phase='align' if controller.phase=='observe' else controller.phase
            domain=domain_excursions(encode_state(estimate,proprio,length,phase)) if t>=12 and estimate is not None else None
            p,R,closed,stage=controller.command(t,estimate,proprio)
            row=dict(t=t,state=snapshot(env),command_p=p.copy(),command_R=R.copy(),closed=bool(closed),stage=stage,
                estimate=copy.deepcopy(estimate),proprio=copy.deepcopy(proprio),adaptation=copy.deepcopy(runtime.records[-1]),
                controller_before=pre_state,controller_after=controller_state(controller),head_domain=domain,
                neural_action=copy.deepcopy(controller.records[-1]) if controller.records and controller.records[-1]['t']==t else None,
                controls_before=env.data.ctrl.copy())
            if cell['video']:videos.append(np.concatenate((env.render_global(),rgb),axis=1));details.append(env.render_detail())
            env.step(p,R,closed)
            row.update(controls_after=env.data.ctrl.copy(),robot_tracking=env.robot_tracking_diagnostics());traces.append(row)
            previous=np.r_[p,R.ravel(),float(closed)]
        terminal=snapshot(env)
        for name,data in [('TRACE.npy',traces),('OBS.npy',observations),('PREDICTIONS.npy',runtime.records)]:
            np.save(output/name,np.array(data,dtype=object),allow_pickle=True)
        np.save(output/'TERMINAL.npy',terminal,allow_pickle=True)
        np.savez_compressed(output/'PUBLIC_PIXELS.npz',rgb=np.asarray(rgb_frames),visible_masks=np.asarray(mask_frames))
        filenames=['TRACE.npy','OBS.npy','PREDICTIONS.npy','TERMINAL.npy','PRE_RUN.json','PUBLIC_PIXELS.npz']
        if cell['video']:
            for name,data in [('FRAMES.npz',videos),('DETAIL_FRAMES.npz',details)]:np.savez_compressed(output/name,frames=np.asarray(data));filenames.append(name)
        after={k:neural_state_sha256(v) for k,v in models.items()}
        if before!=after:raise RuntimeError('frozen weights changed before evaluator')
        validate_freeze(freeze_path,approval_path)
        dump_new(output/'SEAL.json',dict(schema='prove.p62-trajectory-seal.v1',freeze=binding(freeze_path),
            files={name:sha256(output/name) for name in filenames},neural_parameters_before=before,neural_parameters_after=after))
        facts=evaluate_sealed(env,traces,terminal,output)
        result=dict(schema='prove.p62-physical-result.v1',cell=cell,cell_index=index,**facts,steps=STEPS,
            seal_sha256=sha256(output/'SEAL.json'),frozen_neural_weights_unchanged=True,
            gradient_steps=sum(r['U'] for r in runtime.records),admitted_contour_frames=sum(r['admitted'] for r in runtime.records),
            inactive_rotation_z_exact_zero=all(np.array_equal(r['z_after'][3:6],np.zeros(3)) for r in runtime.records),
            runtime_public_fields=list(PUBLIC_POSE_KEYS),runtime_pixel_inputs=['rgb','actor_mask','held_object_mask','goal_mask'],
            neural_precision_commands=sum(r['neural_action'] is not None for r in traces),
            final_controller_phase=controller.phase,tool_calibration={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in calibration.items()},
            elapsed_seconds=time.monotonic()-started,fresh_test_claim=False)
        dump_new(output/'RESULT.json',result);return result
    except Exception as exc:
        for name,data in [('TRACE_PARTIAL.npy',traces),('OBS_PARTIAL.npy',observations)]:np.save(output/name,np.array(data,dtype=object),allow_pickle=True)
        partial={name:sha256(output/name) for name in ('TRACE_PARTIAL.npy','OBS_PARTIAL.npy')}
        if rgb_frames:
            np.savez_compressed(output/'PUBLIC_PIXELS_PARTIAL.npz',rgb=np.asarray(rgb_frames),visible_masks=np.asarray(mask_frames));partial['PUBLIC_PIXELS_PARTIAL.npz']=sha256(output/'PUBLIC_PIXELS_PARTIAL.npz')
        if env is not None:
            np.save(output/'STATE_AT_ABORT.npy',snapshot(env),allow_pickle=True);partial['STATE_AT_ABORT.npy']=sha256(output/'STATE_AT_ABORT.npy')
        dump_new(output/'ABORT.json',dict(cell=cell,type=type(exc).__name__,reason=str(exc),completed_steps=len(traces),partial_files=partial,automatic_retry=False,task_success_denominator_included=True))
        raise
    finally:
        if env is not None:env.close()

def summarize(path):
    frozen=validate_freeze(path,check_runtime=False);rows=[]
    for i,cell in enumerate(frozen['cells']):
        directory=Path(path).parent/'cells'/cell_name(i,cell);result=directory/'RESULT.json';abort=directory/'ABORT.json'
        if result.exists():
            record=json.loads(result.read_text());seal=json.loads((directory/'SEAL.json').read_text())
            if record['cell']!=cell or record['seal_sha256']!=sha256(directory/'SEAL.json') or seal['freeze']['sha256']!=sha256(path):raise ValueError('result seal drift')
            if seal['neural_parameters_before']!=seal['neural_parameters_after']:raise ValueError('weight drift')
            for name,digest in seal['files'].items():
                if sha256(directory/name)!=digest:raise ValueError('trajectory changed')
            rows.append(dict(cell=cell,status='complete',result=record))
        elif abort.exists():rows.append(dict(cell=cell,status='abort',abort=json.loads(abort.read_text())))
        else:rows.append(dict(cell=cell,status='not_complete'))
    return dict(schema='prove.p62-physical-summary.v1',expected=16,completed=sum(r['status']=='complete' for r in rows),
        aborted=sum(r['status']=='abort' for r in rows),cells=rows,fresh_test_claim=False)

def main():
    p=argparse.ArgumentParser();subs=p.add_subparsers(dest='mode',required=True)
    prep=subs.add_parser('prepare')
    for name in ('output','checkpoint','head','dino-dir','dataset-manifest','qualification','plan'):prep.add_argument('--'+name,required=True)
    run=subs.add_parser('run-cell');run.add_argument('--freeze',required=True);run.add_argument('--approval',required=True);run.add_argument('--index',required=True,type=int)
    summary=subs.add_parser('summarize');summary.add_argument('--freeze',required=True)
    a=p.parse_args()
    if a.mode=='prepare':value=prepare(a.output,a.checkpoint,a.head,a.dino_dir,a.dataset_manifest,a.qualification,a.plan)
    elif a.mode=='run-cell':value=run_cell(a.freeze,a.approval,a.index)
    else:value=summarize(a.freeze)
    print(json.dumps(value,indent=2,allow_nan=False))
if __name__=='__main__':main()
