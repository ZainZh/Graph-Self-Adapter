#!/usr/bin/env python3
"""Two original-initial-state P52 engineering episodes; no terminal-state restore.

s1..s6 preserve frozen P51 A2 code/MLP and supervisor. Only s7 uses the new
frozen stage policy, fixed or observed-feedback z. GT is supervisor-only there.
Prepare is simulator-free; execution needs a separately reviewed freeze.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from prove.task_graph.schema import TaskGraph
from prove.task_graph.executor import GraphExecutor
from prove.task_graph.libero_precision_p51 import MixedController,MixedRouter
from prove.task_graph.libero_supervisor import LiberoSupervisor,contact_facts
from prove.task_graph.p52_stage_policy import Config,StagePolicy,StageRuntime
from tools.task_graph.run_libero_p51_v02 import grounding,insert_entry_domain
from tools.task_graph.collect_libero_p52_variants import verify as verify_collection
from tools.task_graph.collect_libero_demo_gt import source_demo,snapshot,dump
from tools.task_graph.collect_libero_p52 import PublicRenderer
from tools.task_graph.p52_policy_pilot import Vision,validate_features
from tools.task_graph.p52_deployment import source_closure
from tools.task_graph.p52_pack_dataset import sha

ROOT=Path(__file__).resolve().parents[2]


class InsertJudge:
    """New declared local structural rule; never feeds action or adaptation loss."""
    def __init__(self):self.no_contact=0;self.goal_hold=0
    def observe(self,official,fingers,wrong_target):
        if fingers is None or wrong_target is None:
            return dict(stop='structural_measurement_unavailable',complete=False,attachment='UNKNOWN',no_contact_streak=self.no_contact,goal_hold=self.goal_hold)
        attached=any(fingers);self.no_contact=0 if attached else self.no_contact+1
        # Three goal observations must also retain some book contact; a dropped
        # book reaching the region cannot beat the five-frame loss stop.
        self.goal_hold=self.goal_hold+1 if official and attached and not wrong_target else 0
        stop=('wrong_target' if wrong_target else 'book_contact_lost_5' if self.no_contact>=5 else
              'official_goal_contact_hold' if self.goal_hold>=3 else None)
        return dict(stop=stop,complete=stop=='official_goal_contact_hold',
                    attachment='attached_bilateral' if all(fingers) else 'UNKNOWN_single_sided' if attached else 'no_book_contact',
                    no_contact_streak=self.no_contact,goal_hold=self.goal_hold)


def prepare(collection_binding,checkpoint,dino,lila,vision_python,vision_device):
    parent=json.loads(collection_binding.read_text());verify_collection(parent)
    package=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if package['schema']!='p52_stage_skill_v01' or package['settings']['training'].get('ordinary',False):
        raise ValueError('Require the final joint-trained stage checkpoint')
    model=StagePolicy(Config(**package['config']));model.load_state_dict(package['model'],strict=True)
    code='import json,importlib.metadata as m;print(json.dumps({n:m.version(n) for n in ("torch","transformers","numpy")}))'
    packages=json.loads(subprocess.check_output([vision_python or sys.executable,'-c',code],text=True))
    assets=[dino/'model.safetensors',dino/'config.json',dino/'preprocessor_config.json',lila/'VISUAL_WEIGHTS.pt',lila/'models/vla_model_fm.py']
    return dict(schema='p52.full-original-task.v1',phase='development_engineering_only',
        parent=dict(path=str(collection_binding.resolve()),sha256=sha(collection_binding)),environment=parent['environment'],prefix=parent['prefix'],public=parent['public'],
        checkpoint=dict(path=str(checkpoint.resolve()),sha256=sha(checkpoint)),
        sources={p:sha(ROOT/p) for p in source_closure(ROOT,['tools/task_graph/p52_full_task.py','tools/task_graph/p52_vision_worker.py'])},
        visual_assets={str(p.resolve()):sha(p) for p in assets},
        packages={n:importlib.metadata.version(n) for n in ('torch','numpy','scipy','mujoco')},
        vision=dict(dino=str(dino.resolve()),lila=str(lila.resolve()),python=vision_python,device=vision_device,packages=packages),
        cells=[dict(index=0,arm='fixed'),dict(index=1,arm='online')],seed=52,
        prefix_max_actions=400,insert_max_actions=180,original_initial_restores=1,later_state_restores=0,
        grasp_offset=[0,0,0],entry_offset=[0,0,0],entry_transition_actions=0,
        prefix_contract='Frozen P51 A2 s1..s6 MixedController, GT supervisor, public Insert entry guard; no predicate relaxation',
        insert_contract='Public pretrained visual/proprio model only; fixed closed; stage z initialized once, networks frozen',
        structural_rule='wrong_target immediate stop; zero book contacts on both fingers for five consecutive frames stops; single-sided UNKNOWN continues and is logged',
        completion='native official goal AND at least one finger touching book for three consecutive observed frames; then s7 and graph complete',
        automatic_retries=0,independent_statistical_evidence=False)


def verify(binding):
    if binding['schema']!='p52.full-original-task.v1' or binding['cells']!=[dict(index=0,arm='fixed'),dict(index=1,arm='online')]:raise ValueError('Wrong full-task scope')
    for key,value in dict(prefix_max_actions=400,insert_max_actions=180,original_initial_restores=1,later_state_restores=0,entry_transition_actions=0).items():
        if binding[key]!=value:raise ValueError('Full-task budget/init changed')
    if binding['grasp_offset']!=[0,0,0] or binding['entry_offset']!=[0,0,0]:raise ValueError('Perturbations prohibited')
    if sha(binding['parent']['path'])!=binding['parent']['sha256']:raise ValueError('Parent changed')
    parent=json.loads(Path(binding['parent']['path']).read_text());verify_collection(parent)
    if binding['environment']!=parent['environment'] or binding['prefix']!=parent['prefix'] or binding['public']!=parent['public']:raise ValueError('Frozen original environment/prefix differs')
    for name,digest in binding['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Source changed: {name}')
    for name,digest in binding['visual_assets'].items():
        if sha(name)!=digest:raise ValueError(f'Visual asset changed: {name}')
    if sha(binding['checkpoint']['path'])!=binding['checkpoint']['sha256']:raise ValueError('Stage model changed')
    for name,version in binding['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Runtime package changed')


def initialize(binding,output):
    environment=binding['environment'];config=output/'libero_config';config.mkdir()
    base=Path(environment['libero_root'])/'libero/libero'
    dump(config/'config.yaml',dict(benchmark_root=str(base),bddl_files=str(base/'bddl_files'),init_states=str(base/'init_files'),
         datasets=str(Path(environment['source']).parent),assets=str(base/'assets')))
    os.environ['LIBERO_CONFIG_PATH']=str(config.resolve());os.environ.setdefault('MUJOCO_GL','egl');sys.path.insert(0,environment['libero_root'])
    from libero.libero.envs import TASK_MAPPING
    kwargs=dict(environment['env_kwargs']);kwargs.update(use_camera_obs=False,has_offscreen_renderer=False)
    env=TASK_MAPPING[environment['env_args_original']['problem_name']](**kwargs)
    try:
        env.reset();env.reset_from_xml_string(environment['relocated_xml']);env.sim.reset()
        env.sim.set_state_from_flattened(source_demo(environment['source'])['initial'])
        env.sim.forward();env._post_process();env._update_observables(force=True)
        return env
    except Exception:env.close();raise


def run_prefix(env,binding,output,instance):
    prefix=binding['prefix'];graph=TaskGraph.from_dict(prefix['runtime_graph'])
    if [s.id for s in graph.nodes]!=[f's{i}' for i in range(1,8)]:raise ValueError('Unexpected original graph')
    skill=MixedController(binding['environment']['source'],graph,prefix['checkpoint'])
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


def structural_measurement(env,names):
    contacts=[dict(geom1=int(env.sim.data.contact[i].geom1),geom2=int(env.sim.data.contact[i].geom2)) for i in range(env.sim.data.ncon)]
    fingers,wrong=contact_facts(names,contacts)
    return bool(env._check_success()),fingers,wrong


def run_insert(env,binding,runtime,vision,renderer,output,names):
    judge=InsertJudge();obs=env._get_observations(force_update=True);previous=action_previous=None;actions=0
    while True:
        path=output/'public'/f'frame_{actions:04d}.npz';np.savez_compressed(path,**renderer.observe(obs))
        features=vision.encode(path);validate_features(features,runtime.model.config)
        torch.save(features,output/'public'/f'features_{actions:04d}.pt')
        if previous is None:runtime.enter(features)
        z_before=runtime.z.clone();loss=None
        if previous is not None:loss=runtime.feedback(previous,action_previous,features)
        action=runtime.command(features).numpy()
        if action.shape!=(7,) or not np.isfinite(action).all() or action[6]!=1:raise ValueError('Invalid neural OSC command')
        official,fingers,wrong=structural_measurement(env,names)
        judgment=judge.observe(official,fingers,wrong)
        stop=judgment['stop'] or ('insert_budget' if actions>=binding['insert_max_actions'] else None)
        record=dict(t=actions,stage_id='s7',public_observation=path.name,sha256=sha(path),z_before=z_before.tolist(),z_after=runtime.z.tolist(),
                    inner_loss=loss,updates=runtime.updates,action=action.tolist(),executed=stop is None)
        with (output/'public/ACTIONS.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        arrays,_=snapshot(env,obs);arrays={k:v for k,v in arrays.items() if k.startswith('supervisor_')}
        np.savez_compressed(output/'private'/f'frame_{actions:04d}.npz',**arrays)
        with (output/'private/JUDGMENTS.jsonl').open('a') as f:
            f.write(json.dumps(dict(t=actions,official_success=official,finger_contacts=fingers,wrong_target=wrong,**judgment),allow_nan=False)+'\n')
        if stop:return dict(actions=actions,graph_complete=judgment['complete'],official_success=official,final_judgment=judgment,stop_reason=stop,updates=runtime.updates)
        previous=features;action_previous=torch.tensor(action[:6],dtype=torch.float32)
        obs,_,_,_=env.step(action.copy());actions+=1


def run(binding_path,index,output):
    binding=json.loads(binding_path.read_text());verify(binding)
    if index not in (0,1):raise ValueError('Only fixed/online original-state pair is authorized')
    output.mkdir(parents=True,exist_ok=False)
    for folder in ('prefix','public','private'):(output/folder).mkdir()
    dump(output/'BINDING.json',binding);cell=binding['cells'][index];dump(output/'CELL.json',cell)
    torch.set_num_threads(1);torch.manual_seed(binding['seed']);np.random.seed(binding['seed'])
    ck=torch.load(binding['checkpoint']['path'],map_location='cpu',weights_only=True)
    model=StagePolicy(Config(**ck['config']));model.load_state_dict(ck['model'],strict=True);runtime=StageRuntime(model,online=cell['arm']=='online')
    before={k:v.clone() for k,v in model.state_dict().items()};env=renderer=vision=None
    try:
        env=initialize(binding,output)
        names={kind:[getattr(env.sim.model,kind+'_id2name')(i) for i in range(getattr(env.sim.model,count))] for kind,count in [('body','nbody'),('geom','ngeom')]}
        dump(output/'private/NAMES.json',names)
        prefix=run_prefix(env,binding,output/'prefix',output.name);dump(output/'PREFIX_RESULT.json',prefix)
        if prefix['qualified']:
            vision=Vision(binding['vision']);dump(output/'VISUAL_BINDING.json',vision.binding)
            renderer=PublicRenderer(env,binding['public'])
            result=run_insert(env,binding,runtime,vision,renderer,output,names)
        else:result=dict(actions=0,graph_complete=False,official_success=bool(env._check_success()),stop_reason=prefix['stop_reason'],updates=0)
        if any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):raise RuntimeError('Stage network mutated')
        verify(binding)
        files={str(p.relative_to(output)):sha(p) for folder in ('prefix','public','private') for p in sorted((output/folder).iterdir())}
        files['PREFIX_RESULT.json']=sha(output/'PREFIX_RESULT.json');files['CELL.json']=sha(output/'CELL.json')
        dump(output/'SEAL.json',dict(binding_sha256=sha(output/'BINDING.json'),files=files,cell=cell,original_restores=1,later_restores=0))
        result.update(cell=cell,prefix_actions=prefix['actions'],total_actions=prefix['actions']+result['actions'],
                      seal_sha256=sha(output/'SEAL.json'),scope='complete original task development engineering; not independent statistical evidence',automatic_retry=False)
        dump(output/'RESULT.json',result);return result
    except Exception as e:
        dump(output/'ABORT.json',dict(error=type(e).__name__,reason=str(e),cell=cell,automatic_retry=False));raise
    finally:
        runtime.exit()
        if renderer is not None:renderer.close()
        if env is not None:env.close()
        if vision is not None:vision.close()


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ('collection-binding','checkpoint','dino','lila','output'):q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--vision-python');q.add_argument('--vision-device',default='cuda')
    q=sub.add_parser('run');q.add_argument('--binding',type=Path,required=True);q.add_argument('--index',type=int,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if a.command=='prepare':
        if a.output.exists():raise FileExistsError(a.output)
        value=prepare(a.collection_binding,a.checkpoint,a.dino,a.lila,a.vision_python,a.vision_device);a.output.parent.mkdir(parents=True,exist_ok=True);dump(a.output,value)
    else:print(json.dumps(run(a.binding,a.index,a.output)))


if __name__=='__main__':main()
