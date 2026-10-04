#!/usr/bin/env python3
"""Original LIBERO initial state -> frozen prefix/Code -> P53 neural short Insert.

prepare binds three engineering attempts only. No teacher action is called.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,hashlib,importlib.metadata,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from prove.task_graph.p52_stage_policy import Config,StagePolicy,StageRuntime
from tools.task_graph.collect_libero_p53_v03 import run_prefix,run_code,validate_graph,verify as verify_parent
from tools.task_graph.p52_full_task import initialize,InsertJudge,structural_measurement
from tools.task_graph.p52_policy_pilot import Vision,validate_features
from tools.task_graph.collect_libero_p52 import PublicRenderer
from tools.task_graph.collect_libero_demo_gt import snapshot,dump,sha
from tools.task_graph.p52_deployment import source_closure

CELLS=[dict(index=i,arm=a,zmode='observed_feedback' if a=='online' else 'fixed') for i,a in enumerate(('bc','fixed','online'))]
PARENT_KEYS=('environment','public','prefix','relation','reference_poses','reference_authority','near_reference_z_offset','entry_contract','allowed_phases','teacher_authority','micro_domain','code_position_tolerance','code_rotation_tolerance','prefix_max_actions')


def neural_graph(parent_graph):
    graph=copy.deepcopy(parent_graph);graph['graph_id']+='_frozen_neural_full_v01'
    graph['nodes'][2]['backend']='frozen_stage_policy';validate_graph(graph)
    return graph


def check_packages(bc,joint,data_hash,warmup_hash):
    for package,branch in ((bc,'bc'),(joint,'joint')):
        if package.get('schema')!='p53_short_insert_skill_v01' or package.get('branch')!=branch:raise ValueError('P53 final branch required')
        if package.get('common_warmup_sha256')!=warmup_hash or package.get('data_sha256')!=data_hash:raise ValueError('Shared warmup/data mismatch')
        if package.get('horizon')!=24 or package.get('total_outer_updates')!=3000:raise ValueError('P53 H24/3000 final checkpoints required')
        model=StagePolicy(Config(**package['config']));model.load_state_dict(package['model'],strict=True)
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite policy weights')


def prepare(parent_path,bc,joint,common_warmup,data,dino,lila,vision_python,vision_device):
    parent_path=Path(parent_path).resolve();parent=json.loads(parent_path.read_text());verify_parent(parent)
    paths={k:Path(p).resolve() for k,p in dict(bc=bc,joint=joint,common_warmup=common_warmup,data=data).items()}
    packages={k:torch.load(paths[k],map_location='cpu',weights_only=True) for k in ('bc','joint')}
    check_packages(packages['bc'],packages['joint'],sha(paths['data']),sha(paths['common_warmup']))
    common=torch.load(paths['common_warmup'],map_location='cpu',weights_only=True)
    if common['schema']!='p53_common_warmup_v01' or common['data_sha256']!=sha(paths['data']) or common['common_outer_updates']!=1000:raise ValueError('Common warmup provenance mismatch')
    command='import json,importlib.metadata as m;print(json.dumps({n:m.version(n) for n in ("torch","transformers","numpy")}))'
    visual_packages=json.loads(subprocess.check_output([vision_python or sys.executable,'-c',command],text=True))
    assets=[Path(dino)/p for p in ('model.safetensors','config.json','preprocessor_config.json')]+[Path(lila)/p for p in ('VISUAL_WEIGHTS.pt','models/vla_model_fm.py')]
    binding={k:copy.deepcopy(parent[k]) for k in PARENT_KEYS}
    binding.update(schema='prove.p53-full-original-task.v1',parent=dict(path=str(parent_path),sha256=sha(parent_path)),task_graph=neural_graph(parent['task_graph']),
        checkpoints={a:dict(path=str(paths['bc' if a=='bc' else 'joint']),sha256=sha(paths['bc' if a=='bc' else 'joint'])) for a in ('bc','fixed','online')},
        training_provenance={k:dict(path=str(paths[k]),sha256=sha(paths[k])) for k in ('common_warmup','data')},
        vision=dict(dino=str(Path(dino).resolve()),lila=str(Path(lila).resolve()),python=vision_python,device=vision_device,packages=visual_packages),
        visual_assets={str(p.resolve()):sha(p) for p in assets},sources={p:sha(ROOT/p) for p in source_closure(ROOT,['tools/task_graph/p53_full_task.py','tools/task_graph/p52_vision_worker.py'])},
        packages={n:importlib.metadata.version(n) for n in ('torch','numpy','scipy','mujoco')},
        cells=CELLS,seed=53,code_max_actions=120,micro_max_actions=24,total_max_actions=544,original_initial_restores=1,later_state_restores=0,
        grasp_offset=[0.,0.,0.],automatic_retries=0,independent_statistical_evidence=False,
        scope='Three original-initial-state complete-task engineering attempts, not independent success rates',
        completion='s1-s8 complete plus native official goal and at least one book finger contact jointly hold3 in s9',
        structural_stop='Wrong target immediate; neither finger touching book five consecutive observations; unavailable measurement stops',
        runtime_inputs='RGB/visible role masks/proprio; observed public transition feedback only for online z; no teacher labels or hidden geometry in neural action/loss')
    return binding


def verify(binding):
    if binding['schema']!='prove.p53-full-original-task.v1' or binding['cells']!=CELLS:raise ValueError('Full three-arm scope changed')
    for k,v in dict(prefix_max_actions=400,code_max_actions=120,micro_max_actions=24,total_max_actions=544,original_initial_restores=1,later_state_restores=0,automatic_retries=0).items():
        if binding[k]!=v:raise ValueError('Scope/budget changed:'+k)
    if binding['grasp_offset']!=[0.,0.,0.]:raise ValueError('Original nominal initial condition only')
    if sha(binding['parent']['path'])!=binding['parent']['sha256']:raise ValueError('Qualification parent changed')
    # Parent remains its four-qualification teacher scope; never verify the neural
    # child using that teacher-only schema/conditions contract.
    parent=json.loads(Path(binding['parent']['path']).read_text());verify_parent(parent)
    for k in PARENT_KEYS:
        if binding[k]!=parent[k]:raise ValueError('Qualified Code/environment changed:'+k)
    if binding['task_graph']!=neural_graph(parent['task_graph']):raise ValueError('Actual neural graph route changed')
    for p,h in binding['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen source changed:'+p)
    for p,h in binding['visual_assets'].items():
        if sha(p)!=h:raise ValueError('Frozen visual asset changed:'+p)
    for v in list(binding['checkpoints'].values())+list(binding['training_provenance'].values()):
        if sha(v['path'])!=v['sha256']:raise ValueError('Frozen checkpoint/provenance changed')
    if binding['checkpoints']['fixed']!=binding['checkpoints']['online']:raise ValueError('Fixed/online require identical model')
    for n,v in binding['packages'].items():
        if importlib.metadata.version(n)!=v:raise ValueError('Runtime version changed')


def weight_digest(model):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


def run_neural(env,binding,runtime,vision,renderer,output,names):
    node=binding['task_graph']['nodes'][2]
    if node['id']!='s9' or node['backend']!='frozen_stage_policy' or node['max_actions']!=binding['micro_max_actions']:raise ValueError('Unbound neural s9')
    judge=InsertJudge();obs=env._get_observations(force_update=True);previous=previous_action=None;steps=0
    while True:
        path=output/'public'/f'frame_{steps:04d}.npz';np.savez_compressed(path,**renderer.observe(obs))
        features=vision.encode(path);validate_features(features,runtime.model.config);torch.save(features,output/'public'/f'features_{steps:04d}.pt')
        if previous is None:runtime.enter(features)
        before_z=runtime.z.clone();loss=None
        if previous is not None:loss=runtime.feedback(previous,previous_action,features)
        action=runtime.command(features).numpy()
        if action.shape!=(7,) or not np.isfinite(action).all() or action[-1]!=1:raise ValueError('Invalid normalized OSC action')
        official,fingers,wrong=structural_measurement(env,names);judgment=judge.observe(official,fingers,wrong)
        stop=judgment['stop'] or ('micro_budget24' if steps>=node['max_actions'] else None)
        row=dict(t=steps,stage_id=node['id'],backend=node['backend'],observation=path.name,sha256=sha(path),z_before=before_z.tolist(),z_after=runtime.z.tolist(),inner_loss=loss,updates=runtime.updates,action=action.tolist(),executed=stop is None)
        with (output/'public/ACTIONS.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        private,truth=snapshot(env,obs);np.savez_compressed(output/'private'/f'frame_{steps:04d}.npz',**{k:v for k,v in private.items() if k.startswith('supervisor_')})
        with (output/'private/JUDGMENTS.jsonl').open('a') as f:f.write(json.dumps(dict(t=steps,official_success=official,finger_contacts=fingers,wrong_target=wrong,contacts=truth['contacts'],**judgment),allow_nan=False)+'\n')
        if stop:return dict(actions=steps,graph_complete=judgment['complete'],official_success=official,final_judgment=judgment,stop_reason=stop,updates=runtime.updates)
        previous=features;previous_action=torch.tensor(action[:6],dtype=torch.float32);obs,_,_,_=env.step(action.copy());steps+=1


def run(binding_path,index,output):
    binding=json.loads(binding_path.read_text());verify(binding)
    if index not in range(3):raise ValueError('Outside frozen three cells')
    cell=binding['cells'][index];output.mkdir(parents=True,exist_ok=False)
    for d in ('prefix','code','public','private'):(output/d).mkdir()
    dump(output/'BINDING.json',binding);dump(output/'CELL.json',cell);dump(output/'TASK_GRAPH.json',binding['task_graph'])
    torch.set_num_threads(1);torch.manual_seed(binding['seed']);np.random.seed(binding['seed'])
    ck=torch.load(binding['checkpoints'][cell['arm']]['path'],map_location='cpu',weights_only=True);model=StagePolicy(Config(**ck['config']));model.load_state_dict(ck['model'],strict=True)
    runtime=StageRuntime(model,online=cell['zmode']=='observed_feedback');before=weight_digest(model);env=vision=renderer=None
    try:
        env=initialize(binding,output)
        names={kind:[getattr(env.sim.model,kind+'_id2name')(i) for i in range(getattr(env.sim.model,count))] for kind,count in [('body','nbody'),('site','nsite'),('geom','ngeom')]};dump(output/'private/NAMES.json',names)
        prefix=run_prefix(env,binding,output/'prefix',output.name,dict(grasp_world_offset=binding['grasp_offset']));dump(output/'PREFIX_RESULT.json',prefix)
        code=run_code(env,binding,output/'code',names) if prefix['qualified'] else dict(qualified=False,actions=0,near_entry_qualified=False,stop_reason='prefix_failed')
        dump(output/'CODE_RESULT.json',code)
        transitions=[]
        if prefix['qualified']:transitions.append(dict(from_stage='s6',to_stage='s7',after_actions=prefix['actions']))
        if (output/'code/TRACE.jsonl').exists():
            for line in (output/'code/TRACE.jsonl').read_text().splitlines():
                row=json.loads(line)
                for transition in row['transitions']:transitions.append(dict(**transition,after_actions=prefix['actions']+row['t']))
        if code['qualified']:
            if code['next_stage']!='s9':raise ValueError('Code handoff differs from neural graph')
            transitions.append(dict(from_stage='s8',to_stage='s9',after_actions=prefix['actions']+code['actions']))
            vision=Vision(binding['vision']);dump(output/'VISUAL_BINDING.json',vision.binding);renderer=PublicRenderer(env,binding['public'])
            result=run_neural(env,binding,runtime,vision,renderer,output,names)
        else:result=dict(actions=0,graph_complete=False,official_success=bool(env._check_success()),stop_reason=code['stop_reason'],updates=0)
        if result['graph_complete']:transitions.append(dict(from_stage='s9',to_stage=None,after_actions=prefix['actions']+code['actions']+result['actions']))
        after=weight_digest(model)
        if before!=after:raise RuntimeError('Frozen network weights mutated')
        verify(binding);dump(output/'STAGE_TRANSITIONS.json',transitions)
        files={str(p.relative_to(output)):sha(p) for d in ('prefix','code','public','private') for p in sorted((output/d).iterdir()) if p.is_file()}
        files.update({n:sha(output/n) for n in ('BINDING.json','CELL.json','TASK_GRAPH.json','PREFIX_RESULT.json','CODE_RESULT.json','STAGE_TRANSITIONS.json')})
        if (output/'VISUAL_BINDING.json').exists():files['VISUAL_BINDING.json']=sha(output/'VISUAL_BINDING.json')
        dump(output/'SEAL.json',dict(files=files,binding_sha256=sha(output/'BINDING.json'),cell=cell,original_initial_restores=1,later_state_restores=0,network_before_sha256=before,network_after_sha256=after))
        result.update(cell=cell,prefix_actions=prefix['actions'],code_actions=code['actions'],near_entry_qualified=code['qualified'],total_actions=prefix['actions']+code['actions']+result['actions'],network_weights_unchanged=before==after,seal_sha256=sha(output/'SEAL.json'),automatic_retry=False,independent_statistical_evidence=False)
        dump(output/'RESULT.json',result);return result
    except Exception as exc:dump(output/'ABORT.json',dict(cell=cell,error=type(exc).__name__,reason=str(exc),automatic_retry=False));raise
    finally:
        runtime.exit()
        if renderer is not None:renderer.close()
        if env is not None:env.close()
        if vision is not None:vision.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True);q=sub.add_parser('prepare')
    for name in ('parent','bc','joint','common-warmup','data','dino','lila','output'):q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--vision-python');q.add_argument('--vision-device',default='cuda')
    q=sub.add_parser('run');q.add_argument('--binding',type=Path,required=True);q.add_argument('--index',type=int,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare':
        if a.output.exists():raise FileExistsError(a.output)
        b=prepare(a.parent,a.bc,a.joint,a.common_warmup,a.data,a.dino,a.lila,a.vision_python,a.vision_device);verify(b);dump(a.output,b)
    else:print(json.dumps(run(a.binding,a.index,a.output),indent=2))
