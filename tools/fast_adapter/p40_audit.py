"""Numerical action/latent audit; first condition in each batch gets physics replay.

Does not generate new observations for learning or new task trials. All outputs
derive from sealed runtime actions. Audit scope is explicitly not all-physics.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.contact_skills import ContactSkill,STEPS
from prove.fast_adapter.skill_tokens_v02 import SkillAdapter,SkillControl,action_force
from tools.fast_adapter.p40_evaluate import ROOT,load_model,load_head,js,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--phase',required=True);a=p.parse_args()
    root=ROOT/a.task/a.phase;files=sorted(root.rglob('*_RUNTIME.pt'));cache={};references={};checks=[];head=load_head(a.task)
    means=torch.load(ROOT/a.task/'MEANS_v03.pt',weights_only=True) if (ROOT/a.task/'MEANS_v03.pt').exists() else {}
    for raw_path in files:
        arm=raw_path.name.removesuffix('_RUNTIME.pt');kind=raw_path.parents[2].name;seed=int(raw_path.parents[1].name.removeprefix('seed'));split=raw_path.parent.name
        result=raw_path.with_name(arm+'_RESULT.json')
        if not result.exists():raise RuntimeError('Cannot audit incomplete batch')
        seal=json.loads(raw_path.with_name(arm+'_SEAL.json').read_text());assert sha(raw_path)==seal['raw']
        raw=torch.load(raw_path,weights_only=True);count=raw['actions'].shape[0]
        key=(kind,seed)
        if key not in cache:cache[key]=load_model(a.task,kind,seed) if kind!='references' else SkillAdapter(a.task,'relation').cuda().eval().requires_grad_(False)
        n=cache[key];z=n.initial(count)
        if arm=='mean':z=means[f'{kind}_{seed}'].cuda()[None].repeat(count,1)
        feedback={k:v.cuda() for k,v in raw['feedback'].items()}
        if arm=='donor':
            donor=torch.load(root/'references/seed0'/split/'base_RUNTIME.pt',weights_only=True)['feedback'];order=(torch.arange(count)^1) if split=='paired' else torch.arange(count).roll(-1)
            feedback={k:v[order].cuda() for k,v in donor.items()}
            with torch.no_grad():feedback['jacobian']=head(feedback['local']) if kind in ('local','fused','history') else torch.zeros(count,STEPS,16,4,device='cuda')
        cs=[SkillControl(a.task) for _ in range(count)];maxa=0.;maxz=0.;maxp=0.
        for t in range(STEPS+1):
            if t and kind!='references' and (arm in ('continuous','donor') or arm=='first12' and t<=12):
                w={k:v[:,max(0,t-4):t] for k,v in feedback.items()};newz,_=n.update(z,w);z=torch.where(raw['aborted'][:,t].cuda()[:,None],z,newz)
            e=n.estimate(z);maxz=max(maxz,float((z.detach().cpu()-raw['latent'][:,t]).abs().max()));maxp=max(maxp,float((e.detach().cpu()-raw['parameters'][:,t]).abs().max()))
            if t==STEPS:break
            obs=[dict(positions=raw['positions'][i,t].numpy(),proprio=raw['proprio'][i,t].numpy(),visible=raw['visible'][i,t].numpy()) for i in range(count)]
            wanted=torch.tensor([min(.6 if a.task=='hammer' else 1.,max(.08,c.remaining(o)*1000*.20)) for c,o in zip(cs,obs)],device='cuda')
            with torch.no_grad():force=action_force(a.task,e,wanted).cpu().numpy()
            if arm.startswith('fixed'):
                from tools.fast_adapter.p40_evaluate import FORCES
                force[:]=FORCES[a.task][int(arm[-1])]
            if t<12:force[:]=dict(cup=1.5,hammer=.2,insert=.4)[a.task]
            actions=np.array([c.action(o,t,force[i]) if not raw['aborted'][i,t] else np.zeros(3 if a.task=='cup' else 2) for i,(c,o) in enumerate(zip(cs,obs))])
            maxa=max(maxa,float(np.max(np.abs(actions-raw['actions'][:,t].numpy()))))
        assert maxa<1e-5 and maxz<1e-4 and maxp<1e-4,(raw_path,maxa,maxz,maxp)
        # Same fixed common probe on every condition, not merely the same count.
        if split not in references:
            refs=torch.load(root/'references/seed0'/split/'base_RUNTIME.pt',weights_only=True);references[split]=refs
        refs=references[split]
        assert torch.equal(raw['actions'][:,:12],refs['actions'][:,:12]),str(raw_path)
        assert torch.equal(raw['qpos'][:,:13],refs['qpos'][:,:13]),str(raw_path)
        paired=0
        if split=='paired':
            for i in range(0,count,2):
                assert raw['rgb_hashes'][0][i]==raw['rgb_hashes'][0][i+1]
                assert raw['mask_hashes'][0][i]==raw['mask_hashes'][0][i+1];paired+=1
        env=ContactSkill(seal['configs'][0],render=False);phy=0.
        for t in range(STEPS+1):
            state=env.state();phy=max(phy,float(np.max(np.abs(state['qpos']-raw['qpos'][0,t].numpy()))),float(np.max(np.abs(state['qvel']-raw['qvel'][0,t].numpy()))))
            if t<STEPS and not raw['aborted'][0,t]:env.step(raw['actions'][0,t].numpy())
        env.close();assert phy<1e-7,(raw_path,phy)
        checks.append(dict(path=str(raw_path),episodes=count,action_error=maxa,latent_error=maxz,parameter_error=maxp,physics_sample_index=0,physics_error=phy,paired_initial_checks=paired))
    out=dict(task=a.task,phase=a.phase,batches=len(checks),episodes=sum(c['episodes'] for c in checks),physics_replayed_episodes=len(checks),all_action_latent_reconstructed=True,common12_exact=True,checks=checks)
    js(ROOT/a.task/(a.phase+'_AUDIT.json'),out);print({k:v for k,v in out.items() if k!='checks'})

if __name__=='__main__':torch.set_num_threads(2);torch.backends.cudnn.enabled=False;main()
