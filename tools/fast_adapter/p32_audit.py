"""Replay recorded decisions and original physical success after full P32 seal."""
import torch
from tools.fast_adapter.p32_common import *
from tools.fast_adapter.p32_eval import transition
from prove.fast_adapter.learned_interaction_loss import LossState
from prove.fast_adapter.object_interaction_env import ObjectEnv,mujoco
from prove.fast_adapter.update_direction_diagnostic import causal_pairs,action_direction
from tools.fast_adapter.p31_update_direction import state_labels


def main():
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    out=ROOT/'development';result=json.loads((out/'RESULT.json').read_text())
    assert len(result['rows'])==144
    for p in [out/'SOURCE_FREEZE.json']+[ROOT/'models'/f'seed{s}'/'SOURCE_FREEZE.json' for s in (11,22)]:
        for b in json.loads(p.read_text())['bindings']:assert digest(b['path'])==b['sha256'],b['path']
    base=base_model(TASK);obs=observer();nets={(s,f):adapter(s,f) for s in (11,22) for f in ('predictive','learned')}
    groups={}
    for r in result['rows']:groups.setdefault(r['source']['path'],[]).append(r)
    outdiag=ROOT/'diagnostics';outdiag.mkdir(exist_ok=False)
    maximum_action=maximum_z=maximum_zero=0.;physical=0;directions=[];exposure=[]
    for path,rows in groups.items():
        assert digest(path)==rows[0]['source']['sha256']
        b=torch.load(path,weights_only=True);arm=rows[0]['arm'];seed=rows[0]['seed'];split=rows[0]['split']
        net=None if arm=='base' else nets[seed,'predictive' if arm=='predictive' else 'learned']
        learner=None if net is None else LossState(net,len(rows),mode='first3' if arm.endswith('first3') else 'gradient')
        donor=None
        if arm=='learned_donor':
            ref=json.loads(Path(path).with_suffix('.json').read_text())['donor_source'];assert digest(ref['path'])==ref['sha256']
            original=torch.load(ref['path'],weights_only=True);ix=torch.tensor(ref['recipient_to_donor'])
            assert ix.tolist()==[(i+1)%8 for i in range(8)]
            donor={k:original[k][ix].cuda() for k in ('visual','proprio','actions','active')}
        previous=torch.zeros(len(rows),4,device='cuda')
        for t in range(200):
            v,p,nv,np_=[b[k][:,j].cuda() for k,j in [('visual',t),('proprio',t),('visual',t+1),('proprio',t+1)]]
            actual=b['actions'][:,t].cuda()
            with torch.no_grad():ba=base(v,p,previous);active=obs.active(v,p,previous)
            assert torch.equal(active.cpu(),b['active'][:,t])
            a=ba if learner is None else learner.action(v,p,previous,ba,active)
            maximum_action=max(maximum_action,float((a-actual).abs().max()))
            if learner is not None:
                maximum_z=max(maximum_z,float((learner.z-b['latent_before_action'][:,t].cuda()).abs().max()))
                assert torch.equal(learner.counts.cpu(),b['counts_before_action'][:,t])
                assert torch.equal(a[~active],ba[~active])
                with torch.no_grad():zero=net.action(v,p,previous,ba,net.z0[None].expand(len(rows),-1))
                maximum_zero=max(maximum_zero,float((zero-ba).abs().max()))
                if donor is None:learner.observe(v,p,previous,actual,nv,np_,active)
                else:learner.observe(*transition(donor,t),active,history_active=donor['active'][:,t])
            previous=actual
        if learner is not None:
            assert torch.equal(learner.counts.cpu(),b['updates'])
            maximum_z=max(maximum_z,float((learner.z.cpu()-b['final_z']).abs().max()))
            if arm.endswith('first3'):assert bool((learner.counts<=3).all())
            d=b['update_diagnostics'];u=d['updated']
            exposure.append({'seed':seed,'split':split,'arm':arm,'attempts':int(u.sum()),
                    'empty_window_attempts':int((u&(d['valid_transitions']==0)).sum()),
                    'zero_delta_attempts':int((u&(d['delta_norm']<1e-10)).sum()),
                    'mean_valid_transitions_per_attempt':float(d['valid_transitions'][u].mean()) if bool(u.any()) else None,
                    'final_z_rms':float(b['final_z'].square().mean().sqrt())})
        for row in rows:
            j=row['source_row'];env=ObjectEnv(row['config'],render=False);env.reset(row['reset_seed']);success=[];contacts=[]
            try:
                for t in range(1,201):
                    for key,x in b['states'].items():getattr(env.env.data,key)[:]=x[j,t].numpy()
                    mujoco.mj_forward(env.env.model,env.env.data)
                    _,m=env.env.evaluate_state(env.env._get_obs(),b['actions'][j,t-1].numpy())
                    success.append(bool(m['success']));contacts.append(bool(m['bilateral_handle']));physical+=1
                assert any(success)==row['success'] and success[-1]==row['terminal_success']
                assert any(contacts)==row['ever_bilateral'] and sum(contacts)==row['held_steps']
            finally:env.close()
        # Same-state local action proxy; cannot substitute for executed success.
        if arm in ('predictive','learned'):
            labels=state_labels(b,rows);p={k:x.cuda() for k,x in causal_pairs(b).items()}
            with torch.no_grad():ba=base(p['visual'],p['proprio'],p['previous'])
            target=labels['teacher'][:,1:].cuda();n,t=p['update'].shape
            flat=lambda x:x.reshape(n*t,*x.shape[2:]);q={k:flat(v) for k,v in p.items()};allvalues={}
            for start in range(0,n*t,256):
                sl=slice(start,start+256);x={k:v[sl] for k,v in q.items()}
                values=action_direction(net,x['visual'],x['proprio'],x['previous'],flat(ba)[sl],x['before'],x['after'],flat(target)[sl],x['active'],net.z0[None].expand(len(x['active']),-1))
                for k,v in values.items():
                    if k in ('action_benefit','translation_benefit','gripper_benefit','task_descent_cosine','direction_defined'):
                        allvalues.setdefault(k,[]).append(v.cpu())
            values={k:torch.cat(v).reshape(n,t) for k,v in allvalues.items()}
            mask=p['update'].cpu()&p['active'].cpu()&labels['near'][:,:-1]&labels['near'][:,1:]
            defined=mask&values['direction_defined'];benefit=values['action_benefit'][mask]
            means=[float(values['action_benefit'][i,mask[i]].mean()) for i in range(n) if bool(mask[i].any())]
            directions.append({'seed':seed,'split':split,'arm':arm,'true_stage_pairs':int(mask.sum()),
                    'episodes_with_pairs':len(means),'help_fraction':float((benefit>1e-6).float().mean()) if len(benefit) else None,
                    'harm_fraction':float((benefit<-1e-6).float().mean()) if len(benefit) else None,
                    'gradient_agreement_fraction':float((values['task_descent_cosine'][defined]>0).float().mean()) if bool(defined.any()) else None,
                    'episode_balanced_mean_benefit':sum(means)/len(means) if means else None})
            torch.save({'values':values,'true_stage_mask':mask,'source':path},outdiag/(Path(path).stem+'.pt'))
        print('AUDIT_BLOCK',Path(path).name,flush=True)
    assert maximum_action<1e-5 and maximum_z<1e-5 and maximum_zero==0 and physical==28800
    dump(ROOT/'AUDIT.json',{'passed':True,'physical_states_replayed':physical,'max_action_error':maximum_action,
         'max_latent_error':maximum_z,'max_z0_base_error':maximum_zero,'all_source_bindings_unchanged':True,
         'donor_full_history_replay_verified':True,'full_visual_reencoding':False,
         'proxy_limit':'Correlated same-state teacher-action proxies, not new executed counterfactual success',
         'directions':directions,'update_exposure':exposure,'bindings':bindings([Path(__file__),out/'RESULT.json'])})
    print('AUDIT_COMPLETE',physical,maximum_action,maximum_z,flush=True)


if __name__=='__main__':main()
