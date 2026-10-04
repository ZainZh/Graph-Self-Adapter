"""Independent causal replay, source integrity and physical-value checks."""
import numpy as np
import torch
from tools.fast_adapter.p34_common import *
from tools.fast_adapter.p34_runtime import RoleState
from tools.fast_adapter.p34_branches import physical_value

def main():
    from prove.fast_adapter.object_interaction_env import ObjectEnv,mujoco
    from prove.fast_adapter.held_hammer_env import HeldHammerEnv
    torch.set_num_threads(2);pr=torch.load(ROOT/'branches/PREDICTIONS.pt',weights_only=True)
    outcomes=torch.load(ROOT/'branches/OUTCOMES.pt',weights_only=True)['outcomes'];checks=[]
    # The reused hammer bank must still match the earlier sealed source.
    old=Path('evaluations/fast_adapter_p33_relation_dynamics_v01/data/RECEIPT.json')
    x=json.loads(old.read_text());bankpath=old.parent/'BANK.pt'
    expected=next(r['sha256'] for r in x['bindings'] if r['path']==str(bankpath))
    assert digest(bankpath)==expected
    selected=json.loads((ROOT/'validation/SELECTION_SEAL.json').read_text())['selected']
    for task in TASKS:
        b=load_data(task);assert not any(k in b for k in ('teacher','metrics','success','states'))
        fit=set(b['object_ids'][b['object_ids']<48].tolist());val=set(b['object_ids'][b['object_ids']>=48].tolist())
        assert not fit&val and len(fit)==48 and len(val)==16
        for seed in (11,22):
            for f in ('flat',selected[task]):
                row=next(r for r in pr['rows'] if r['task']==task and r['seed']==seed and r['family']==f and r['mode']=='continuous')
                n=model(task,seed,f);before=state_hash(n);ids=row['episode_indices'].cuda();times=row['times'].cuda()
                st=RoleState(n,len(ids),'continuous');zerr=0.;scoreerr=0.;choices_ok=True
                snaps=[s for s in pr['snapshots'] if s['task']==task]
                base=torch.stack([s['candidates'][0] for s in snaps]).cuda()
                for t in range(int(times.max())+1):
                    ob={k:b[k][ids,t] for k in ('tokens','boxes','proprio')}
                    a,choice,scores=st.act(ob,base)
                    matches=times==t
                    if bool(matches.any()):
                        m=matches.cpu();zerr=max(zerr,float((st.z[matches].cpu()-row['z'][m]).abs().max()))
                        scoreerr=max(scoreerr,float((scores[matches].cpu()-row['scores'][m]).abs().max()))
                        choices_ok &= torch.equal(choice[matches].cpu(),row['choice'][m])
                    st.observe(b['actions'][ids,t],{k:b[k][ids,t+1] for k in ('tokens','boxes','proprio')})
                assert zerr<2e-5 and scoreerr<2e-5 and choices_ok,(task,seed,f,zerr,scoreerr)
                assert state_hash(n)==before
                checks.append({'task':task,'seed':seed,'family':f,'max_z_error':zerr,
                               'max_score_error':scoreerr,'choices_match':choices_ok,'weights_frozen':True})
                print('CAUSAL_AUDIT',task,seed,f,zerr,scoreerr,flush=True)
    maxerr=0.;num=0
    for o in outcomes:
        cls=HeldHammerEnv if o['task']=='hammer-block' else ObjectEnv;e=cls(o['config'],render=False)
        try:
            e.reset(o['reset_seed'])
            for j,tr in enumerate(o['traces']):
                values=[]
                for t in (0,4):
                    for k,v in tr['states'].items():getattr(e.env.data,k)[:]=v[t].numpy()
                    mujoco.mj_forward(e.env.model,e.env.data);values.append(physical_value(e,o['task']))
                err=abs((values[1]-values[0])-o['progress'][j]);maxerr=max(maxerr,err);num+=1
        finally:e.close()
    assert maxerr<1e-10
    # Also verify sealed outcomes and all training/checkpoint inputs stayed unchanged.
    for path in list(ROOT.glob('*/models/seed*/SOURCE_FREEZE.json'))+[ROOT/'branches/PREDICTION_SEAL.json']:
        for x in json.loads(path.read_text())['bindings']:assert digest(x['path'])==x['sha256'],x['path']
    dump(ROOT/'AUDIT.json',{'passed':True,'causal_replay':checks,'physical_branches_regraded':num,
        'physical_max_error':maxerr,'old_hammer_bank_matches_original_seal':True,'train_validation_objects_disjoint':True,
        'no_privileged_fields_in_learning_banks':True,'no_new_full_rollouts':True,
        'bindings':bindings([ROOT/'branches/RESULTS.json',ROOT/'branches/OUTCOMES.pt',ROOT/'validation/PREDICTION_RESULTS.json',
                             Path(__file__),Path('tools/fast_adapter/p34_runtime.py')])})
    print('P34_AUDIT_PASSED',flush=True)

if __name__=='__main__':main()
