"""Posthoc direction checks on P31 records. No training and no env.step calls."""
import json
from pathlib import Path
import numpy as np
import torch
from tools.fast_adapter.p31v2_common import ROOT as SOURCE, TASK, base_model, base_path, adapter, observer, bindings, dump, digest, state_hash
from tools.fast_adapter.p31v2_data import relation_labels
from prove.fast_adapter.object_interaction_env import ObjectEnv, mujoco
from prove.fast_adapter.continuous_adapter import feedback
from prove.fast_adapter.interaction_stage_adapter import StageState
from prove.fast_adapter.update_direction_diagnostic import causal_pairs, action_direction

OUT=Path('evaluations/fast_adapter_p31_update_direction_v01')
EPS=1e-6


def state_labels(record, rows):
    n,t=record['actions'].shape[:2]
    labels={'teacher':torch.zeros(n,t,4),'near':torch.zeros(n,t,dtype=torch.bool),
            'held':torch.zeros(n,t,dtype=torch.bool),'lift':torch.zeros(n,t)}
    for j,row in enumerate(rows):
        env=ObjectEnv(row['config'],render=False);env.reset(row['reset_seed'])
        try:
            for k in range(t):
                for name,states in record['states'].items():getattr(env.env.data,name)[:]=states[j,k].numpy()
                mujoco.mj_forward(env.env.model,env.env.data)
                action,near=relation_labels(env);left,right,*_=env.env.contacts()
                labels['teacher'][j,k]=torch.from_numpy(action);labels['near'][j,k]=bool(near)
                labels['held'][j,k]=bool(left and right);labels['lift'][j,k]=float(env.env.data.body('actor').xpos[2]-env.env.start_z)
        finally:env.close()
    return labels


def score(record, labels, net, base, verify_execution):
    pairs={k:v.cuda() for k,v in causal_pairs(record).items()};n,t=pairs['update'].shape
    teacher=labels['teacher'][:,1:].cuda()
    with torch.no_grad():ba=base(pairs['visual'],pairs['proprio'],pairs['previous'])
    # z3 is used only AFTER that third update was actually observed.
    anchor=net.z0[None].expand(n,-1).clone()
    for j in range(n):
        found=(record['counts_before_action'][j]>=3).nonzero().flatten()
        if len(found):anchor[j]=record['latent_before_action'][j,int(found[0])].cuda()
    flat=lambda x:x.reshape(n*t,*x.shape[2:])
    flat_pairs={k:flat(v) for k,v in pairs.items()};a0=flat(ba);target=flat(teacher)
    anchors=anchor[:,None].expand(-1,t,-1).reshape(n*t,-1)
    values={}
    for start in range(0,n*t,256):
        sl=slice(start,start+256);p={k:v[sl] for k,v in flat_pairs.items()}
        result=action_direction(net,p['visual'],p['proprio'],p['previous'],a0[sl],p['before'],p['after'],target[sl],p['active'],anchors[sl])
        for k,v in result.items():values.setdefault(k,[]).append(v.cpu())
    values={k:torch.cat(v).reshape(n,t,*v[0].shape[1:]) for k,v in values.items()}
    error=float((values.pop('actual_after')-record['actions'][:,1:]).abs().max()) if verify_execution else None
    if verify_execution:assert error<1e-5,error
    values['current_minus_first3_loss'][~(pairs['update_number'].cpu()>=3)]=float('nan')
    values.update({k:pairs[k].cpu() for k in ['active','source_active','update','update_number']})
    values['source_true_stage']=labels['near'][:,:-1]
    values['next_true_stage']=labels['near'][:,1:]
    if 'held' in labels:
        values['next_held']=labels['held'][:,1:]
        ever=labels['held'].int().cumsum(1)>0
        values['next_pregrasp_near']=labels['near'][:,1:] & ~ever[:,1:]
        values['next_after_contact_loss']=ever[:,1:] & ~labels['held'][:,1:]
    # Recompute the actual masked8-transition adaptation objective at each z.
    v,p,actions=[record[k].cuda() for k in ['visual','proprio','actions']]
    previous=torch.cat((torch.zeros_like(actions[:,:1]),actions[:,:-1]),1)
    with torch.no_grad():
        u=net.encode(v[:,:-1],p[:,:-1],previous);y,w=feedback(v[:,:-1],p[:,:-1],v[:,1:],p[:,1:])
        w=w*record['active'].cuda()[...,None]
        loss_arrays={key:[] for key in ['adapt_loss_before','adapt_loss_after']}
        for start in range(0,n*t,256):
            indices=torch.arange(start,min(start+256,n*t),device='cuda');batch=indices//t;time=indices%t
            history=time[:,None]-torch.arange(7,-1,-1,device='cuda')[None,:];valid=history>=0;history=history.clamp_min(0)
            hw=w[batch[:,None],history]*valid[...,None]
            for name,zkey in [('adapt_loss_before','before'),('adapt_loss_after','after')]:
                z=flat_pairs[zkey][indices]
                pred=net.prediction(u[batch[:,None],history],actions[batch[:,None],history],z[:,None].expand(-1,8,-1))
                loss=64*((pred-y[batch[:,None],history]).square()*hw).sum((1,2))/hw.sum((1,2)).clamp_min(1)
                loss_arrays[name].append(loss.cpu())
        for k,xs in loss_arrays.items():values[k]=torch.cat(xs).reshape(n,t)
    values['adapt_benefit']=values['adapt_loss_before']-values['adapt_loss_after']
    return values,error


def shadow_train(record,net,obs):
    learner=StageState(net,len(record['actions']),stage=True)
    z=[];counts=[];gate=[]
    for t in range(200):
        v,p=record['visual'][:,t].cuda(),record['proprio'][:,t].cuda()
        previous=record['actions'][:,t-1].cuda() if t else torch.zeros(len(v),4,device='cuda')
        active=obs.active(v,p,previous);gate.append(active.cpu())
        z.append(learner.z.cpu().clone());counts.append(learner.counts.cpu().clone())
        learner.observe(v,p,previous,record['actions'][:,t].cuda(),record['visual'][:,t+1].cuda(),record['proprio'][:,t+1].cuda(),active)
    return {**record,'latent_before_action':torch.stack(z,1),'counts_before_action':torch.stack(counts,1),'active':torch.stack(gate,1)}


def masks(v):
    u=v['update'];executed=u&v['active'];true=executed&v['source_true_stage']&v['next_true_stage']
    result={'all_updates':u,'affects_next_action':executed,'true_stage':true,
         'false_stage_trigger':executed&~v['source_true_stage'],
         'true_stage_first3':true&(v['update_number']<=3),'true_stage_later':true&(v['update_number']>3)}
    if 'next_held' in v:
        result['pregrasp_true_stage']=true&v['next_pregrasp_near']
        result['held_true_stage']=true&v['next_held']
        result['after_contact_loss']=executed&v['next_after_contact_loss']
    return result


def summarize(v,mask):
    count=int(mask.sum())
    if not count:return {'steps':0,'episodes_with_steps':0}
    benefit=v['action_benefit'][mask];adapt=v['adapt_benefit'][mask]
    defined=mask&v['direction_defined'];ncos=int(defined.sum())
    macro={}
    for name in ['action_benefit','adapt_benefit','current_minus_base_loss','current_minus_first3_loss','translation_benefit','gripper_benefit']:
        mm=[]
        for j in range(len(mask)):
            m=mask[j]&torch.isfinite(v[name][j])
            if bool(m.any()):mm.append(float(v[name][j,m].mean()))
        macro[name]=float(np.mean(mm)) if mm else None
    fractions=[]
    for j in range(len(mask)):
        m=mask[j]
        if bool(m.any()):fractions.append([float((v['action_benefit'][j,m]>EPS).float().mean()),float((v['action_benefit'][j,m]<-EPS).float().mean())])
    return {'steps':count,'episodes_with_steps':len(fractions),
        'step_action_help_fraction':float((benefit>EPS).float().mean()),'step_action_harm_fraction':float((benefit<-EPS).float().mean()),
        'episode_mean_action_help_fraction':float(np.mean([x[0] for x in fractions])),
        'episode_mean_action_harm_fraction':float(np.mean([x[1] for x in fractions])),
        'adapt_loss_decreases_fraction':float((adapt>EPS).float().mean()),
        'adapt_down_but_action_worse_steps':int(((adapt>EPS)&(benefit<-EPS)).sum()),
        'defined_direction_steps':ncos,
        'gradient_agreement_fraction':float((v['task_descent_cosine'][defined]>0).float().mean()) if ncos else None,
        'mean_task_descent_cosine':float(v['task_descent_cosine'][defined].mean()) if ncos else None,
        'close_flips_against_teacher':int((v['close_flip_harm']&mask).sum()),
        'open_flips_against_teacher':int((v['open_flip_harm']&mask).sum()),
        'episode_balanced_means':macro}


def main():
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    raw=OUT/'raw';raw.mkdir(exist_ok=False)
    allrows=[]
    for seed in (11,22):allrows+=json.loads((SOURCE/f'development/seed{seed}/RESULT.json').read_text())['rows']
    sources=[SOURCE/f'development/seed{s}/RESULT.json' for s in (11,22)]
    sources += [SOURCE/f'models/seed{s}/own_stage.pt' for s in (11,22)]
    sources += [SOURCE/'data/own.pt',SOURCE/'observer.pt',base_path(TASK),Path(__file__),
            Path('prove/fast_adapter/update_direction_diagnostic.py'),Path('prove/fast_adapter/continuous_adapter.py'),
            Path('prove/fast_adapter/interaction_stage_adapter.py'),Path('tools/fast_adapter/p31v2_data.py'),OUT/'PROTOCOL.json']
    sources += sorted({Path(r['source']['path']) for r in allrows if r['arm']=='own_stage'})
    dump(OUT/'SOURCE_FREEZE.json',{'before_diagnostic':True,'new_rollouts':0,'bindings':bindings(sources)})
    lookup={(r['seed'],r['split'],r['condition'],r['arm']):r for r in allrows}
    base=base_model(TASK);obs=observer();base_hash=state_hash(base);obs_hash=state_hash(obs)
    training=torch.load(SOURCE/'data/own.pt',weights_only=True)
    blocks=[];episode_rows=[];maxerror=0.;failures=[];init_hashes={}
    for seed in (11,22):
        net=adapter(seed,'own_stage');init_hashes[seed]=state_hash(net)
        for split in ('train','geometry_ood'):
            for off in (0,8):
                rows=sorted([r for r in allrows if r['seed']==seed and r['split']==split and r['arm']=='own_stage' and off<=r['condition']<off+8],key=lambda x:x['condition'])
                path=Path(rows[0]['source']['path']);assert digest(path)==rows[0]['source']['sha256']
                b=torch.load(path,weights_only=True);labels=state_labels(b,rows)
                values,error=score(b,labels,net,base,True);maxerror=max(maxerror,error)
                dest=raw/f'dev_{split}_s{seed}_{off:03d}.pt';torch.save({'metrics':values,'labels':labels,'source':str(path)},dest)
                blocks.append({'domain':'development/'+split,'seed':seed,'path':str(dest)})
                for j,row in enumerate(rows):
                    base_row=lookup[seed,split,row['condition'],'base'];outcome='harm' if base_row['success'] and not row['success'] else 'both_success' if row['success'] else 'both_failure'
                    sub={k:x[j:j+1] for k,x in values.items()}
                    episode_rows.append({'domain':'development/'+split,'seed':seed,'condition':row['condition'],'outcome':outcome,
                           'updated_steps_with_next_action':int(sub['update'].sum()),
                           'false_stage_updates':int((sub['update']&~sub['source_true_stage']).sum()),
                           'metrics':{k:summarize(sub,m) for k,m in masks(sub).items()}})
                    if outcome=='harm':
                        metrics=json.loads(path.with_suffix('.json').read_text())['metrics'][j]
                        failures.append({'seed':seed,'split':split,'condition':row['condition'],'failure':'never_bilateral' if not row['ever_bilateral'] else 'contact_without_stable_lift',
                              'first_contact':next((t+1 for t,x in enumerate(metrics) if x['bilateral_handle']),None),
                              'max_lift':row['max_lift'],'source':str(dest),'row':j})
                print('DIRECTION_DEV',seed,split,off,'action replay max',error,flush=True)
        selected=torch.arange(0,128,4)
        for start in range(0,len(selected),8):
            ids=selected[start:start+8];b={k:v[ids] for k,v in training.items()};labels={'teacher':b['teacher'],'near':b['near']}
            record=shadow_train(b,net,obs);values,_=score(record,labels,net,base,False)
            dest=raw/f'training_s{seed}_{start:03d}.pt';torch.save({'metrics':values,'episode_ids':ids,'latent_before_action':record['latent_before_action']},dest)
            blocks.append({'domain':'training_replay','seed':seed,'path':str(dest)})
            print('DIRECTION_TRAIN',seed,start,flush=True)
        assert init_hashes[seed]==state_hash(net)
    assert base_hash==state_hash(base) and obs_hash==state_hash(obs)
    summaries=[]
    for domain in ('development/train','development/geometry_ood','training_replay'):
        for seeds in ([11],[22],[11,22]):
            data=[torch.load(x['path'],weights_only=True)['metrics'] for x in blocks if x['domain']==domain and x['seed'] in seeds]
            v={k:torch.cat([d[k] for d in data]) for k in data[0]}
            for name,mask in masks(v).items():summaries.append({'domain':domain,'seeds':seeds,'subset':name,**summarize(v,mask)})
    dump(OUT/'SUMMARY.json',{'scope':'offline posthoc; not new success-rate evidence','new_rollouts':0,'development_episodes':64,'training_shadow_trajectories':64,
         'candidate_updates_with_next_action':64*199,'max_next_action_replay_error':maxerror,'network_weights_unchanged':True,
         'summary':summaries,'failure_decomposition':failures,'source_blocks':blocks})
    dump(OUT/'PER_EPISODE.json',episode_rows)
    print('DIRECTION_COMPLETE',len(failures),'harms',flush=True)


if __name__=='__main__':main()
