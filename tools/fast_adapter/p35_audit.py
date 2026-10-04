"""Verify common targets, frozen inputs, causal choices, and physical branch grading."""
import numpy as np
import torch
from tools.fast_adapter.p35_common import *
from tools.fast_adapter.p35_runtime import MultiViewState
from tools.fast_adapter.p34_branches import raw_source,physical_value

def main():
    from prove.fast_adapter.object_interaction_env import ObjectEnv,mujoco
    from prove.fast_adapter.held_hammer_env import HeldHammerEnv
    from tools.fast_adapter.p35_camera import CameraRig
    torch.set_num_threads(2);pred=torch.load(ROOT/'branches/PREDICTIONS.pt',weights_only=True)
    checks=[];data_checks=[];mount=json.loads((ROOT/'preflight/WRIST_MOUNT.json').read_text());camera_error=0.
    for task in TASKS:
        receipt=json.loads((ROOT/task/'data/RECEIPT.json').read_text())
        for group in ('source_bindings','output_bindings','code_bindings'):
            for x in receipt[group]:assert digest(x['path'])==x['sha256'],x['path']
        # Camera attachment modifies rendering attributes, never physical state/object parameters.
        raw,cfg,j,_=raw_source(task,'expert',0);e=(ObjectEnv if task==TASKS[0] else HeldHammerEnv)(cfg);e.reset(raw['reset_seeds'][j])
        before=e.state();model_arrays={k:getattr(e.env.model,k).copy() for k in ('body_pos','geom_pos','body_mass')}
        rig=CameraRig(e,mount)
        try:
            for size,view in ((224,'global'),(448,'global'),(224,'wrist'),(448,'wrist')):
                im,_=rig.render(size,view);assert im.shape==(size,size,3)
            assert all(np.array_equal(v,e.state()[k]) for k,v in before.items())
            assert all(np.array_equal(v,getattr(e.env.model,k)) for k,v in model_arrays.items())
            cam=int(e.env.model.camera('gripperPOV').id);bid=int(e.env.model.cam_bodyid[cam])
            for t in (20,40,60):
                for k,v in raw['states'].items():getattr(e.env.data,k)[:]=v[j,t].numpy()
                mujoco.mj_forward(e.env.model,e.env.data)
                expected=e.env.data.body(bid).xpos+e.env.data.body(bid).xmat.reshape(3,3)@np.array(mount['pos'])
                camera_error=max(camera_error,float(np.abs(expected-e.env.data.cam_xpos[cam]).max()))
        finally:rig.close();e.close()
        common=None;stats0=None
        for view in VIEWS:
            b=load_data(task,view);stats=torch.load(stats_path(task,view),weights_only=True)
            fit=set(b['object_ids'][b['object_ids']<24].tolist());val=set(b['object_ids'][b['object_ids']>=48].tolist())
            assert len(fit)==24 and len(val)==8 and not fit&val
            assert not any(k in b for k in ('teacher','states','metrics','success'))
            if common is None:
                common=(b['target_tokens'].clone(),b['target_boxes'].clone());stats0=stats
            else:
                assert torch.equal(common[0],b['target_tokens']) and torch.equal(common[1],b['target_boxes'])
                for k in ('effect_sd','direction','progress_sd'):assert torch.equal(stats[k],stats0[k]),k
            if not view.endswith('w'):assert not bool(b['tokens'][:,:,1].any()) and not bool(b['boxes'][:,:,1].any())
            ids=torch.where(b['object_ids']>=48)[0];snaps=[s for s in pred['snapshots'] if s['task']==task]
            bases={t:torch.stack([s['candidates'][0] for s in snaps if s['t']==t]).cuda() for t in (20,40,60)}
            for seed in (11,22):
                for f in FAMILIES:
                    r=next(r for r in pred['rows'] if r['task']==task and r['view']==view and r['seed']==seed and r['family']==f and r['mode']=='continuous')
                    n=model(task,view,seed,f);weight=state_hash(n);st=MultiViewState(n,16,'continuous');zerr=0.;serr=0.;chosen=True
                    for t in range(61):
                        ob={k:b[k][ids,t] for k in ('tokens','boxes','proprio','target_tokens','target_boxes')}
                        _,c,s=st.act(ob,bases[t] if t in bases else bases[20])
                        if t in bases:
                            sl=slice(((t//20)-1)*16,(t//20)*16)
                            zerr=max(zerr,float((st.z.cpu()-r['z'][sl]).abs().max()));serr=max(serr,float((s.cpu()-r['scores'][sl]).abs().max()))
                            chosen &= torch.equal(c.cpu(),r['choice'][sl])
                        st.observe(b['actions'][ids,t],{k:b[k][ids,t+1] for k in ('tokens','boxes','proprio','target_tokens','target_boxes')})
                    assert zerr<2e-5 and serr<2e-5 and chosen,(task,view,seed,f,zerr,serr)
                    assert state_hash(n)==weight
                    checks.append({'task':task,'view':view,'seed':seed,'family':f,'z_error':zerr,'score_error':serr,'choices_match':chosen,'weights_frozen':True})
                    print('P35_CAUSAL_AUDIT',task,view,seed,f,flush=True)
            data_checks.append({'task':task,'view':view,'common_feedback_identical':True,'fit_objects':24,'validation_objects':8})
    assert camera_error<1e-10
    oc=torch.load(ROOT/'branches/OUTCOMES.pt',weights_only=True)['outcomes'];num=0;err=0.
    for o in oc:
        e=(ObjectEnv if o['task']==TASKS[0] else HeldHammerEnv)(o['config'],render=False)
        try:
            e.reset(o['reset_seed'])
            for j,tr in enumerate(o['traces']):
                vv=[]
                for t in (0,4):
                    for k,v in tr['states'].items():getattr(e.env.data,k)[:]=v[t].numpy()
                    mujoco.mj_forward(e.env.model,e.env.data);vv.append(physical_value(e,o['task']))
                err=max(err,abs(vv[1]-vv[0]-o['progress'][j]));num+=1
        finally:e.close()
    assert num==864 and err<1e-10
    for p in list(ROOT.glob('*/models/*/seed*/SOURCE_FREEZE.json'))+[ROOT/'branches/PREDICTION_SEAL.json']:
        for x in json.loads(p.read_text())['bindings']:assert digest(x['path'])==x['sha256'],x['path']
    dump(ROOT/'AUDIT.json',{'passed':True,'causal_replays':checks,'data_checks':data_checks,
         'native_render_shapes_passed':True,'camera_attachment_max_error':camera_error,'rendering_did_not_change_physics':True,
         'physical_branches_regraded':num,'physical_value_max_error':err,'full_task_rollouts':0,
         'bindings':bindings([ROOT/'branches/RESULTS.json',ROOT/'validation/PREDICTION_RESULTS.json',Path(__file__),Path('tools/fast_adapter/p35_runtime.py')])})
    print('P35_AUDIT_PASSED',flush=True)

if __name__=='__main__':main()
