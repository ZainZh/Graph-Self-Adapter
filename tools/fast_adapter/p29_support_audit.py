"""Inspect sealed RGB presence/masking and contact visibility; never policy input."""
import argparse,json,time
import torch
from tools.fast_adapter.p29_common import R,task_root,save_json,digest
from tools.fast_adapter.p29_eval import STRATA

def main(task):
    root=task_root(task);rnd=json.loads((root/'FINAL_FREEZE.json').read_text())['round'];rows={};sources=[]
    for st in STRATA:
        folder=root/'final'/rnd/st;result=json.loads((folder/'RESULT.json').read_text());n=result['conditions']
        counts={'support_episodes':0,'frames':0,'actor_present_frames':0,'object_present_frames':0,'held_steps':0,'actor_transition_present_when_held':0,'both_transition_present_when_held':0}
        for off in range(0,n,8):
            for source in ('correct','donor'):
                p=folder/f'SUPPORT_{source}_{off:03d}.pt';b=torch.load(p,weights_only=True);side=json.loads(p.with_suffix('.json').read_text());assert digest(p)==side['source_sha256']
                assert 'teacher' not in b
                v=b['visual'].reshape(len(b['visual']),-1,2,71);present=v[...,70]>0;valid=present[:,:-1]&present[:,1:]
                if task=='mug-handle-lift':assert (v[:,:,1]==0).all()
                held=torch.tensor([[bool(x['bilateral_handle']) for x in m] for m in side['metrics']]);counts['support_episodes']+=len(v);counts['frames']+=len(v)*v.shape[1]
                counts['actor_present_frames']+=int(present[...,0].sum());counts['object_present_frames']+=int(present[...,1].sum());counts['held_steps']+=int(held.sum());counts['actor_transition_present_when_held']+=int((valid[...,0]&held).sum());counts['both_transition_present_when_held']+=int((valid.all(-1)&held).sum())
                sources.append({'path':str(p),'sha256':digest(p)})
        rows[st]=counts
    out=R/'summary'/f'SUPPORT_INPUT_AUDIT_{task}.json';assert not out.exists()
    save_json(out,{'time':time.time(),'task':task,'round':rnd,'passed':True,'absent_second_slot_zero':task=='mug-handle-lift','training_labels_absent':True,'strata':rows,'sources':sources,'scope':'Descriptive presence in sealed support inputs, not proof that the visual representation captures useful physical relations. Contact flags used only for this post hoc diagnostic.'})
    print(task,rows,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args();main(a.task)
