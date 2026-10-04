"""Check the native pipeline's 224 preprocessing against the frozen predecessor."""
import numpy as np
from PIL import Image
from tools.fast_adapter.p35_common import ROOT,TASKS,dump,digest,bindings,Path,json
from tools.fast_adapter.p35_visual import crops_for
from tools.fast_adapter.p34_visual import gripper_box
from tools.fast_adapter.p29_common import entities
from tools.fast_adapter.p35_camera import CameraRig
from tools.fast_adapter.p34_branches import raw_source
from prove.fast_adapter.object_interaction_env import ObjectEnv,mujoco
from prove.fast_adapter.held_hammer_env import HeldHammerEnv


def main():
    mount=json.loads((ROOT/'preflight/WRIST_MOUNT.json').read_text());rows=[]
    for task,t in zip(TASKS,(40,20)):
        raw,cfg,j,_=raw_source(task,'expert',0)
        e=(ObjectEnv if task==TASKS[0] else HeldHammerEnv)(cfg);e.reset(raw['reset_seeds'][j]);rig=CameraRig(e,mount)
        try:
            for k,v in raw['states'].items():getattr(e.env.data,k)[:]=v[j,t].numpy()
            mujoco.mj_forward(e.env.model,e.env.data)
            im,cal=rig.render(224,'global');q=raw['proprio'][j,t].numpy()
            c,b,p,_,_=crops_for(im[None],q[None],[cal],task)
            old,meta,present=entities(im[None],task)
            if task==TASKS[0]:
                old[:,1]=old[:,0];meta[:,1]=meta[:,0];present[:,1]=present[:,0]
                box=gripper_box(q,cal);assert box is not None
                old[0,0]=np.asarray(Image.fromarray(im).crop(box).resize((224,224),Image.Resampling.BILINEAR))
                x0,y0,x1,y1=box;meta[0,0,:4]=[(x0+x1)/224-1,(y0+y1)/224-1,(x1-x0)/224,(y1-y0)/224];present[0,0]=1
            assert np.array_equal(c,old) and np.array_equal(b,meta[...,:4]) and np.array_equal(p,present)
            rows.append({'task':task,'object_id':0,'t':t,'crops_boxes_presence_exactly_equal':True})
        finally:rig.close();e.close()
    previous=Path('evaluations/fast_adapter_p34_role_tokens_v01/VISUAL_SOURCE_RECHECK.json')
    refs=json.loads(previous.read_text())['files']
    for r in refs:assert digest(r['path'])==r['sha256'],r['path']
    dump(ROOT/'VISION_AUDIT.json',{'passed':True,'frozen_vision_sources_match_P34':refs,
        'low_resolution_preprocessing_examples':rows,'limits':'two training examples; not an exhaustive feature-value replay',
        'bindings':bindings([previous,Path(__file__)])})
    print('P35_VISION_AUDIT_PASSED',flush=True)


if __name__=='__main__':main()
