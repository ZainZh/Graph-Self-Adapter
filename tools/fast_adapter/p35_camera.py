"""Native-resolution, fixed eye-in-hand camera; robot calibration only."""
import json
from pathlib import Path
import numpy as np
from prove.fast_adapter.object_interaction_env import mujoco

ROOT=Path('evaluations/fast_adapter_p35_resolution_views_v01')

def make_mount(env):
    m=env.env.model;d=env.env.data;c=int(m.camera('gripperPOV').id);body=int(m.cam_bodyid[c])
    r=d.body(body).xmat.reshape(3,3);p=d.body(body).xpos;tcp=env.public()[:3]
    pos=tcp+np.array([.10,-.08,.10]);look=tcp+np.array([.06,.05,-.02])
    forward=look-pos;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    up=np.cross(right,forward);rot=np.stack((right,up,-forward),axis=1);q=np.zeros(4)
    mujoco.mju_mat2Quat(q,(r.T@rot).reshape(-1))
    return {'pos':(r.T@(pos-p)).tolist(),'quat':q.tolist(),'fovy':100.,
            'source':'fixed wrist mount calibrated to measured robot EEF only; no object targeting'}

def calibration(renderer):
    cs=renderer.scene.camera
    return {k:np.mean([np.array(getattr(c,k)) for c in cs],axis=0).tolist()
            for k in ('pos','forward','up','frustum_bottom','frustum_top','frustum_center','frustum_near')}

class CameraRig:
    def __init__(self,env,mount):
        self.env=env;m=env.env.model;c=int(m.camera('gripperPOV').id)
        m.cam_mode[c]=mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        m.cam_pos[c]=mount['pos'];m.cam_quat[c]=mount['quat'];m.cam_fovy[c]=mount['fovy']
        m.vis.global_.offwidth=448;m.vis.global_.offheight=448
        self.renderers={s:mujoco.Renderer(m,s,s) for s in (224,448)}
    def render(self,size,view):
        renderer=self.renderers[size]
        renderer.update_scene(self.env.env.data,camera=self.env.cam if view=='global' else 'gripperPOV')
        return renderer.render().copy(),calibration(renderer)
    def close(self):
        for r in self.renderers.values():r.close()

def preflight():
    import torch
    from PIL import Image,ImageDraw
    from tools.fast_adapter.p34_branches import raw_source
    from prove.fast_adapter.object_interaction_env import ObjectEnv
    from prove.fast_adapter.held_hammer_env import HeldHammerEnv
    out=ROOT/'preflight';out.mkdir(parents=True,exist_ok=True);mount=None
    canvas=Image.new('RGB',(1280,720),(25,30,35))
    for row,task in enumerate(('mug-handle-lift','hammer-block')):
        b,cfg,j,_=raw_source(task,'expert',0);e=(ObjectEnv if row==0 else HeldHammerEnv)(cfg);e.reset(b['reset_seeds'][j])
        if mount is None:
            mount=make_mount(e);(out/'WRIST_MOUNT.json').write_text(json.dumps(mount,indent=2)+'\n')
        rig=CameraRig(e,mount)
        try:
            for col,t in enumerate((20,40,60,100)):
                for k,v in b['states'].items():getattr(e.env.data,k)[:]=v[j,t].numpy()
                mujoco.mj_forward(e.env.model,e.env.data);rgb,_=rig.render(448,'wrist')
                im=Image.fromarray(rgb);im.save(out/f'{task}_mount_t{t}.png')
                canvas.paste(im.resize((320,320)),(col*320,row*360+36))
                ImageDraw.Draw(canvas).text((col*320+4,row*360+6),f'{task} mounted wrist t{t}',fill='white')
        finally:rig.close();e.close()
    canvas.save(out/'WRIST_MOUNT_REVIEW.jpg',quality=80)

if __name__=='__main__':preflight()
