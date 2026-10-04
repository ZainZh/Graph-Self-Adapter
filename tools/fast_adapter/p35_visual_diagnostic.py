"""Display actual native pixels and visible object support, without new rollouts."""
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.fast_adapter.p35_common import *
from tools.fast_adapter.p35_camera import CameraRig
from tools.fast_adapter.p35_visual import crops_for
from tools.fast_adapter.p34_branches import raw_source
from prove.fast_adapter.object_interaction_env import ObjectEnv,mujoco
from prove.fast_adapter.held_hammer_env import HeldHammerEnv

def main():
    out=ROOT/'visual_review';out.mkdir(exist_ok=False);mount=json.loads((ROOT/'preflight/WRIST_MOUNT.json').read_text())
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20);rows=[]
    for task in TASKS:
        b,cfg,j,path=raw_source(task,'expert',0);e=(ObjectEnv if task==TASKS[0] else HeldHammerEnv)(cfg);e.reset(b['reset_seeds'][j])
        rig=CameraRig(e,mount);t=40 if task==TASKS[0] else 20
        im=Image.new('RGB',(1440,790),(22,31,40));draw=ImageDraw.Draw(im)
        try:
            for k,v in b['states'].items():getattr(e.env.data,k)[:]=v[j,t].numpy()
            mujoco.mj_forward(e.env.model,e.env.data)
            for col,(size,view) in enumerate(((224,'global'),(448,'global'),(224,'wrist'),(448,'wrist'))):
                rgb,cal=rig.render(size,view);crops,boxes,visible,rects,pixels=crops_for(rgb[None],b['proprio'][j,t:t+1].numpy(),[cal],task)
                Image.fromarray(rgb).save(out/f'{task}_{view}{size}.png')
                title=('外部' if view=='global' else '腕部')+f' 原生{size}×{size}'
                draw.text((col*360+10,12),title,font=font,fill='white')
                im.paste(Image.fromarray(rgb).resize((344,344)),(col*360+8,52))
                crop=Image.fromarray(crops[0,1]).resize((300,300));im.paste(crop,(col*360+30,440))
                width,height=boxes[0,1,2:]*size
                draw.text((col*360+10,399),f'物体框约 {width:.0f}×{height:.0f} 原生像素',font=font,fill=(127,232,203))
                draw.text((col*360+10,744),f'物体可见色标像素：{pixels[0,1]}',font=font,fill='white')
                rows.append({'task':task,'source_object':0,'source_role':'expert','step':t,'view':view,'resolution':size,
                             'object_bbox_native_pixels':[float(width),float(height)],'object_mask_pixels':int(pixels[0,1]),
                             'object_present':bool(visible[0,1]),'image':str(out/f'{task}_{view}{size}.png')})
            im.save(out/f'{task}_comparison.jpg',quality=85)
        finally:rig.close();e.close()
    dump(out/'PIXEL_EXAMPLES.json',{'display_resize_only':True,'native_images_saved':True,'no_performance_claim':True,
         'rows':rows,'bindings':bindings([Path(__file__),ROOT/'preflight/WRIST_MOUNT.json'])})

if __name__=='__main__':main()
