"""Five fixed terminal-window sealed P69 states; zero integration or policy calls."""
import argparse,json,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.task_graph.p69_video import metadata,scene_binding,sha
from tools.task_graph.p52_render_full_online import choose_font


def build(root,output):
    episode=metadata(root);scene,chain=scene_binding(episode['binding'])
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    import mujoco
    model=mujoco.MjModel.from_xml_string(scene['environment']['relocated_xml']);data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=448,width=448);option=mujoco.MjvOption();option.geomgroup[0]=0;option.geomgroup[1]=1
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    canvas=Image.new('RGB',(896,5*488),(17,23,34));draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(str(choose_font(None)),22)
    records=[];total=episode['result']['total_actions']
    try:
        for i,offset in enumerate((40,30,20,10,0)):
            index=max(0,total-offset);row=episode['timeline'][index]
            with np.load(Path(root)/row['state'],allow_pickle=False) as x:data.qpos[:]=x['supervisor_qpos']
            mujoco.mj_fwdPosition(model,data)
            label=f"R2 初态{episode['cell']['init_id']} 执行步{index}/{total} 阶段{row['stage']} 距末态{offset}步"
            draw.text((10,i*488+8),label,font=font,fill='white')
            for view,camera in enumerate(scene['public']['cameras']):
                renderer.update_scene(data,camera=camera,scene_option=option)
                picture=Image.fromarray(renderer.render().copy());canvas.paste(picture,(view*448,i*488+40))
                picture.save(output/f'step_{index:04d}_view{view}.png')
            records.append(dict(index=index,offset=offset,source=row,source_sha256=episode['source_hashes'][row['state']]))
    finally:renderer.close()
    canvas.save(output/'CONTACT_SHEET.png')
    (output/'MANIFEST.json').write_text(json.dumps(dict(episode=str(root),result_sha256=sha(Path(root)/'RESULT.json'),seal_sha256=sha(Path(root)/'SEAL.json'),selected=records,physics_steps=0,policy_calls=0,scene_chain=chain,files={p.name:sha(p) for p in output.glob('*.png')}),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();build(a.episode,a.output)
