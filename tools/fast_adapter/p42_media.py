"""Post-seal Chinese annotated video derivatives, following frozen selection."""
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
from prove.fast_adapter.geometry3d_env import GeometrySkill
from tools.fast_adapter.p41_run import js,sha
from tools.fast_adapter.p42_batch import ROOT

LABEL={'single':'单帧视觉','fixed':'同模型固定几何','gradient':'持续几何梯度'}
TASK={'cup':'杯柄抓持与抬升','insert':'倾斜槽口插入'}


def font(size):
    p=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    return ImageFont.truetype(p,size)


def main():
    results=json.loads((ROOT/'summary/RESULTS.json').read_text())
    out=ROOT/'summary';entries=[];large=font(23);small=font(17);tiny=font(16)
    for task in ['cup','insert']:
        for wanted in [True,False]:
            matches=sorted(r['seed'] for r in results['rows']
                           if r['task']==task and r['split']=='geometry' and r['arm']=='gradient'
                           and r['execution_status']=='complete' and r['success']==wanted)
            if not matches:
                continue
            seed=matches[0];kind='success' if wanted else 'failure';records=[]
            for arm in ['single','fixed','gradient']:
                p=ROOT/'independent_v01'/task/f'geometry_{seed}_{arm}'
                if not (p/'RESULT.json').exists():
                    raise RuntimeError('Selected paired media contains an execution exception')
                pre=json.loads((p/'PRE_RUN.json').read_text())
                records.append(dict(arm=arm,path=p,env=GeometrySkill(pre['cfg']),
                    trace=np.load(p/'TRACE.npy',allow_pickle=True),
                    obs=np.load(p/'OBS.npy',allow_pickle=True),
                    result=json.loads((p/'RESULT.json').read_text())))
            name=f'{task}_geometry_{kind}_{seed}.mp4';W,H=1344,1120
            corner_audit=dict(compared_frames=0,missingness_mismatches=0,coordinate_mismatches=0,max_coordinate_error_px=0.)
            proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo',
                '-pix_fmt','rgb24','-s',f'{W}x{H}','-r','20','-i','-','-an','-c:v','libx264',
                '-threads','2','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out/name)],stdin=subprocess.PIPE)
            try:
                final=None
                for t in range(190):
                    frame=Image.new('RGB',(W,H),'#edf2f5');d=ImageDraw.Draw(frame)
                    title=TASK[task]+f' · 范围外几何 {seed} · '+('预注册首个梯度成功条件' if wanted else '预注册首个梯度失败条件')
                    d.text((16,8),title,font=large,fill='#173247')
                    for col,r in enumerate(records):
                        x=448*col;e=r['env'];q=r['trace'][t]
                        e.data.qpos[:]=q['qpos'];e.data.qvel[:]=q['qvel'];e.mj.mj_forward(e.model,e.data)
                        o,rgb,mask=e.observe();old=r['obs'][t]
                        corner_audit['compared_frames']+=1
                        if (o['corners'] is None)!=(old['corners'] is None):
                            corner_audit['missingness_mismatches']+=1
                        elif o['corners'] is not None:
                            error=float(np.max(np.abs(o['corners']-old['corners'])))
                            corner_audit['coordinate_mismatches']+=int(error>0)
                            corner_audit['max_coordinate_error_px']=max(corner_audit['max_coordinate_error_px'],error)
                        frame.paste(Image.fromarray(e.render_global()),(x,82))
                        frame.paste(Image.fromarray(rgb),(x,530))
                        d.text((x+10,47),LABEL[r['arm']],font=large,fill='#173247')
                        for yy,caption in [(86,'斜视全景'),(534,'工具相机')]:
                            d.rectangle((x+4,yy-2,x+104,yy+25),fill='#183043')
                            d.text((x+8,yy),caption,font=small,fill='white')
                        d.text((x+8,983),f'步 {t+1}/190 · '+q['stage'],font=small,fill='#173247')
                        est=q['estimate']
                        if est is None:
                            lines=['几何估计：暂无可用轮廓','适配位置／朝向：尚未形成']
                        else:
                            p,Q,size=est;angles=Rotation.from_matrix(Q).as_euler('xyz',degrees=True)
                            lines=['估计位置 XYZ(mm): '+', '.join(f'{v:.1f}' for v in p*1000),
                                   '估计朝向 XYZ(度): '+', '.join(f'{v:.1f}' for v in angles)]
                        for i,line in enumerate(lines):d.text((x+8,1009+23*i),line,font=tiny,fill='#173247')
                        if t==189:
                            d.text((x+8,1060),'末10状态：'+('成功并保持' if r['result']['success'] else '未达成功判据'),
                                   font=small,fill='#007d68' if r['result']['success'] else '#b43c39')
                    d.text((16,1091),'封存轨迹回放；只适配位置与朝向。数值来自控制器估计，成功在封存后评价。',font=small,fill='#526675')
                    proc.stdin.write(np.asarray(frame).tobytes());final=frame
                for _ in range(30):proc.stdin.write(np.asarray(final).tobytes())
                cover=name.replace('.mp4','.jpg');final.save(out/cover)
                proc.stdin.close();assert proc.wait()==0
            finally:
                if proc.poll() is None:proc.terminate();proc.wait()
                for r in records:r['env'].close()
            entries.append(dict(task=task,split='geometry',seed=seed,kind=kind,file=name,cover=cover,
                selection='lowest declared OOD seed with gradient '+kind,
                arms=['single','fixed','gradient'],post_seal_derivative=True,
                all_rendered_corner_observations_exactly_reproduced=(corner_audit['missingness_mismatches']==0 and corner_audit['coordinate_mismatches']==0),
                rendered_corner_audit=corner_audit,
                render_boundary='Images are reconstructed after mj_forward on sealed qpos/qvel; saved online observations remain the action-audit source. Any rendering/corner difference is retained and is not substituted into metrics.',
                sha256=sha(out/name),cover_sha256=sha(out/cover),
                trace_sources={r['arm']:{'path':str(r['path']/'TRACE.npy'),'sha256':sha(r['path']/'TRACE.npy')} for r in records}))
            print(name,flush=True)
    js(out/'VIDEOS.json',entries)


if __name__=='__main__':main()
