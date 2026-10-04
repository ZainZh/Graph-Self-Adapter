"""Two annotated representative comparisons, reconstructed from sealed states."""
import subprocess
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from tools.fast_adapter.p33_common import *
from prove.fast_adapter.held_hammer_env import HeldHammerEnv
from prove.fast_adapter.object_interaction_env import mujoco

ARMS=['base','cross_fixed','cross_first3','cross_continuous','within_continuous']
LABELS=['冻结动作基座','关系模型 · 固定z','关系模型 · 前三次','关系模型 · 持续更新','物体内模型 · 持续更新']

def main():
    out=ROOT/'summary/media';out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((ROOT/'development/RESULT.json').read_text())['rows']
    lookup={(r['seed'],r['split'],r['condition'],r['arm']):r for r in rows}
    picks={};fallback=[]
    for split in ('geometry_ood','train'):
        for seed in (11,22):
            for c in range(8):
                a=lookup[seed,split,c,'cross_continuous'];b=lookup[seed,split,c,'cross_fixed']
                if a['source_row'] is None:continue
                if a['success'] and not b['success']:picks.setdefault('adaptation_rescue',(seed,split,c))
                if b['success'] and not a['success']:picks.setdefault('adaptation_harm',(seed,split,c))
                fallback.append(('success' if a['success'] else 'failure',(seed,split,c)))
    chosen=list(picks.items())
    for item in fallback:
        if len(chosen)>=2:break
        if item[1] not in [v for _,v in chosen]:chosen.append(item)
    fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';title=ImageFont.truetype(fp,22);font=ImageFont.truetype(fp,17)
    videos=[]
    for kind,(seed,split,c) in chosen:
        rr={a:lookup[None if a=='base' else seed,split,c,a] for a in ARMS}
        b={a:torch.load(rr[a]['source']['path'],weights_only=True) for a in ARMS}
        meta={a:json.loads(Path(rr[a]['source']['path']).with_suffix('.json').read_text()) for a in ARMS}
        env=HeldHammerEnv(rr['base']['config']);env.reset(rr['base']['reset_seed'])
        path=out/f'hammer_{split}_{kind}_s{seed}_c{c}.mp4';w,h=1600,600
        proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}',
            '-framerate','12','-i','-','-c:v','libx264','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
        try:
            for t in range(201):
                im=Image.new('RGB',(w,h),(21,30,39));draw=ImageDraw.Draw(im)
                draw.text((12,9),f'P33 已持锤子作用木块｜'+('范围外新锤子' if split=='geometry_ood' else '范围内新锤子')+f'｜执行步 {t}/200｜种子{seed} · 条件{c}',font=title,fill='white')
                draw.text((12,48),'同初态配对；任务为水平工具推击。接触与成功注释是封存后的物理评价，不进入动作或z更新。',font=font,fill=(190,210,220))
                for col,a in enumerate(ARMS):
                    x=col*320;j=rr[a]['source_row'];draw.text((x+8,87),LABELS[col],font=font,fill=(146,227,207))
                    for k,ss in b[a]['states'].items():getattr(env.env.data,k)[:]=ss[j,t].numpy()
                    mujoco.mj_forward(env.env.model,env.env.data);im.paste(Image.fromarray(env.rgb()).resize((296,296)),(x+12,115))
                    if a=='base':count=0;changed=0;phase='基座执行'
                    else:
                        count=b[a]['counts_before_action'][t] if t<200 else b[a]['updates']
                        changed=sum(float(d['delta_norm'][j])>1e-10 for d in b[a]['update_diagnostics'][:t])
                        phase='预测效果，选择下一动作'
                        if a=='cross_first3' and count>=3:phase='已更新3次：z保持固定'
                        if a=='cross_fixed':phase='相同规划流程：z全程固定'
                    draw.text((x+8,419),f'梯度更新 {count} 次｜z改变 {changed} 次',font=font,fill='white')
                    draw.text((x+8,448),phase,font=font,fill=(190,213,233))
                    if t:
                        m=meta[a]['metrics'][j][t-1]
                        stage='已达到任务成功判据' if m['success'] else '锤头作用到木块' if m['head_target_contact'] else '移动或调整，当前未接触木块'
                        if not m['bilateral_handle']:stage='当前未保持双侧工具抓持'
                        draw.text((x+8,477),stage,font=font,fill='white')
                        draw.text((x+8,506),f'木块位移 {m["target_displacement"]*1000:.1f} mm｜要求45 mm',font=font,fill=(190,213,233))
                    if t==200:
                        msg='回合成功，末步达标' if rr[a]['terminal_success'] else '回合曾成功，末步未达标' if rr[a]['success'] else '本回合失败'
                        draw.text((x+8,551),msg,font=font,fill=(128,229,160) if rr[a]['success'] else (255,160,120))
                proc.stdin.write(np.asarray(im).tobytes())
                if t==200:im.save(path.with_suffix('.png'))
            proc.stdin.close();assert proc.wait()==0
        finally:
            env.close()
            if proc.poll() is None:proc.terminate();proc.wait()
        videos.append({'kind':kind,'seed':seed,'split':split,'condition':c,'file':str(path),'sha256':digest(path),'frames':201,
            'outcomes':{a:rr[a]['success'] for a in ARMS},'sources':bindings(sorted({Path(r['source']['path']) for r in rr.values()}))})
        print('VIDEO_COMPLETE',path,flush=True)
    dump(out/'VIDEO_INDEX.json',{'selection':'First continuous/fixed rescue and harm, OOD then ID, seeds11/22; fill success/failure if absent.',
         'state_reconstruction_no_new_physical_trials':True,'generator_sha256':digest(Path(__file__)),'videos':videos})

if __name__=='__main__':main()
