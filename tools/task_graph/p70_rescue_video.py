"""Selected sealed same-init development comparison, public RGB only."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.task_graph.p70_full_video_zh import collect,lines,sha
from tools.task_graph.p52_render_full_online import choose_font
STAGES=dict(public_settling='公开观测稳定',approach_book='接近书本',short_grasp='短程抓持',lift_clear='抬离桌面',orient_clear='安全定向',transport_above_caddy='运输至笔筒上方',near_entry='靠近插入口',p68_insert='P68局部插入',placement_verification='放置稳定性验证')

def main():
 p=argparse.ArgumentParser();p.add_argument('--cell',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if len(a.cell) not in (1,2):raise ValueError('One or two selected cells only')
 episodes=[]
 for cell in a.cell:
  result,frames=collect(cell)
  if result['cell']['init_id']!=33:raise ValueError('Only Root-selected init33')
  events=[]
  for row in lines(cell/'prefix/controller/TRACE.jsonl'):
   event=row.get('action_diagnostics',{}).get('lift_clearance_refinement',{})
   if event.get('applied') is True:events.append(dict(t=row['t'],stage=row['stage'],event=event))
  episodes.append((result,frames,events))
 a.output.mkdir(parents=True,exist_ok=False);font=ImageFont.truetype(str(choose_font(None)),18);width=896*len(episodes);height=664
 video=a.output/'full_task.mp4';proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','20','-i','-','-an','-c:v','libx264','-threads','1','-crf','22','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
 last=max(len(e[1]) for e in episodes)-1
 try:
  for global_t in range(last+41):
   canvas=Image.new('RGB',(width,height),(16,22,32));d=ImageDraw.Draw(canvas)
   for panel,(result,frames,events) in enumerate(episodes):
    x=896*panel;t=min(global_t,len(frames)-1);row=frames[t];terminal=t==len(frames)-1
    with np.load(row['path']) as z:rgb=z['rgb']
    for v in range(2):canvas.paste(Image.fromarray(rgb[v]),(x+448*v,164))
    d.text((x+10,5),f"P70 {result['cell']['version']} | 开发初态33 | 原生初态连续完整回合",font=font,fill='white')
    d.text((x+10,32),f"动作 {t}/{result['total_actions']} | {STAGES[row['stage']]}"+(' | 已结束，终帧保持' if terminal else ''),font=font,fill='white')
    d.text((x+10,59),f"控制器：{row['controller']} | 本帧局部更新：{'有' if row['update'] else '无'}",font=font,fill='white')
    seen=[e for e in events if e['t']<=t]
    eventlabel=(f"间隙修订已触发：第{seen[-1]['t']}步／{STAGES[seen[-1]['stage']]}（仅一次）" if seen else '尚无间隙修订触发')
    d.text((x+10,86),eventlabel,font=font,fill=(255,210,100))
    d.text((x+10,113),'外层图修订的代表例；各版本开发分母均为10，不归因于Adapter独立收益',font=font,fill='white')
    d.text((x+10,139),'全局视角',font=font,fill='white');d.text((x+458,139),'腕部视角',font=font,fill='white')
    label=(f"最终完整成功：{int(result['graph_complete'])}/1 | 原生持续：{int(result['official_success'])}/1" if terminal else '真实封存轨迹；结果仅在各自终帧显示')
    d.text((x+10,618),label,font=font,fill=(255,210,100));d.text((x+10,641),'20Hz；短片结束后只定格，不补动作、不改变时间顺序',font=font,fill='white')
   proc.stdin.write(np.asarray(canvas).tobytes())
 finally:proc.stdin.close()
 if proc.wait():raise RuntimeError('Video encoding failed')
 inputs=[]
 for cell,(result,frames,events) in zip(a.cell,episodes):
  inputs.append(dict(cell=str(cell),events=events,bindings={str(cell/name):sha(cell/name) for name in ['RESULT.json','SEAL.json','BINDING.json','prefix/controller/TRACE.jsonl']},frames=[dict(t=r['t'],path=str(r['path']),sha256=sha(r['path'])) for r in frames]))
 (a.output/'MANIFEST.json').write_text(json.dumps(dict(schema='prove.p70-selected-rescue-video.v1',sources={str(Path(__file__)):sha(__file__),str(Path('tools/task_graph/p70_full_video_zh.py')):sha('tools/task_graph/p70_full_video_zh.py')},inputs=inputs,video_sha256=sha(video),fps=20,terminal_hold_seconds=2,physical_steps_added=0),indent=2)+'\n')
 (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>P70开发初态33</title><p>同初态开发救回例；各版分母10。外层图修订，不声称Adapter独立收益。短侧结束后保持真实终帧。</p><video controls width="100%" src="full_task.mp4"></video><a href="full_task.mp4">原生视频</a>')
if __name__=='__main__':main()
