"""Render only stored public RGB, without environment or inference."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import json,hashlib,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def main():
 root=Path('evaluations/outer_rsi_p70_object_graph_v01');cell=root/'qualification/Q0b/queue/cell_000';out=root/'presentation_v01';out.mkdir(exist_ok=True)
 result=json.loads((cell/'PREFIX_RESULT.json').read_text());assert result['qualified'] and result['completed_stage']=='lift_clear' and result['actions']==220
 traces={}
 for name in ['PUBLIC_SETTLING_TRACE.jsonl','controller/TRACE.jsonl']:
  for line in (cell/'prefix'/name).read_text().splitlines():
   r=json.loads(line);traces[r['t']]=r
 assert set(traces)==set(range(221))
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
 video=out/'qualification_init20.mp4'
 pipe=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','896x584','-r','20','-i','-','-an','-c:v','libx264','-threads','1','-crf','22','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
 sources={}
 try:
  for t in range(221):
   row=traces[t];path=cell/'prefix'/row['public'];digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==row['public_sha256'];sources[str(path)]=digest
   with np.load(path) as z:rgb=z['rgb']
   stage=row.get('stage','public_settling');mode=row.get('action_diagnostics',{}).get('controller','Code stationary open')
   controller='grasp MLP' if 'mlp' in mode else 'Code'
   canvas=Image.new('RGB',(896,584),(16,22,32));d=ImageDraw.Draw(canvas)
   for v in range(2):canvas.paste(Image.fromarray(rgb[v]),(448*v,100))
   d.text((10,5),'P70 | Init 20 | Book approach -> grasp -> lift qualification',font=font,fill='white')
   d.text((10,32),f'Step {t}/220 | Stage: {stage} | Controller: {controller}',font=font,fill='white')
   facts=row.get('facts',{});d.text((10,60),f"Attached: {facts.get('attached','unknown')} | Lift complete: {facts.get('lift_clear',False)}",font=font,fill='white')
   label='STAGE QUALIFICATION ONLY / NOT FULL TASK SUCCESS'
   if t==220:label='PASS: approach, grasp, lift | 220 actions | NOT full task success'
   d.text((10,556),label,font=font,fill=(255,213,113));frame=np.asarray(canvas).tobytes();pipe.stdin.write(frame)
  for _ in range(40):pipe.stdin.write(frame)
 finally:pipe.stdin.close()
 if pipe.wait()!=0:raise RuntimeError('Video encoder failed')
 (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>P70 stage qualification</title><h1>Init20: approach, grasp, lift qualification</h1><p>NOT full task success. Stored dual-view RGB, original 20Hz; no new physics or environment rendering. Includes 6 settling actions; 220 total actions. Terminal frame held 2 seconds.</p><video controls width="896" src="qualification_init20.mp4"></video><p><a href="qualification_init20.mp4">Native MP4</a></p>')
 (out/'qualification_init20_manifest.json').write_text(json.dumps(dict(video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),source_files=sources,fps=20,terminal_hold_seconds=2,physical_actions_added=0,source_prefix_result_sha256=hashlib.sha256((cell/'PREFIX_RESULT.json').read_bytes()).hexdigest()),indent=2)+'\n')
if __name__=='__main__':main()
