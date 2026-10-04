"""One sealed P70 full-cell video from stored public RGB only (no simulator)."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []

def collect(cell):
    if (cell/'ABORT.json').exists():raise ValueError('Aborted cell cannot be presented as completed')
    result=read(cell/'RESULT.json');seal=read(cell/'SEAL.json')
    if result.get('full_task_success_evidence') is not True or result.get('stage_qualification_only') is not False:
        raise ValueError('Full-task evidence required; stage qualification cannot be labelled full task')
    if result['seal_sha256']!=sha(cell/'SEAL.json'):raise ValueError('Result/seal mismatch')
    for name,digest in seal['files'].items():
        if sha(cell/name)!=digest:raise ValueError('Seal mismatch: '+name)
    if result['original_initial_restores']!=1 or result['later_state_restores']!=0:raise ValueError('Not a continuous native-initial episode')
    prefix=result['prefix'];total=result['total_actions'];offset=prefix['actions'];frames={}
    for name in ['PUBLIC_SETTLING_TRACE.jsonl','controller/TRACE.jsonl']:
        for row in lines(cell/'prefix'/name):
            t=int(row['t']);stage=row.get('stage','public_settling');mode=row.get('action_diagnostics',{}).get('controller','Code')
            frames[t]=dict(t=t,path=cell/'prefix'/row['public'],stage=stage,controller='grasp MLP' if 'mlp' in mode.lower() else 'Code',update=False)
    s9rows=lines(cell/'s9/public/PREDICTIONS.jsonl')
    for row in s9rows:
        t=int(row['t']);global_t=offset+t;path=cell/f's9/public/frame_{t:04d}.npz'
        if sha(path)!=row['observation_sha256']:raise ValueError('s9 frame hash mismatch')
        if global_t in frames:
            if t!=0 or global_t!=offset:raise ValueError('Unexpected duplicated global timestamp')
            with np.load(frames[global_t]['path']) as left,np.load(path) as right:
                differences={key:float(np.max(np.abs(left[key]-right[key]))) for key in ('eef_pos','eef_quat_xyzw','gripper_qpos')}
            lhs=cell/f'prefix/private/frame_{offset:04d}.npz';rhs=cell/'s9/private/frame_0000.npz'
            if lhs.exists() and rhs.exists():
                with np.load(lhs) as left,np.load(rhs) as right:
                    if not np.array_equal(left['supervisor_sim_state'],right['supervisor_sim_state']):raise ValueError('Handoff simulator state discontinuity')
                result['_presentation_handoff_audit']=dict(full_sim_state_exact_equal=True,public_refresh_max_abs_difference=differences)
            elif any(v>1e-8 for v in differences.values()):raise ValueError('Handoff proprio discontinuity without exact-state evidence')
        frames[global_t]=dict(t=global_t,path=path,stage='p68_insert',controller='P68 FastAdapter',update=bool(row.get('telemetry',{}).get('event')))
    verification=result.get('placement_verification',{})
    verification_rows=lines(cell/'placement_verification/actions/VERIFY_ACTIONS.jsonl')
    if len(verification_rows)!=verification.get('actions',0):raise ValueError('Placement action/frame mismatch')
    for local,row in enumerate(verification_rows):
        global_t=offset+result['s9']['actions']+local+1
        if row['t']!=local or row['global_t']!=global_t or row.get('executed') is not True:raise ValueError('Invalid post-action placement time')
        if global_t in frames:raise ValueError('Duplicated placement handoff frame')
        path=cell/'placement_verification'/row['public']
        if sha(path)!=row['public_sha256']:raise ValueError('Placement public hash mismatch')
        if bool(row.get('complete'))!=bool(row.get('native_hold3')):raise ValueError('Placement strict/native hold mismatch')
        frames[global_t]=dict(t=global_t,path=path,stage='placement_verification',controller='Code stationary closed',update=False)
    if verification.get('graph_complete') and not any(r.get('complete') is True and r.get('native_hold3') is True for r in verification_rows):raise ValueError('Placement success without complete/hold3 evidence')
    if result.get('graph_complete') != bool(result['s9'].get('graph_complete') or verification.get('graph_complete')):raise ValueError('Strict result inconsistent with stage outcomes')
    if offset+result['s9']['actions']+verification.get('actions',0)!=total or set(frames)!=set(range(total+1)):raise ValueError('Missing/unknown timeline; no invented frames permitted')
    if s9rows and len(s9rows)!=result['s9']['actions']+1:raise ValueError('s9 action/frame mismatch')
    return result,[frames[t] for t in range(total+1)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cell',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    result,frames=collect(a.cell)
    if a.check_only:print(json.dumps(dict(frames=len(frames),actions=result['total_actions'],timeline_valid=True)));return
    if a.output is None:p.error('--output required for presentation')
    a.output.mkdir(parents=True,exist_ok=False)
    from PIL import Image,ImageDraw,ImageFont
    from tools.task_graph.p52_render_full_online import choose_font
    font=ImageFont.truetype(str(choose_font(None)),18)
    video=a.output/'full_task.mp4';pipe=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','896x608','-r','20','-i','-','-an','-c:v','libx264','-threads','1','-crf','22','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
    cell=result['cell'];last_stage=None;stage_entry=0
    stages=dict(public_settling='公开观测稳定',approach_book='接近书本',short_grasp='短程抓持',lift_clear='抬离桌面',orient_clear='安全定向',transport_above_caddy='运输至笔筒上方',near_entry='靠近插入口',p68_insert='P68局部插入',placement_verification='放置稳定性验证')
    try:
        for row in frames:
            with np.load(row['path']) as z:rgb=z['rgb']
            if rgb.shape!=(2,448,448,3):raise ValueError('Wrong native RGB dimensions')
            if row['stage']!=last_stage:last_stage=row['stage'];stage_entry=row['t']
            canvas=Image.new('RGB',(896,608),(16,22,32));draw=ImageDraw.Draw(canvas)
            for view in range(2):canvas.paste(Image.fromarray(rgb[view]),(view*448,110))
            draw.text((10,4),f"P70 {cell['version']} | 初态 {cell['init_id']} | 原生初态完整任务／无中途恢复",font=font,fill='white')
            draw.text((10,31),f"动作步 {row['t']}/{result['total_actions']} | 阶段：{stages[row['stage']]}（第{stage_entry}步进入）",font=font,fill='white')
            draw.text((10,59),f"控制器：{dict(Code='Code 控制器',**{'grasp MLP':'抓持小模型','P68 FastAdapter':'P68 局部适配器','Code stationary closed':'Code 静止闭合'})[row['controller']]} | 本帧局部更新：{'有' if row['update'] else '无'}",font=font,fill='white')
            draw.text((10,85),'全局视角',font=font,fill='white');draw.text((458,85),'腕部视角',font=font,fill='white')
            if row['t']==result['total_actions']:
                label=f"最终完整成功：{int(result['graph_complete'])}/1 | 原生持续成功：{int(result['official_success'])}/1 | 瞬时命中： {int(bool(result['s9'].get('native_instant_ever') or result.get('placement_verification',{}).get('native_instant_ever')))}/1"
                detail=f"总动作：{result['total_actions']} | {result.get('placement_verification',{}).get('stop_reason') if result.get('placement_verification',{}).get('actions',0) else result['s9']['stop_reason']}"
            else:label='真实连续任务记录；仅在终帧显示最终结果';detail='原始20Hz公开RGB；不启动物理、不补动作或中间画面'
            draw.text((10,564),label,font=font,fill=(255,215,120));draw.text((10,586),detail,font=font,fill='white')
            frame=np.asarray(canvas).tobytes();pipe.stdin.write(frame)
        for _ in range(40):pipe.stdin.write(frame)
    finally:pipe.stdin.close()
    if pipe.wait()!=0:raise RuntimeError('Encoder failed')
    write_manifest(a.cell,a.output,result,frames)

def write_manifest(cell,output,result,frames):
    video=output/'full_task.mp4'
    bindings={str(cell/name):sha(cell/name) for name in ['RESULT.json','SEAL.json','BINDING.json']}
    bindings[str(Path(__file__))]=sha(__file__)
    (output/'MANIFEST.json').write_text(json.dumps(dict(schema='prove.p70-full-video.v1',bindings=bindings,video_sha256=sha(video),fps=20,terminal_hold_seconds=2,physics_steps_added=0,handoff_audit=result.get('_presentation_handoff_audit'),handoff='prefix actions + s9 t; s9 t0 replaces identical-time prefix terminal; exact sealed simulator state checked when available, otherwise proprio checked; verification post-action offset+s9+local+1',frames=[dict(r,path=str(r['path']),sha256=sha(r['path'])) for r in frames]),indent=2)+'\n')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>P70 full recorded episode</title><video width="896" controls src="full_task.mp4"></video><p><a href="full_task.mp4">Native MP4</a></p>')
if __name__=='__main__':main()
