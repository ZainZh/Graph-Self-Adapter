"""Post-seal P41 audit, task tables and annotated desktop review."""
import argparse,hashlib,json,html,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.fast_adapter.p41_run import ROOT,sha,js
from prove.fast_adapter.geometry3d_env import GeometrySkill
from prove.fast_adapter.geometry3d_adapter import GeometryLatent,command

LABEL={'single':'单帧视觉位置控制','fixed':'同模型固定几何 z','prefix':'前12步更新后固定','gradient':'持续几何梯度更新','multiview':'同信息多视图拟合'}
TASK={'cup':'杯柄抓取并抬升','insert':'三维倾斜插入'}

def audit():
    frozen=json.loads((ROOT/'DEV_FREEZE.json').read_text());rows=[];prefixes={};max_state=0.;max_action=0.;replay_count=0
    for task in frozen['tasks']:
        assert (ROOT/'development_v01'/(task+'_DONE.json')).exists()
        for split in frozen['splits']:
            for seed in frozen['seeds']:
                for arm in frozen['arms']:
                    path=ROOT/'development_v01'/task/f'{split}_{seed}_{arm}'
                    pre=json.loads((path/'PRE_RUN.json').read_text());seal=json.loads((path/'SEAL.json').read_text());r=json.loads((path/'RESULT.json').read_text())
                    for key,file in [('trace','TRACE.npy'),('observations','OBS.npy'),('terminal','TERMINAL.npy')]:assert sha(path/file)==seal[key]
                    for source,h in pre['sources'].items():assert h==frozen['sources'][source] and sha(path/'source'/source)==h
                    trace=np.load(path/'TRACE.npy',allow_pickle=True);obs=np.load(path/'OBS.npy',allow_pickle=True);terminal=np.load(path/'TERMINAL.npy',allow_pickle=True).item()
                    assert len(trace)==len(obs)==190
                    key=(task,split,seed);prefix=np.array([t['qpos'] for t in trace[:13]])
                    if key in prefixes:assert np.allclose(prefix,prefixes[key],atol=1e-12,rtol=0)
                    else:prefixes[key]=prefix
                    env=GeometrySkill(pre['cfg'],render=False);n=GeometryLatent(task,arm);anchor=None
                    try:
                        for t,x in enumerate(trace):
                            if task!='cup' or t<105:n.update(obs[t],t)
                            est=n.estimate()
                            if task=='cup' and t==105 and est is not None:
                                pp,Q,size=est;anchor=(pp+Q[:,0]*(size[0]-.003)-Q[:,2]*.004,Q.copy())
                            pp,Q,closed,stage=command(task,t,est,env.start_pos,env.start_R,r['tool_calibration'].get('length',0.),anchor)
                            delta=max(np.max(np.abs(pp-x['command_p'])),np.max(np.abs(Q-x['command_R'])))
                            max_action=max(max_action,float(delta));assert delta<1e-10 and closed==x['closed'] and stage==x['stage']
                            error=float(np.max(np.abs(env.data.qpos-x['qpos'])));max_state=max(max_state,error);assert error<1e-8,(path,t,error)
                            env.step(x['command_p'],x['command_R'],x['closed'])
                        assert np.max(np.abs(env.data.qpos-terminal['qpos']))<1e-8
                        good=[];penetration=[]
                        for state in list(trace[-9:])+[terminal]:
                            env.data.qpos[:]=state['qpos'];env.data.qvel[:]=state['qvel'];env.mj.mj_forward(env.model,env.data)
                            good.append(env.facts()['success']);penetration.append(max([max(0.,-float(c.dist)) for c in env.data.contact]+[0.]))
                        assert bool(all(good) and max(penetration)<.002)==r['success']
                        replay_count+=1
                    finally:env.close()
                    rows.append(dict(r,source=str(path/'RESULT.json')))
    cells=[]
    for task in frozen['tasks']:
        for split in frozen['splits']:
            for arm in frozen['arms']:
                selected=[r for r in rows if (r['task'],r['split'],r['arm'])==(task,split,arm)]
                cells.append(dict(task=task,split=split,arm=arm,n=len(selected),successes=sum(r['success'] for r in selected)))
    pairs=[]
    for task in frozen['tasks']:
        for split in frozen['splits']:
            cur={r['seed']:r['success'] for r in rows if (r['task'],r['split'],r['arm'])==(task,split,'gradient')}
            for arm in ['single','fixed','prefix','multiview']:
                ref={r['seed']:r['success'] for r in rows if (r['task'],r['split'],r['arm'])==(task,split,arm)}
                pairs.append(dict(task=task,split=split,reference=arm,wins=sum(cur[k] and not ref[k] for k in cur),losses=sum(ref[k] and not cur[k] for k in cur),ties=sum(cur[k]==ref[k] for k in cur)))
    js(ROOT/'summary/RESULTS.json',dict(scope='development only; not DINO or meta-trained neural adapter',rows=rows,cells=cells,paired=pairs))
    js(ROOT/'summary/AUDIT.json',dict(episodes=replay_count,actions=replay_count*190,all_source_and_trace_hashes=True,common_prefix_equal=True,max_replayed_state_error=max_state,max_replayed_action_error=max_action,outcomes_recomputed=True))
    print('audited',replay_count,'episodes',flush=True)

def font(size):
    path=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip();return ImageFont.truetype(path,size)

def media():
    out=ROOT/'summary';out.mkdir(exist_ok=True,parents=True);entries=[];f=font(23);small=font(19)
    for task in ('cup','insert'):
        arms=['single','gradient','multiview'];records=[]
        for arm in arms:
            p=ROOT/'development_v01'/task/f'geometry_411001_{arm}'
            records.append(dict(frames=np.load(p/'FRAMES.npz')['frames'],trace=np.load(p/'TRACE.npy',allow_pickle=True),result=json.loads((p/'RESULT.json').read_text())))
        name=task+'_geometry_comparison.mp4';W,H=1344,1050
        process=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','20','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out/name)],stdin=subprocess.PIPE)
        try:
            for t in list(range(190))+[189]*30:
                frame=Image.new('RGB',(W,H),'#edf2f5');d=ImageDraw.Draw(frame)
                d.text((18,10),TASK[task]+' · 几何范围外 · 同一新物体／相同观察和动作预算',font=f,fill='#163047')
                for col,r in enumerate(records):
                    x=448*col;src=r['frames'][t];frame.paste(Image.fromarray(src[:,:448]),(x,88));frame.paste(Image.fromarray(src[:,448:]),(x,536))
                    d.text((x+10,48),LABEL[arms[col]],font=f,fill='#163047')
                    d.text((x+8,92),'斜视全景',font=small,fill='white');d.text((x+8,540),'工具相机',font=small,fill='white')
                    stage=r['trace'][t]['stage'];d.text((x+10,987),f'步 {t+1}/190  ·  '+stage,font=small,fill='#163047')
                    if t==189:d.text((x+10,1013),'最后10步：'+('成功并保持' if r['result']['success'] else '未达到成功条件'),font=small,fill='#007d68' if r['result']['success'] else '#ae3939')
                process.stdin.write(np.array(frame).tobytes())
                if t==189:frame.save(out/(task+'_cover.jpg'))
        finally:process.stdin.close()
        assert process.wait()==0
        entries.append(dict(task=task,file=name,seed=411001,split='geometry',selection='First declared development seed; no best-case selection',sha256=sha(out/name)))
    js(out/'VIDEOS.json',entries)

def failure():
    """Render recorded failure states only; never start a new rollout."""
    out=ROOT/'summary';p=ROOT/'development_v01/insert/geometry_411003_gradient'
    pre=json.loads((p/'PRE_RUN.json').read_text());trace=np.load(p/'TRACE.npy',allow_pickle=True);obs=np.load(p/'OBS.npy',allow_pickle=True)
    e=GeometrySkill(pre['cfg']);f=font(22);name='insert_retained_failure.mp4';W,H=896,550
    proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','20','-i','-','-an','-c:v','libx264','-threads','2','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out/name)],stdin=subprocess.PIPE)
    try:
        for t in list(range(190))+[189]*30:
            q=trace[t];e.data.qpos[:]=q['qpos'];e.data.qvel[:]=q['qvel'];e.mj.mj_forward(e.model,e.data)
            frame=Image.new('RGB',(W,H),'#edf2f5');d=ImageDraw.Draw(frame)
            o,rgb,mask=e.observe();assert (o['corners'] is None)==(obs[t]['corners'] is None)
            frame.paste(Image.fromarray(e.render_global()),(0,60));frame.paste(Image.fromarray(rgb),(448,60))
            d.text((12,10),'保留失败：槽口未形成可用轮廓，几何适配没有启动',font=f,fill='#9f3333')
            d.text((12,515),f'步 {t+1}/190  ·  '+q['stage']+'  ·  无力度调整',font=f,fill='#173247')
            proc.stdin.write(np.array(frame).tobytes())
            if t==189:frame.save(out/'insert_retained_failure_cover.jpg')
        proc.stdin.close();assert proc.wait()==0
    finally:e.close()
    entries=json.loads((out/'VIDEOS.json').read_text());entries.append(dict(task='insert',file=name,seed=411003,split='geometry',selection='Retained failed gradient episode; replayed sealed states',sha256=sha(out/name)));js(out/'VIDEOS.json',entries)

def page():
    out=ROOT/'summary';results=json.loads((out/'RESULTS.json').read_text());videos=json.loads((out/'VIDEOS.json').read_text())
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>P41 三维几何泛化</title><style>body{font:18px system-ui;max-width:1280px;margin:40px auto;color:#173247;background:#f2f6f8}table{border-collapse:collapse;width:100%;background:white}td,th{padding:14px;border:1px solid #d3dee5;text-align:left}video{width:100%;background:#15222b}p{line-height:1.7}a{color:#08786e}</style>',
          '<h1>P41：三维几何泛化与位置／姿态适配</h1><p>物体尺寸、部位位置和朝向变化；六自由度浮动工具／夹爪，非完整机械臂。动作只有位置、朝向及开闭，不学习力。质量、摩擦和执行增益固定。</p>',
          '<p><b>本页为小批开发结果：</b>每任务每设置4个条件、5种对照，共80回合。使用允许的部件mask和标定。几何latent更新是解析投影损失上的梯度优化，尚非DINO或元学习神经Adapter证据。</p>',
          '<table><tr><th>任务／设置</th>'+''.join('<th>'+html.escape(LABEL[k])+'</th>' for k in ['single','fixed','prefix','gradient','multiview'])+'</tr>']
    for task in ('cup','insert'):
        for split in ('id','geometry'):
            line='<tr><td>'+TASK[task]+' · '+('范围内' if split=='id' else '范围外几何')+'</td>'
            for arm in ['single','fixed','prefix','gradient','multiview']:
                c=next(c for c in results['cells'] if (c['task'],c['split'],c['arm'])==(task,split,arm));line+=f'<td>{c["successes"]}/{c["n"]}（{100*c["successes"]/c["n"]:.0f}%）</td>'
            body.append(line+'</tr>')
    body+=['</table><p>范围外只改变几何；不是执行不准确、动力不足或摩擦变化。插入包含倾斜槽轴和矩形朝向；杯子为自由刚体，最后要求真实双侧接触并保持抬升。当前不是任意网格物体泛化。</p>']
    for v in videos:
        failed='failure' in v['file'];caption='保留失败：完整开口未能被观察，控制保持等待。根据封存状态回放，没有补跑新实验。' if failed else '预定第一个新几何条件，同时展示斜视全景与工具相机。成功与失败只在执行结束后标注。'
        poster='insert_retained_failure_cover.jpg' if failed else v['task']+'_cover.jpg'
        body.append('<h2>'+TASK[v['task']]+(' · 保留失败' if failed else '')+'</h2><p>'+caption+'</p><video controls preload="metadata" poster="'+poster+'" src="'+v['file']+'"></video>')
    body.append('<p><a href="ROOT_REPORT_ZH.md">完整结论</a> · <a href="RESULTS.json">全部逐条件结果</a> · <a href="AUDIT.json">动作与物理复算</a> · <a href="../DEV_FREEZE.json">结果前冻结</a></p></html>')
    (out/'index.html').write_text('\n'.join(body))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','media','failure','page']);a=p.parse_args();globals()[a.mode]()
