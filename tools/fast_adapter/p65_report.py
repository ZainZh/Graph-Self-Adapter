#!/usr/bin/env python3
"""P65 postseal routing audit and preselected Chinese three-arm videos.

No simulator or fitting. Outcome access requires all 36 sealed results.
Original P47 success/mechanical labels are retained without reinterpretation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path

import numpy as np

from prove.fast_adapter.p61_data import sha256

ARMS=('fixed','online_local','online_history')
SPLITS=('id','size','position')
SEEDS=tuple(range(640100,640104))
PHASES=('observe','clearance','align','descend','approach','close','transport','hold')
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def read(path):return json.loads(Path(path).read_text())
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha256(path))
def need(value,message):
    if not value:raise ValueError(message)


def status(directory):
    """List completion artifacts only; do not open results."""
    root=Path(directory)
    results=sorted(root.glob('*/RESULT.json'))
    seals=sorted(root.glob('*/SEAL.json'))
    aborts=sorted(root.glob('*/ABORT.json'))+sorted(root.glob('*/PROGRAM_ERROR.json'))
    return dict(expected=36,result_files=len(results),seal_files=len(seals),aborts=len(aborts),
                complete=len(results)==len(seals)==36 and not aborts)


def key(result):
    cell=result['cell'];return cell['split'],cell['seed'],cell['arm']


def paired_prefix(fixed,local):
    """Same Code trajectory until local adaptation can change geometry."""
    stop=local['first_update_t'] if local['first_update_t'] is not None else 240
    need(fixed['result']['cell']['arm']=='fixed' and local['result']['cell']['arm']=='online_local',
         'Prefix comparison must be fixed versus local')
    count=0
    for a,b in zip(fixed['trace'][:stop],local['trace'][:stop]):
        need(a['t']==b['t'] and a['phase_before']==b['phase_before'] and a['phase_after']==b['phase_after'],
             'Fixed/local prefix routing differs before local update')
        for name in ('command_p','command_R','controls_before','controls_after'):
            need(np.allclose(a[name],b[name],atol=1e-9,rtol=0,equal_nan=False),
                 'Fixed/local Code prefix differs: '+name)
        need(a['closed']==b['closed'] and a['command_mode']==b['command_mode'],
             'Fixed/local command mode differs before local update')
        need(all(np.allclose(a['state'][k],b['state'][k],atol=1e-9,rtol=0) for k in ('integration','qpos')),
             'Fixed/local simulator prefix state differs')
        pa,pb=fixed['observations'][count],local['observations'][count]
        need(set(pa)==set(pb) and all(np.allclose(pa[k],pb[k],atol=1e-9,rtol=0) for k in pa),
             'Fixed/local public observations differ before local update')
        count+=1
    need(count==stop,'Incomplete common-prefix trace')
    images_checked=None
    if 'path' in fixed and 'path' in local:
        with np.load(Path(fixed['path'])/'PUBLIC_IMAGES.npz') as aa, np.load(Path(local['path'])/'PUBLIC_IMAGES.npz') as bb:
            need(np.array_equal(aa['t'],bb['t']),'Paired public image schedules differ')
            take=aa['t']<stop
            need(np.array_equal(aa['rgb'][take],bb['rgb'][take]) and
                 np.array_equal(aa['masks_packed'][take],bb['masks_packed'][take]),
                 'Fixed/local public visual prefix differs before local update')
            images_checked=int(take.sum())
    return dict(verified=True,steps=count,public_image_samples_checked=images_checked,
                ends_before_local_update=local['first_update_t'])


def aggregate(runs):
    lookup={key(r['result']):r for r in runs}
    expected={(split,seed,arm) for split in SPLITS for seed in SEEDS for arm in ARMS}
    need(len(runs)==36 and set(lookup)==expected,'Exact 36-cell condition/arm denominator required')
    groups={};pairs=[]
    for split in SPLITS:
        for arm in ARMS:
            selected=[lookup[split,seed,arm] for seed in SEEDS]
            groups[split+'/'+arm]=dict(n=4,
                success=sum(bool(r['result']['validated_success']) for r in selected),
                sustained_native_success=sum(bool(r['result']['sustained_native_success']) for r in selected),
                mechanical_anomalies=sum(not r['result']['mechanical_valid'] for r in selected),
                phase_entered={phase:sum(phase in r['phase_first_t'] for r in selected) for phase in PHASES},
                phase_steps={phase:sum(r['phase_steps'].get(phase,0) for r in selected) for phase in PHASES},
                terminal_phases=dict(Counter(r['terminal_phase'] for r in selected)),
                did_not_reach_close=sum('close' not in r['phase_first_t'] for r in selected),
                nn_command_steps=sum(r['nn_command_steps'] for r in selected),
                code_command_steps=sum(r['code_command_steps'] for r in selected),
                update_events=sum(r['update_events'] for r in selected),
                update_phase_violations=sum(r['update_phase_violations'] for r in selected))
        for seed in SEEDS:
            fixed=lookup[split,seed,'fixed']
            local=lookup[split,seed,'online_local']
            prefix=paired_prefix(fixed,local)
            for candidate,role in (('online_local','primary'),('online_history','diagnostic')):
                online=lookup[split,seed,candidate]
                a,b=bool(fixed['result']['validated_success']),bool(online['result']['validated_success'])
                pairs.append(dict(split=split,seed=seed,candidate=candidate,role=role,
                    fixed_success=a,online_success=b,rescue=b and not a,harm=a and not b,
                    fixed_local_common_prefix=prefix if candidate=='online_local' else None))
    comparisons={split+'/'+arm:dict(n=4,rescues=sum(p['rescue'] for p in pairs if p['split']==split and p['candidate']==arm),
                                   harms=sum(p['harm'] for p in pairs if p['split']==split and p['candidate']==arm))
                 for split in SPLITS for arm in ('online_local','online_history')}
    return groups,pairs,comparisons


def historical_reference(path):
    source=read(path)
    need(source.get('schema')=='prove.p64-cup-complete-public-runtime-audit.v1'
         and len(source['cells'])==24 and source['all_runtime_invariants_passed'],
         'Historical reference must be completed P64 report')
    return dict(source=ref(path),groups=source['groups'],pairs=source['pairs'],
                interpretation='Historical P64 same-condition reference; different routing, not pooled or a P65 randomized arm')


def caption(row,arm):
    names={'observe':'共同观察运动','clearance':'按 Code 抬手到安全高度','align':'按 Code 在杯柄外上方对齐',
           'descend':'按 Code 在杯柄外下降','approach':'由冻结动作 MLP 接近杯柄',
           'close':'按 Code 闭合真实夹指','transport':'按 Code 维持抓持并抬升','hold':'按 Code 保持抬升'}
    phase=row['phase_before']
    detail=names[phase]
    if phase=='approach' and row['command_mode']=='code':
        detail='公开几何暂不可用，按 Code 保持位置'
    record=row.get('adaptation')
    fresh=record is not None and record['t']==row['t']
    if arm=='fixed':feedback='z 始终为零；持续读取当前视觉。'
    elif arm=='online_local' and phase not in ('approach','close','transport','hold'):
        feedback='尚未进入近接近阶段：积累公开历史，z 保持为零。'
    elif phase in ('close','transport','hold'):
        feedback='已退出近接近：清除活动 z，保留冻结目标供 Code 闭合／抬升。'
    elif fresh and record['U']:
        feedback=f"利用 {record['S']} 帧因果公开轮廓完成 {record['U']} 次内更新。"
    else:feedback='本步未更新 z；保留已有值，等待有效公开反馈。'
    return detail,feedback


def render_triplet(runs,output,font=FONT):
    """Render only saved preaction RGB; outcomes appear on separate end card."""
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    from tools.fast_adapter.p64_cup_presentation import put
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    lookup={key(r['result']):r for r in runs}
    titlefont=ImageFont.truetype(font,23);textfont=ImageFont.truetype(font,18)
    arm_names={'fixed':'固定 z','online_local':'阶段内在线 z（主比较）','online_history':'全程历史在线 z（诊断）'}
    manifest=dict(schema='prove.p65-chinese-three-arm-videos.v1',physics_steps=0,
        selected_seed=640100,selected_strata=list(SPLITS),font=ref(font),videos=[],
        frame_semantics='sealed preaction RGB every4steps; separate postseal terminal text card',fps=8,
        disclosure='No success/failure shown before the terminal card')
    for split in SPLITS:
        chosen=[lookup[split,640100,arm] for arm in ARMS]
        archives=[]
        for run in chosen:
            with np.load(Path(run['path'])/'PUBLIC_IMAGES.npz',allow_pickle=False) as archive:
                archives.append(dict(t=archive['t'],rgb=archive['rgb']))
        annotations=[dict(schema='prove.episode-annotations.v1',arm=arm,segments=[],
                         source_bindings={'RESULT.json':ref(Path(r['path'])/'RESULT.json'),
                                          'TRACE.npy':ref(Path(r['path'])/'TRACE.npy'),
                                          'SEAL.json':ref(Path(r['path'])/'SEAL.json')})
                     for arm,r in zip(ARMS,chosen)]
        for arm,run,notes in zip(ARMS,chosen,annotations):
            public_trace=output/f'{split}_{arm}_PUBLIC_TRACE.jsonl'
            with public_trace.open('w') as stream:
                for row in run['trace']:
                    record=row.get('adaptation')
                    projected=dict(t=row['t'],phase_before=row['phase_before'],phase_after=row['phase_after'],
                        command_mode=row['command_mode'],adaptation=None if record is None else
                        dict(t=record['t'],U=record['U'],S=record['S'],admitted=bool(record['admitted'])))
                    stream.write(json.dumps(projected)+'\n')
            notes['source_bindings'][public_trace.name]=ref(public_trace)
            notes['derived_public_trace']=dict(path=public_trace.name,source_trace=ref(Path(run['path'])/'TRACE.npy'))
        video=output/f'{split}_640100_three_arms.mp4'
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),8,(1680,800))
        need(writer.isOpened(),'MP4 encoder unavailable')
        try:
            need(all(np.array_equal(a['t'],archives[0]['t']) for a in archives),'Video sample times differ')
            for index,t in enumerate(archives[0]['t']):
                canvas=Image.new('RGB',(1680,800),(15,22,34));draw=ImageDraw.Draw(canvas)
                draw.text((18,10),f'P65 杯柄｜{split}｜仅位置／尺寸变化，无朝向 OOD',font=titlefont,fill='white')
                for column,(arm,run,archive) in enumerate(zip(ARMS,chosen,archives)):
                    x=column*560;row=run['trace'][int(t)];purpose,feedback=caption(row,arm)
                    draw.text((x+12,52),arm_names[arm],font=titlefont,fill=(180,210,255))
                    draw.text((x+12,85),f'控制步 {int(t)}/239；实际命令：'+row['command_mode'],font=textfont,fill='white')
                    put(draw,(x+12,115),purpose,textfont,width=535)
                    put(draw,(x+12,150),feedback,textfont,width=535)
                    canvas.paste(Image.fromarray(archive['rgb'][index]).resize((548,548)),(x+6,214))
                    annotations[column]['segments'].append(dict(start_step=int(t),end_step=int(t),stage=purpose,
                        detail=feedback,sources=[dict(file=f'{split}_{arm}_PUBLIC_TRACE.jsonl',line=int(t)+1,pointer='/phase_before'),
                                                 dict(file=f'{split}_{arm}_PUBLIC_TRACE.jsonl',line=int(t)+1,pointer='/adaptation')]))
                draw.text((18,773),'封存动作前 RGB，每 4 步一帧；播放压缩，不代表实际墙钟耗时。',font=textfont,fill=(185,198,215))
                if index==0:canvas.save(output/f'{split}_poster.png')
                writer.write(cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR))
            canvas=Image.new('RGB',(1680,800),(15,22,34));draw=ImageDraw.Draw(canvas)
            draw.text((24,35),'240 步执行完毕：封存后的原 P47 标准评估（文字终态卡）',font=titlefont,fill='white')
            for column,(arm,run) in enumerate(zip(ARMS,chosen)):
                x=column*560;result=run['result']
                draw.text((x+18,115),arm_names[arm],font=titlefont,fill=(180,210,255))
                label='严格成功' if result['validated_success'] else '机械异常，判失败' if not result['mechanical_valid'] else '未完成抓取'
                put(draw,(x+18,190),'结果：'+label,titlefont,width=520)
                put(draw,(x+18,270),f"实际 NN 命令：{run['nn_command_steps']} 步；Code 命令：{run['code_command_steps']} 步",textfont,width=520)
                put(draw,(x+18,360),f"终止阶段：{run['terminal_phase']}；z 更新事件：{run['update_events']}",textfont,width=520)
            put(draw,(24,510),'这是预选的同条件三臂展示，不替代 36 条完整记录。成功标准、机械门与 240 步预算沿用冻结合同；全程异常不因视觉结果看似成功而忽略。',textfont,width=1620)
            frame=cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)
            for _ in range(24):writer.write(frame)
        finally:
            writer.release()
            archives.clear()
        notes=output/f'{split}_ANNOTATIONS.json';notes.write_text(json.dumps(annotations,ensure_ascii=False,indent=2)+'\n')
        manifest['videos'].append(dict(split=split,seed=640100,video=ref(video),annotations=ref(notes),
            sources=[dict(result=ref(Path(r['path'])/'RESULT.json'),seal=ref(Path(r['path'])/'SEAL.json')) for r in chosen]))
    (output/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


def audit_trace(trace,observations,arm):
    need(arm in ARMS and len(trace)==len(observations)==240,'P65 requires all 240 control steps')
    phase_first={};occupancy=Counter();nn=events=violations=resets=0
    anchor=previous_record=None;first_update=local_entry=None
    for t,(row,public) in enumerate(zip(trace,observations)):
        need(row['t']==public['t']==t,'Noncontiguous control/public trace')
        need(set(public)=={'t','K','camera_position','camera_rotation','position','rotation'},
             'Nonpublic or missing runtime input')
        before,after=row['phase_before'],row['phase_after']
        need(before in PHASES and after in PHASES,'Unknown routed phase')
        if t:need(before==trace[t-1]['phase_after'],'Controller phase continuity broken')
        phase_first.setdefault(before,t);phase_first.setdefault(after,t)
        occupancy[before]+=1
        route=row['neural_action']
        qualified=row['estimate'] is not None and all(np.isfinite(x).all() for x in row['estimate'])
        expected_neural=bool(t>=12 and before=='approach' and qualified)
        need(route['used']==expected_neural and route['route']==row['command_mode']
             and row['command_mode']==('neural' if expected_neural else 'code'),
             'Actual neural command escaped approach or routing log differs')
        need(route['phase_before']==before and route['phase_after']==after
             and bool(route['gripper_closed'])==bool(row['closed']), 'Command boundary metadata differs')
        need(np.array_equal(np.asarray(row['command_p']),np.asarray(route['target'])),
             'Command does not equal recorded actual target')
        if expected_neural:
            nn+=1
            need(route['neural_phase']=='approach','Network invoked with wrong stage label')
            need(all(name in route for name in ('center','size','start','anchor')),'Missing true network inputs')
        else:
            need(np.array_equal(row['command_p'],route['code_target'])
                 and np.array_equal(row['command_R'],route['code_rotation']), 'Code target was overridden')
        record=row['adaptation']
        need(record is not None,'Missing public encoder/feedback record')
        fresh=record['t']==t
        if before in ('close','transport','hold'):
            need(not fresh,'Encoder or z updated after close')
            need(previous_record is not None and record['t']==previous_record['t']
                 and np.array_equal(record['z_after'],previous_record['z_after']), 'Postclose latent changed')
            continue
        need(fresh,'Missing current-frame encoder call before closing')
        schedule=record['stage_schedule']
        need(schedule['t']==t and schedule['arm']==arm and schedule['phase_before']==before,
             'Stage-local schedule metadata differs')
        enabled=arm=='online_history' or (arm=='online_local' and before=='approach')
        need(schedule['update_enabled']==enabled and schedule['U']==record['U'], 'Update schedule differs')
        z0,z1=np.asarray(record['z_before']),np.asarray(record['z_after'])
        need(z0.shape==z1.shape==(8,) and np.isfinite(z0).all() and np.isfinite(z1).all(), 'Invalid latent8')
        need(np.array_equal(z0[3:6],np.zeros(3)) and np.array_equal(z1[3:6],np.zeros(3)), 'Orientation latent changed')
        if anchor is None:anchor=np.asarray(record['T_world_anchor'])
        need(schedule['readonly_episode_anchor'] is True and np.array_equal(record['T_world_anchor'],anchor),
             'First-EEF anchor was rebased')
        need(len(record['support_times'])==record['S'] and all(0<=x<=t for x in record['support_times']),
             'Noncausal public support')
        need(schedule['history_after']==schedule['history_before']+int(record['admitted']), 'Public history was reset or fabricated')
        if previous_record is not None:
            need(schedule['history_before']==previous_record['stage_schedule']['history_after'], 'Episode history was discarded')
        reset=bool(schedule['local_entry_reset'])
        if reset:
            need(arm=='online_local' and before=='approach' and local_entry is None and np.array_equal(z0,np.zeros(8)),
                 'Invalid local-entry reset')
            local_entry=t;resets+=1
        elif previous_record is not None:
            need(np.array_equal(z0,previous_record['z_after']), 'Persistent latent continuity broken')
        else:need(np.array_equal(z0,np.zeros(8)),'Episode latent not reset')
        if arm=='online_local' and before=='approach':need(local_entry is not None,'Local phase entered without zero reset')
        if not enabled:need(np.array_equal(z1,np.zeros(8)) and not record['U'], 'Online z outside allowed phase')
        if record['U']:
            violations+=int(not enabled)
            need(record['U']==3 and record['admitted'] and record['S']>=3,'Wrong inner budget or missing public feedback')
            events+=1
            if first_update is None:first_update=t
        else:need(np.array_equal(z0,z1),'Latent changed without feedback update')
        previous_record=record
    return dict(phase_first_t=phase_first,phase_steps=dict(occupancy),terminal_phase=trace[-1]['phase_after'],
                nn_command_steps=nn,code_command_steps=240-nn,update_events=events,
                update_phase_violations=violations,first_update_t=first_update,
                local_entry_t=local_entry,local_entry_resets=resets,
                invariants=dict(actual_routing_verified=True,causal_public_support=True,readonly_anchor=True,
                                orientation_latent_zero=True,update_stage_contract=True,public_inputs_only=True))



def audit_lifecycle(trace,lifecycle):
    need(len(lifecycle)==1,'Exactly one critical-stage/episode latent exit required')
    event=lifecycle[0]
    need(np.asarray(event['z_before']).shape==(8,) and np.isfinite(event['z_before']).all()
         and np.array_equal(event['z_after'],np.zeros(8))
         and event['cached_estimate_retained'] is True and event['archived_prediction_unchanged'] is True,
         'Live latent was not cleared while preserving archived prediction')
    boundaries=[r for r in trace if r['phase_before']=='approach' and r['phase_after']=='close']
    if boundaries:
        need(len(boundaries)==1 and event['reason']=='approach_to_close' and event['t']==boundaries[0]['t'],
             'Wrong critical-stage exit boundary')
        row=boundaries[0]
        need(row['latent_exit'] is not None and row['latent_exit']['t']==event['t']
             and np.array_equal(row['latent_exit']['z_after'],event['z_after'])
             and np.array_equal(row['adaptation']['z_after'],event['z_before']),
             'Exit receipt differs from last archived prediction')
    else:
        need(event['reason']=='episode_budget_end' and event['t']==240,'Missing H240 latent cleanup')
        need(np.array_equal(event['z_before'],trace[-1]['adaptation']['z_after']),
             'Budget-exit latent differs from final archived prediction')
    return True


def load_cell(path,freeze,freeze_path):
    path=Path(path);result=read(path/'RESULT.json');seal=read(path/'SEAL.json')
    need(not (path/'ABORT.json').exists() and not (path/'PROGRAM_ERROR.json').exists(),
         'Aborted cell cannot be reported as completed')
    for name,digest in seal['files'].items():
        target=(path/name).resolve()
        need(target.is_relative_to(path.resolve()) and sha256(target)==digest,'Changed sealed artifact: '+name)
    required=('TRACE.npy','OBS.npy','TERMINAL.npy','PRE_RUN.json','PUBLIC_IMAGES.npz','LATENT_LIFECYCLE.npy')
    need(all(name in seal['files'] for name in required),'Incomplete sealed trace')
    pre=read(path/'PRE_RUN.json')
    need(pre['freeze_sha256']==sha256(freeze_path) and pre['runtime_truth'] is False and pre['steps']==240,
         'Pre-run physical scope differs')
    need(pre['cell']==result['cell']==freeze['cells'][pre['index']], 'Result/freeze identity differs')
    need(seal['weights_before']==seal['weights_after'] and result['weights_unchanged'],'Frozen network weights changed')
    trace=np.load(path/'TRACE.npy',allow_pickle=True).tolist()
    observations=np.load(path/'OBS.npy',allow_pickle=True).tolist()
    stats=audit_trace(trace,observations,result['cell']['arm'])
    lifecycle=np.load(path/'LATENT_LIFECYCLE.npy',allow_pickle=True).tolist()
    audit_lifecycle(trace,lifecycle)
    need(result['latent_end_reset'] is True and result['local_entry_resets']==stats['local_entry_resets']
         and result['local_updates_outside_approach']==0 and result['code_action_steps']==stats['code_command_steps']
         and result['final_phase']==stats['terminal_phase'], 'Latent lifecycle/routing result fields differ')
    need(result['neural_action_steps']==stats['nn_command_steps'] and result['update_events']==stats['update_events'],
         'Recorded action/update totals differ')
    return dict(path=str(path.resolve()),result=result,trace=trace,observations=observations,
                result_binding=ref(path/'RESULT.json'),seal_binding=ref(path/'SEAL.json'),
                weights_unchanged=True,**stats)



def audit_births(runs):
    lookup={key(r['result']):r for r in runs}
    receipts=[]
    for split in SPLITS:
        for seed in SEEDS:
            first=lookup[split,seed,'fixed']
            for arm in ('online_local','online_history'):
                other=lookup[split,seed,arm]
                need(all(np.allclose(first['trace'][0]['state'][k],other['trace'][0]['state'][k],atol=1e-9,rtol=0)
                         for k in ('integration','qpos')),'Initial simulator state differs across P65 arms')
                a,b=first['observations'][0],other['observations'][0]
                need(set(a)==set(b) and all(np.allclose(a[k],b[k],atol=1e-9,rtol=0) for k in a),
                     'Initial public observations differ across arms')
            receipts.append(dict(split=split,seed=seed,three_arm_same_initial_state=True))
    return receipts


def replay_neural_commands(runs,checkpoint):
    """Independent CPU batch forward from saved true network inputs; no physics."""
    import torch
    from prove.fast_adapter.p64_cup_policy import CupActionPolicy,PHASES as MODEL_PHASES
    package=torch.load(checkpoint,map_location='cpu',weights_only=True)
    need(package.get('schema')=='prove.p64-cup-action.v1','Unexpected frozen action checkpoint')
    model=CupActionPolicy();model.load_state_dict(package['state_dict'],strict=True)
    model.eval().requires_grad_(False)
    receipts=[]
    for run in runs:
        selected=[r for r in run['trace'] if r['neural_action']['used']]
        if not selected:
            receipts.append(dict(cell=run['result']['cell'],n=0,max_error_m=None,passed=True,
                                 interpretation='no neural command occurred; not positive action-model evidence'))
            continue
        inputs=[torch.tensor(np.stack([r['neural_action'][name] for r in selected]),dtype=torch.float32)
                for name in ('center','size','start','anchor')]
        phase=torch.full((len(selected),),MODEL_PHASES.index('approach'),dtype=torch.long)
        with torch.no_grad():predicted=model(*inputs,phase).numpy()
        error=np.linalg.norm(predicted-np.stack([r['command_p'] for r in selected]),axis=-1)
        need(np.isfinite(error).all() and error.max()<=1e-6,'CPU checkpoint replay differs from actual neural command')
        receipts.append(dict(cell=run['result']['cell'],n=len(selected),max_error_m=float(error.max()),
                             mean_error_m=float(error.mean()),passed=True))
    return dict(checkpoint=ref(checkpoint),method='independent CPU batch forward using recorded inputs',
                tolerance_m=1e-6,physics_steps=0,optimizer_updates=0,cells=receipts)


def report(freeze_path,directory,output,p64_report,*,videos=False,font=FONT):
    receipt=status(directory)
    need(receipt['complete'],'Require all 36 sealed cells and zero aborts before opening any result')
    freeze=read(freeze_path)
    need(freeze.get('schema')=='prove.p65-cup-local-routing-freeze.v1'
         and freeze.get('status')=='frozen' and freeze.get('steps')==240 and len(freeze['cells'])==36,
         'Wrong frozen P65 budget')
    for path,digest in freeze['files'].items():need(sha256(path)==digest,'Frozen source/model changed: '+path)
    runs=[load_cell(p.parent,freeze,freeze_path) for p in sorted(Path(directory).glob('*/RESULT.json'))]
    groups,pairs,comparisons=aggregate(runs)
    command_replay=replay_neural_commands(runs,freeze['action_policy'])
    births=audit_births(runs)
    history=historical_reference(p64_report)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    value=dict(schema='prove.p65-stage-local-cup-complete-audit.v1',complete=True,freeze=ref(freeze_path),report_source=ref(__file__),
        groups=groups,pairs=pairs,comparisons=comparisons,historical_p64_reference=history,
        all_runtime_invariants_passed=True,independent_neural_command_replay=command_replay,
        initial_state_pairing=births,
        definitions=dict(primary='online_local versus fixed with same P65 routing',
            diagnostic='online_history versus fixed; original preclose update timing',
            phase_entry='first trace step where phase_before or phase_after names the stage',
            phase_occupancy='actual command steps by phase_before',
            stalls='descriptive terminal phase and no close by H240; no new threshold or stopping gate',
            success='unchanged frozen P47 validated_success/native/mechanical_valid',
            denominator='all four predeclared conditions per stratum per arm, including no-neural-entry and mechanical failures'),
        cells=[{k:v for k,v in r.items() if k not in ('trace','observations')} for r in runs])
    if videos:value['videos']=render_triplet(runs,output/'videos',font)
    (output/'RESULTS.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    write_index(value,output)
    return value



def write_index(value,output):
    output=Path(output)
    labels={'fixed':'固定 z','online_local':'阶段内在线（主比较）','online_history':'闭合前持续更新（诊断）'}
    split_names={'id':'ID','size':'仅尺寸变化','position':'仅位置变化'}
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>P65 杯柄阶段路由</title>',
          '<style>body{font:16px system-ui;max-width:1250px;margin:36px auto;padding:0 18px;line-height:1.6;color:#182637}table{border-collapse:collapse;width:100%;margin:18px 0}th,td{padding:8px;border:1px solid #ddd;text-align:left}a{color:#165cab}</style>',
          '<h1>P65：杯柄 Code／神经阶段接口验证</h1>',
          '<p>已开放开发条件，每组四条件，三臂共 36 条。沿用 P64 冻结权重，改变阶段路由；无朝向 OOD，不是 fresh held-out。简单阶段由 Code 执行，动作 MLP 只在 approach 调用。</p>',
          '<p>主比较为阶段内 online_local 对固定 z。online_history 保留闭合前的历史更新，仅作时序诊断，不能称为仅阶段内适配。</p>',
          '<table><tr><th>分组</th><th>模式</th><th>严格成功</th><th>native 持续成功</th><th>机械异常</th><th>进入 approach／close</th><th>H240 终止阶段</th><th>实际神经步数</th></tr>']
    for split in SPLITS:
        for arm in ARMS:
            g=value['groups'][split+'/'+arm]
            terminal='、'.join(f'{k}: {v}' for k,v in sorted(g['terminal_phases'].items()))
            page.append(f'<tr><td>{split_names[split]}</td><td>{labels[arm]}</td><td>{g["success"]}/4</td>'
                        f'<td>{g["sustained_native_success"]}/4</td><td>{g["mechanical_anomalies"]}/4</td>'
                        f'<td>{g["phase_entered"]["approach"]}/4 ／ {g["phase_entered"]["close"]}/4</td>'
                        f'<td>{html.escape(terminal)}</td><td>{g["nn_command_steps"]}</td></tr>')
    page+=['</table><p>未进入神经阶段、未到达 close、真实任务失败和机械异常均保留在原分母。终止阶段与未达 close 是描述性停滞记录，不增加停止门或重定义成功。</p>',
           '<table><tr><th>分组</th><th>主比较：阶段内更新救回／误伤</th><th>诊断：历史更新救回／误伤</th></tr>']
    for split in SPLITS:
        local=value['comparisons'][split+'/online_local'];history=value['comparisons'][split+'/online_history']
        page.append(f'<tr><td>{split_names[split]}</td><td>{local["rescues"]}／{local["harms"]}（4 对）</td>'
                    f'<td>{history["rescues"]}／{history["harms"]}（4 对）</td></tr>')
    page+=['</table><p>审计覆盖公开输入、因果支持、只读首次 EEF 锚、更新阶段、退出清零、冻结权重，以及固定／局部在线首次更新前共同状态和视觉。保存的真实网络输入另经 CPU 批量前向重算，与实际命令核对。</p>','<h2>预选三臂视频</h2><ul>']
    for video in value.get('videos',{}).get('videos',[]):
        href=html.escape(os.path.relpath(video['video']['path'],output),quote=True)
        page.append(f'<li><a href="{href}">{split_names[video["split"]]} · seed 640100 · 三臂对比</a></li>')
    page+=['</ul><p>仅保存的动作前 RGB；中文解释动作目的、实际执行器和更新时刻，结果在独立终态文字卡才出现。视频不替代全部 36 条统计。</p>',
           '<p>P64 旧路由仅作为单独历史参照，不与 P65 合并，也不把路由改变的收益全部归给 Adapter。</p>',
           '<p><a href="RESULTS.json">完整计数、历史参照、源码／封存绑定与审计凭据（JSON）</a></p></html>']
    (output/'index.html').write_text('\n'.join(page)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells',type=Path,required=True)
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--p64-report',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--status-only',action='store_true')
    parser.add_argument('--videos',action='store_true')
    parser.add_argument('--font',default=FONT)
    args=parser.parse_args()
    if args.status_only:print(json.dumps(status(args.cells)));return
    if any(x is None for x in (args.freeze,args.p64_report,args.output)):
        parser.error('--freeze, --p64-report and --output required for outcome opening')
    value=report(args.freeze,args.cells,args.output,args.p64_report,videos=args.videos,font=args.font)
    print(json.dumps(value['comparisons'],indent=2))


if __name__=='__main__':main()
