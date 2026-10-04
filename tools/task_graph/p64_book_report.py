#!/usr/bin/env python3
"""Audit completed P64 book cells and render preselected Chinese videos.

Reads sealed RGB only. No simulator, model fitting, outcome-based selection or
new physics. The two display pairs are fixed: ID source62 and size source66.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from tools.task_graph.p64_book_evaluate import (
    ARMS, PUBLIC_KEYS, cells, conditions, dump, read, ref, sha, verify)

VIDEO_PAIRS = (('id_source62', 0, 1), ('size_source66', 8, 9))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def status(results):
    root = Path(results)
    values = []
    for cell in cells():
        path = root/f"cell_{cell['index']:03d}"
        values.append(dict(index=cell['index'], result=(path/'RESULT.json').is_file(),
                           seal=(path/'SEAL.json').is_file(), abort=(path/'ABORT.json').is_file()))
    return dict(expected=24, completed=sum(x['result'] and x['seal'] for x in values),
                aborted=sum(x['abort'] for x in values), cells=values)


def _need(value, message):
    if not value:
        raise ValueError(message)


def audit_prediction(prediction, arm, previous=None):
    t = prediction['t']
    telemetry = prediction['telemetry']
    action = np.asarray(prediction['action'], float)
    _need(action.shape == (7,) and np.isfinite(action).all() and action[6] == 1,
          'Invalid sealed ordinary action')
    _need(telemetry['t'] == t and telemetry['arm'] == ('continuous' if arm == 'online' else 'fixed'),
          'Runtime arm/time mismatch')
    _need(telemetry['orientation_update'] is False, 'Unexpected orientation adaptation')
    z0, z1 = (np.asarray(telemetry[key], float) for key in ('z_before', 'z_after'))
    _need(z0.shape == z1.shape == (9,) and np.isfinite(z0).all() and np.isfinite(z1).all(),
          'Invalid persistent z9')
    if previous is None:
        _need(t == 0 and np.array_equal(z0, np.zeros(9)), 'Latent not reset at entry')
    else:
        _need(t == previous['t']+1 and np.allclose(z0, previous['telemetry']['z_after'], atol=1e-8, rtol=0),
              'Latent or time is not continuous')
    support = telemetry['support_times']
    _need(len(support) == telemetry['support_measurements'], 'Support count mismatch')
    _need(all(type(x) is int and max(0,t-7) <= x <= t for x in support), 'Noncausal or expired support')
    geometry=np.asarray(telemetry['geometry9'],float)
    _need(geometry.shape==(9,) and np.isfinite(geometry).all(), 'Invalid public geometry9 prediction')
    updates = telemetry['updates']
    if updates:
        _need(arm == 'online' and t >= 1 and telemetry['admitted_measurements'] > 0 and len(updates) == 3,
              'Update without new public feedback or outside three-step budget')
    else:
        _need(np.array_equal(z0,z1), 'Latent changed without observed update')
    if arm == 'fixed':
        _need(np.array_equal(z0,np.zeros(9)) and np.array_equal(z1,np.zeros(9)), 'Fixed arm adapted')
    _need(np.allclose(action, telemetry['action'], atol=1e-8, rtol=0), 'Proposal/telemetry action mismatch')
    return len(updates)


def audit_cell(root, cell, binding, binding_path):
    root = Path(root)
    result, seal = read(root/'RESULT.json'), read(root/'SEAL.json')
    _need(not (root/'ABORT.json').exists(), 'Aborted cell cannot be treated as complete')
    _need(result['cell'] == cell and read(root/'CELL.json') == cell, 'Cell identity changed')
    condition = conditions()[cell['condition']]
    _need(result['condition'] == condition, 'Condition or position intervention changed')
    _need(read(root/'BINDING.json') == binding, 'Cell binding differs')
    _need(result['seal_sha256'] == sha(root/'SEAL.json') and seal['binding_sha256'] == sha(binding_path),
          'Broken result/seal/binding chain')
    for relative, expected in seal['files'].items():
        path = (root/relative).resolve()
        _need(path.is_relative_to(root.resolve()) and path.is_file() and sha(path) == expected,
              'Sealed artifact changed: '+relative)
    for name in ('CELL.json', 'BINDING.json', 'EXECUTION.json'):
        _need(name in seal['files'], 'Unsealed cell identity')
    steps = result['executed_physical_steps']
    _need(type(steps) is int and 0 <= steps <= 24 and seal['executed_physical_steps'] == steps,
          'H24 budget differs')
    entered = bool(result['near_entry_qualified'])
    original_blocked = condition['source_index'] in (65,70)
    if original_blocked:
        _need(not entered and result['reason'] == 'original_entry_failed', 'Historical failed entry was replaced')
    if not entered:
        _need(not result['system_success'] and steps == 0 and not (root/'public/PREDICTIONS.jsonl').exists(),
              'Blocked entry executed or vanished from denominator')
        return dict(index=cell['index'], arm=cell['arm'], condition=condition,
                    entered=False, strict=False, native=False, steps=0, updates=0,
                    original_blocked=original_blocked, reason=result['reason'],
                    weights='not instantiated', result=ref(root/'RESULT.json'), seal=ref(root/'SEAL.json'))
    for name in ('public/PREDICTIONS.jsonl','public/EXECUTED.jsonl','public/WEIGHT_AUDIT.json',
                 'private/JUDGMENTS.jsonl','private/ENTRY_GATE.json','private/INITIAL_STATE.npz'):
        _need(name in seal['files'], 'Missing sealed runtime/evaluator evidence: '+name)
    _need(read(root/'private/ENTRY_GATE.json')['near_ready'] is True, 'Entry gate differs')
    weights = read(root/'public/WEIGHT_AUDIT.json')
    _need(weights['unchanged'] is True and weights['encoder_before'] == weights['encoder_after']
          and weights['policy_before'] == weights['policy_after'], 'Frozen network state changed')
    predictions, executed, judgments = (rows(root/name) for name in
        ('public/PREDICTIONS.jsonl','public/EXECUTED.jsonl','private/JUDGMENTS.jsonl'))
    _need(len(predictions) == len(executed) == len(judgments) == steps+1, 'Frame/action denominator differs')
    updates, previous = 0, None
    from tools.task_graph.p52_full_task import InsertJudge
    judge, hold, native = InsertJudge(), 0, False
    for t, (prediction, execution, private) in enumerate(zip(predictions,executed,judgments)):
        _need(prediction['t'] == execution['t'] == private['t'] == t, 'Noncontiguous trace')
        updates += audit_prediction(prediction, cell['arm'], previous)
        previous = prediction
        name = f'public/frame_{t:04d}.npz'
        _need(name in seal['files'] and prediction['observation_sha256'] == sha(root/name), 'Unsealed prediction observation')
        with np.load(root/name, allow_pickle=False) as frame:
            _need(set(frame.files) == set(PUBLIC_KEYS), 'Runtime frame contains nonpublic/missing keys')
            _need(frame['rgb'].shape == (2,448,448,3) and frame['visible_masks'].shape == (2,3,448,448),
                  'Public visual resolution/roles changed')
        replay = judge.observe(private['official'], private['fingers'], private['wrong_target'])
        _need(all(private[key] == value for key,value in replay.items()), 'Original InsertJudge mismatch')
        hold = hold+1 if private['official'] else 0
        native = native or hold >= 3
        stop = replay['stop'] or ('micro_budget24' if t >= 24 else None)
        _need(execution['executed'] == (stop is None), 'Executed after evaluator stop')
        if execution['executed']:
            _need(np.array_equal(np.asarray(execution['action']),np.asarray(prediction['action'])),
                  'Executed action differs from sealed proposal')
        else:
            _need(t == steps and execution['action'] is None, 'Premature unexecuted action')
    _need(bool(result['system_success']) == bool(replay['complete']) and result['native_hold3'] == native,
          'Success result differs from sealed judgments')
    _need(result['reason'] == stop, 'Terminal reason differs')
    return dict(index=cell['index'], arm=cell['arm'], condition=condition, entered=True,
                strict=bool(result['system_success']), native=native, steps=steps, updates=updates,
                original_blocked=False, reason=result['reason'], weights=weights,
                initial_state=ref(root/'private/INITIAL_STATE.npz'), result=ref(root/'RESULT.json'), seal=ref(root/'SEAL.json'))


def aggregate(records):
    _need(len(records) == 24 and {r['index'] for r in records} == set(range(24)), 'Full 24-cell denominator required')
    groups = {}
    for group in ('id','size','position_shift'):
        selected = [r for r in records if r['condition']['group'] == group]
        arms = {}
        for arm in ARMS:
            values = [r for r in selected if r['arm'] == arm]
            _need(len(values) == 4, 'Four conditions per stratum/arm required')
            entered = [r for r in values if r['entered']]
            arms[arm] = dict(full_denominator=4, entered_denominator=len(entered),
                             strict_success=sum(r['strict'] for r in values),
                             native_success=sum(r['native'] for r in values),
                             strict_success_entered=sum(r['strict'] for r in entered),
                             native_success_entered=sum(r['native'] for r in entered),
                             entry_failures=4-len(entered), original_entry_failures=sum(r['original_blocked'] for r in values),
                             executed_actions=sum(r['steps'] for r in values), inner_updates=sum(r['updates'] for r in values),
                             wrong_target=sum(r['reason']=='wrong_target' for r in values),
                             contact_loss=sum(r['reason']=='book_contact_lost_5' for r in values))
        paired=[]
        for condition in (x for x in conditions() if x['group']==group):
            fixed=next(r for r in selected if r['condition']['id']==condition['id'] and r['arm']=='fixed')
            online=next(r for r in selected if r['condition']['id']==condition['id'] and r['arm']=='online')
            _need(fixed['entered']==online['entered'], 'Paired entry qualification differs')
            paired.append(dict(condition=condition['id'], entered=fixed['entered'],
                               rescue=not fixed['strict'] and online['strict'],
                               harm=fixed['strict'] and not online['strict'],
                               native_rescue=not fixed['native'] and online['native'],
                               native_harm=fixed['native'] and not online['native']))
        groups[group]=dict(arms=arms, paired_denominator=4, jointly_entered=sum(p['entered'] for p in paired),
                           rescues=sum(p['rescue'] for p in paired), harms=sum(p['harm'] for p in paired),
                           native_rescues=sum(p['native_rescue'] for p in paired),
                           native_harms=sum(p['native_harm'] for p in paired), pairs=paired)
    return groups


def audit(binding_path, results):
    receipt=status(results)
    _need(receipt['completed']==24 and receipt['aborted']==0, 'Wait for all 24 sealed results; no partial outcome opening')
    binding=read(binding_path)
    verify(binding)
    records=[audit_cell(Path(results)/f"cell_{cell['index']:03d}",cell,binding,binding_path) for cell in cells()]
    for index in range(0,24,2):
        first,second=records[index:index+2]
        if first['entered'] and second['entered']:
            with np.load(first['initial_state']['path']) as a,np.load(second['initial_state']['path']) as b:
                _need(set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files),
                      'Matched arms did not begin from identical supervisor state')
    return dict(schema='prove.p64-book-audit.v1', binding=ref(binding_path), results=str(Path(results).resolve()),
                complete=True, groups=aggregate(records), records=records,
                audit=dict(seal_chain=True, public_inputs_allowlisted=True, causal_support=True,
                           fixed_z_zero=True, frozen_weights_receipts=True, insert_judge_replayed=True,
                           same_initial_state=True, no_orientation_updates=True),
                interpretation=dict(scope='opened development states; local s9 H24 only',
                    position='caddy-only ±4 mm entry-position robustness; not certified position OOD',
                    size='original sealed iso0.90 and thickness0.78 assets; no held-book rescaling',
                    blocked='source65 and70 kept in every relevant denominator',
                    causal_sealing='each prediction written before privileged judge call in bound evaluator; terminal seal after trace',
                    mechanical='contact-loss/wrong-target counted; no separate force or joint-limit detector',
                    method='fixed and continuous use same new P64 encoder and frozen P57 action policy'))


def write_report(report, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dump(output/'AUDIT.json',report)
    lines=['<!doctype html><meta charset="utf-8"><title>P64 插书局部验证</title>',
           '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;line-height:1.6}table{border-collapse:collapse}th,td{padding:10px;border:1px solid #ddd}</style>',
           '<h1>P64 插书：位置／尺寸局部适配验证</h1>',
           '<p>已开放开发状态；真实 LIBERO 近入口 s9，H24。位置组为收纳盒 ±4 mm 平移的入口鲁棒性，不认证为位置 OOD。历史入口失败保留；不代表完整任务成功。</p>',
           '<table><tr><th>组</th><th>模式</th><th>严格成功／全组</th><th>严格成功／进入</th><th>native／全组</th><th>入口失败</th><th>救回／误伤</th></tr>']
    for group,value in report['groups'].items():
        for arm,v in value['arms'].items():
            lines.append(f'<tr><td>{html.escape(group)}</td><td>{arm}</td><td>{v["strict_success"]}/4</td>'
                         f'<td>{v["strict_success_entered"]}/{v["entered_denominator"]}</td><td>{v["native_success"]}/4</td>'
                         f'<td>{v["entry_failures"]}</td><td>{value["rescues"]}/{value["harms"]}</td></tr>')
    lines+=['</table><p>严格成功复用原 InsertJudge：native 目标与持书接触同时保持三帧。审计重放原规则，核查公开输入、历史窗口、持续 z、动作、封存链及网络权重不变凭据。</p>',
            '<p>封存证据不包含独立关节限位／力异常检测；接触丢失与错误目标另列于 JSON，不将没有检测器解释为零机械异常。</p>']
    (output/'report.html').write_text('\n'.join(lines))
    return output


def annotation(prediction, result, terminal):
    t=prediction['t'];telemetry=prediction['telemetry']
    fixed=telemetry['arm']=='fixed'
    if fixed:
        detail='每帧重读图像；冻结动作网络使用当前几何估计，z 保持为零。'
    elif telemetry['updates']:
        detail=f"本帧公开轮廓通过资格，使用 {telemetry['support_measurements']} 条因果历史观测更新位置／尺寸 z；朝向不更新。"
    else:
        detail='当前没有触发新的轮廓更新；保留此前 z，由冻结动作网络给出短插入动作。'
    if terminal:
        detail += ' 本臂结束：'+('局部严格成功。' if result['system_success'] else '局部严格未成功。')
    return dict(start_step=t,end_step=t,stage='短 Insert / '+('固定 z' if fixed else '持续 z'),detail=detail,
                sources=[dict(file='public/PREDICTIONS.jsonl',line=t+1,pointer='/telemetry')]+
                        ([dict(file='RESULT.json',pointer='/system_success')] if terminal else []))


def render_videos(binding_path, results, output, font_path=None):
    report=audit(binding_path,results)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    from PIL import Image,ImageDraw,ImageFont
    import imageio.v2 as imageio
    candidates=[Path(font_path)] if font_path else [Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
                                                   Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')]
    font_file=next((p for p in candidates if p.is_file()),None)
    if font_file is None:raise FileNotFoundError('Chinese font required')
    font=ImageFont.truetype(str(font_file),23)
    manifest=dict(schema='prove.p64-book-display.v1',selection='preselected before outcomes: ID source62 and size source66',
                  playback=dict(control_hz=20, video_fps=10, speed=.5, terminal_hold_seconds=2),
                  binding=ref(binding_path),physics_steps_executed=0,font=ref(font_file),pairs=[])
    for name,*indices in VIDEO_PAIRS:
        episodes=[]
        for index in indices:
            root=Path(results)/f'cell_{index:03d}'
            result=read(root/'RESULT.json')
            trace=rows(root/'public/PREDICTIONS.jsonl') if (root/'public/PREDICTIONS.jsonl').exists() else []
            notes=[annotation(p,result,i==len(trace)-1) for i,p in enumerate(trace)]
            episode=dict(root=root,result=result,trace=trace,notes=notes)
            episodes.append(episode)
        annotations=dict(schema='prove.p64-book-paired-annotations.v1',pairs=[dict(
            schema='prove.episode-annotations.v1', arm=ep['result']['cell']['arm'],segments=ep['notes'],
            source_bindings={name:ref(ep['root']/name) for name in ('RESULT.json','SEAL.json')}
            | ({'public/PREDICTIONS.jsonl':ref(ep['root']/'public/PREDICTIONS.jsonl')} if ep['trace'] else {}))
            for ep in episodes])
        folder=output/name;folder.mkdir();dump(folder/'ANNOTATIONS.json',annotations)
        for notes in annotations['pairs']:
            dump(folder/('ANNOTATIONS_'+notes['arm']+'.json'),notes)
        frames=max(1,max(len(ep['trace']) for ep in episodes))
        with imageio.get_writer(folder/'comparison.mp4',fps=10,codec='libx264',quality=8,macro_block_size=16) as writer:
            for displayed in range(frames+20):
                picture=Image.new('RGB',(1792,672),(246,248,250))
                draw=ImageDraw.Draw(picture)
                for column,ep in enumerate(episodes):
                    x=896*column
                    arm='固定 z' if ep['result']['cell']['arm']=='fixed' else '持续 z'
                    draw.text((x+16,10),name+'｜'+arm+'｜半速回放',font=font,fill=(20,30,45))
                    if ep['trace']:
                        t=min(displayed,len(ep['trace'])-1)
                        with np.load(ep['root']/'public'/f'frame_{t:04d}.npz') as frame:rgb=frame['rgb']
                        picture.paste(Image.fromarray(np.concatenate(rgb,axis=1)),(x,50))
                        note=ep['notes'][t]['detail']
                        draw.text((x+16,510),f'动作步 {t} / {ep["result"]["executed_physical_steps"]}；左：外部视角，右：腕部视角',font=font,fill=(20,30,45))
                    else:
                        note='原入口未通过资格，未执行短技能；该条件仍保留在完整分母。'
                    # Fixed character wrapping keeps Chinese text in its arm panel.
                    for line,start in enumerate(range(0,len(note),34)):
                        draw.text((x+16,548+line*30),note[start:start+34],font=font,fill=(20,30,45))
                if displayed==0:picture.save(folder/'poster.png')
                writer.append_data(np.asarray(picture))
        manifest['pairs'].append(dict(name=name,indices=indices,outputs={p.name:ref(p) for p in folder.iterdir()},
                                      sources=[dict(result=ref(ep['root']/'RESULT.json'),seal=ref(ep['root']/'SEAL.json')) for ep in episodes]))
    dump(output/'MANIFEST.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding',type=Path)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--status-only',action='store_true')
    parser.add_argument('--videos',action='store_true')
    parser.add_argument('--font',type=Path)
    args=parser.parse_args()
    if args.status_only: print(json.dumps(status(args.results)));return
    if args.binding is None or args.output is None:parser.error('--binding and --output required')
    if args.videos:render_videos(args.binding,args.results,args.output,args.font)
    else:write_report(audit(args.binding,args.results),args.output)


if __name__=='__main__':main()
