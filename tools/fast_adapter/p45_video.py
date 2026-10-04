"""Read-only P45 video derivatives from sealed raw frames and action traces.

No environment/model imports, rollouts, re-rendering, or source-artifact writes.
The frame at index i precedes action i, so annotations use only actions < i.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
TASKS = {'cup': '杯柄抓取', 'insert': '已持物插入', 'slider': '抓柄拉开滑块'}
ARMS = {'fixed': '固定 z', 'continuous': '在线更新 z', 'prefix3': '前三次更新后固定',
        'mismatched': '错配反馈 z', 'multiview': '普通多视图拟合'}
REPS = {'geometry': '可见几何', 'absolute_fusion': '绝对视觉融合', 'relative_fusion': '角色相对融合'}
STAGES = {
    'shared observation motion': '共同观察',
    'raise hand to shared clearance': '抬升避障',
    'align above handle outside approach': '在把手上方对齐',
    'descend outside handle': '在把手外侧下降',
    'align outside handle': '对齐把手外侧',
    'align outside cup handle': '对齐杯柄外侧',
    'approach handle with open fingers': '张开手指接近把手',
    'close physical fingers': '闭合真实手指',
    'close physical Panda fingers': '闭合真实手指',
    'lift with finger contacts': '抬杯并保持接触',
    'pull held handle': '抓住把手后拉动',
    'align held peg': '对齐已持物块',
    'approach slot': '接近槽口',
    'insert along estimated axis': '沿估计轴线插入',
    'hold insertion': '保持插入',
    'valid target geometry unavailable: hold': '尚无有效几何估计，保持位置',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def source_path(episode, relative, digest, extra=()):
    candidates = [episode / 'source' / relative, episode.parent / 'source' / relative,
                  *(Path(folder) / relative for folder in extra), ROOT / relative]
    for candidate in candidates:
        if candidate.is_file() and sha(candidate) == digest:
            return candidate
    raise ValueError(f'Cannot verify the frozen source bytes: {relative}')


def _strings(node):
    return {item.value for item in ast.walk(node) if isinstance(item, ast.Constant) and isinstance(item.value, str)}


def verify_source_order(path, neural):
    """Check the hash-bound runner's data→seal→evaluation→result code order.

    This establishes the declared runner contract, not a cryptographic wall
    clock proof. Transfer mtimes are deliberately not used as timing evidence.
    """
    tree = ast.parse(Path(path).read_text())
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == ('rollout' if neural else 'run')), None)
    if function is None:
        raise ValueError('Expected the frozen P45 run/rollout entry point')
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    def line_for(name):
        return min((node.lineno for node in calls if name in _strings(node)), default=None)
    # The smallest call containing the constant is the actual save/js call;
    # surrounding try/with statements are not calls and cannot spoof ordering.
    lines = {name: line_for(name) for name in ['TRACE.npy', 'FRAMES.npz', 'SEAL.json', 'RESULT.json']}
    evaluations = [node.lineno for node in calls if
                   (isinstance(node.func, ast.Name) and node.func.id == 'evaluate_sealed') or
                   (isinstance(node.func, ast.Attribute) and node.func.attr == 'facts')]
    seal = lines['SEAL.json']
    evaluation = min((line for line in evaluations if seal is not None and line > seal), default=None)
    if any(value is None for value in lines.values()) or evaluation is None or any(line < seal for line in evaluations) or not (
        lines['TRACE.npy'] < seal and lines['FRAMES.npz'] < seal < evaluation < lines['RESULT.json']
    ):
        raise ValueError('Frozen runner does not establish frame/trace→seal→evaluation→result order')
    return dict(runner=str(path), sha256=sha(path), source_lines=lines, evaluation_line=evaluation,
                contract='hash-bound runner writes frame/trace seal before evaluation and result',
                wall_clock_order_cryptographically_proven=False)


@dataclass
class Episode:
    path: Path
    pre: dict
    result: dict
    trace: list
    frames: np.ndarray
    terminal: dict
    neural: bool
    task: str
    seed: int
    split: str
    arm: str
    rep: str
    dt: float
    verified: dict
    terminal_frame: bool

    @classmethod
    def load(cls, path, view='recorded', reconstruction=None):
        path = Path(path).resolve()
        pre, seal, result = [read_json(path / name) for name in ['PRE_RUN.json', 'SEAL.json', 'RESULT.json']]
        neural = seal.get('schema') == 'prove.p45-trajectory-seal.v1'
        if 'schema' in seal and not neural:
            raise ValueError('Unsupported trajectory seal schema')
        files = seal['files'] if neural else seal
        required = ['TRACE.npy', 'TERMINAL.npy'] + (['PRE_RUN.json', 'OBS.npy'] if neural else [])
        if reconstruction is None:
            required.append('FRAMES.npz')
            if view == 'detail': required.append('DETAIL_FRAMES.npz')
        verified = {}
        for name in required:
            file = path / name
            if not file.is_file() or files.get(name) != sha(file):
                raise ValueError(f'Missing or hash-mismatched sealed input: {file}')
            verified[name] = files[name]
        for name in ['PRE_RUN.json', 'SEAL.json', 'RESULT.json']:
            verified[name] = sha(path / name)
        if neural:
            if seal.get('neural_parameters_before') != seal.get('neural_parameters_after'):
                raise ValueError('Sealed neural parameters changed during execution')
            identity = pre['cell']
            binding = pre['freeze']
            frozen_path = Path(binding['path'])
            if not frozen_path.is_absolute(): frozen_path = ROOT / frozen_path
            if not frozen_path.is_file() or sha(frozen_path) != binding['sha256']:
                raise ValueError('Neural freeze binding cannot be verified')
            frozen = read_json(frozen_path)
            if frozen['cells'][pre['cell_index']] != identity:
                raise ValueError('Neural cell differs from its result-free freeze')
            sources = frozen['sources']; extra = [pre['source_snapshot']]
            relative = 'tools/fast_adapter/p45_evaluate.py'
            verified['freeze'] = dict(path=str(frozen_path),sha256=binding['sha256'])
        else:
            if not str(pre.get('scope', '')).startswith(('p45_engineering_oracle','p45_slider_oracle_engineering')):
                raise ValueError('Unrecognized engineering scope')
            identity = pre['config']; sources = pre['sources']; extra = []
            candidates = [name for name in sources if '/p45_' in name and 'engineering' in name and 'worker' not in name]
            if len(candidates) != 1:
                raise ValueError('Engineering runner source binding is ambiguous')
            relative = candidates[0]
        runner = source_path(path, relative, sources[relative], extra)
        verified['source_order'] = verify_source_order(runner, neural)
        verified['declared_source_inventory'] = sources
        task = identity['task']; seed = identity['condition_seed' if neural else 'seed']; split = identity['split']
        if task not in TASKS or result.get('task') != task or result.get('split') != split or result.get('condition_seed' if neural else 'seed') != seed:
            raise ValueError('Result identity disagrees with PRE_RUN')
        if neural and any(result.get(key) != identity.get(key) for key in ['arm','rep','model_seed']):
            raise ValueError('Result model/arm/representation disagrees with PRE_RUN')
        outcome_fields = ['validated_success', 'mechanical_valid'] if neural else ['qualified']
        if any(type(result.get(key)) is not bool for key in outcome_fields):
            raise ValueError('Complete boolean outcome fields are required; missing results are not failures')
        trace = np.load(path / 'TRACE.npy', allow_pickle=True).tolist()
        reconstructed = None
        if reconstruction is not None:
            if not neural:
                raise ValueError('State reconstruction is currently restricted to complete P45 neural cells')
            folder = Path(reconstruction).resolve()
            reconstructed = read_json(folder/'MANIFEST.json')
            if (reconstructed.get('schema') != 'prove.p45-state-render.v1' or reconstructed.get('status') != 'complete'
                    or reconstructed.get('physics_steps_executed') != 0):
                raise ValueError('Invalid or incomplete zero-integration reconstruction manifest')
            for name in ['PRE_RUN.json','SEAL.json','TRACE.npy','TERMINAL.npy','RESULT.json','OBS.npy']:
                if reconstructed['origin_files'].get(name) != sha(path/name):
                    raise ValueError(f'Reconstruction is bound to different source bytes: {name}')
            name='RECONSTRUCTED_FRAMES.npz'
            if reconstructed['output_sha256'].get(name) != sha(folder/name):
                raise ValueError('Reconstructed frames hash mismatch')
            frames_path=folder/name
            details_path=folder/'RECONSTRUCTED_DETAIL_FRAMES.npz'
            if view == 'detail' and reconstructed['output_sha256'].get(details_path.name) != sha(details_path):
                raise ValueError('Reconstructed detail frames hash mismatch')
            verified['reconstruction']=dict(path=str(folder),manifest_sha256=sha(folder/'MANIFEST.json'),manifest=reconstructed)
        else:
            frames_path=path/'FRAMES.npz';details_path=path/'DETAIL_FRAMES.npz'
        with np.load(frames_path, allow_pickle=False) as array: frames = array['frames'].copy()
        terminal = np.load(path / 'TERMINAL.npy', allow_pickle=True).item()
        if not trace or [row['t'] for row in trace] != list(range(len(trace))):
            raise ValueError('Trace action indices must be contiguous from zero')
        if len(trace) != pre['steps']:
            raise ValueError('Trace length differs from PRE_RUN')
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) not in (len(trace),len(trace)+1):
            raise ValueError('Invalid raw RGB frame shape or action/frame correspondence')
        if neural and len(frames) != len(trace)+1:
            raise ValueError('P45 neural cells require the sealed terminal RGB frame')
        if view == 'detail':
            with np.load(details_path,allow_pickle=False) as array: details=array['frames']
            if details.dtype != np.uint8 or details.shape != (len(frames),frames.shape[1],frames.shape[2]//2,3):
                raise ValueError('Detail/raw frames are not aligned')
            frames = np.concatenate([frames[:,:,:frames.shape[2]//2],details],axis=2)
        times = np.array([row['state']['time'] for row in trace] + [terminal['time']],float)
        intervals=np.diff(times);dt=float(np.median(intervals))
        if not np.isfinite(times).all() or dt<=0 or not np.allclose(intervals,dt,atol=1e-7,rtol=1e-6):
            raise ValueError('Physical timestamps do not match a uniform recorded action stream')
        return cls(path,pre,result,trace,frames,terminal,neural,task,seed,split,
                   identity.get('arm','oracle'),identity.get('rep',''),dt,verified,len(frames)==len(trace)+1)

    def annotation(self, frame_index):
        """Only already executed rows appear beside a pre-action RGB frame."""
        completed = min(frame_index,len(self.trace))
        rows=self.trace[:completed]
        updates=[row.get('adaptation') for row in rows if row.get('adaptation') is not None]
        events=[item for item in updates if item.get('U',0)>0]
        fits=[item for item in updates if item.get('fit_updated',False)]
        last=rows[-1] if rows else None
        controller={} if last is None else last.get('controller_after',last.get('controller',{}))
        stage='初始观察姿态' if last is None else STAGES.get(last['stage'],last['stage'])
        if controller.get('phase')=='hold':stage='保持最终操作姿态'
        fresh=None if last is None else last.get('adaptation')
        return dict(frame_index=frame_index,completed_actions=completed,elapsed_seconds=completed*self.dt,
                    stage=stage,gradient_events=len(events),gradient_steps=sum(int(item.get('U',0)) for item in events),
                    fit_events=len(fits),estimate_frozen=bool(controller.get('adaptation_frozen',False)),
                    current_completed_gradient_steps=0 if fresh is None else int(fresh.get('U',0)),
                    current_completed_latent_change=None if fresh is None else fresh.get('latent_change'),
                    outcome_visible=False)


def validate_pair(left,right):
    if not left.neural or not right.neural or left.arm!='fixed' or right.arm!='continuous':
        raise ValueError('A comparison requires one neural fixed and one continuous cell')
    for key in ['task','seed','split','rep']:
        if getattr(left,key)!=getattr(right,key):raise ValueError(f'Pair mismatch: {key}')
    for key in ['model_seed']:
        if left.pre['cell'][key]!=right.pre['cell'][key]:raise ValueError(f'Pair mismatch: {key}')
    for key in ['candidate','environment','freeze']:
        if left.pre[key]!=right.pre[key]:raise ValueError(f'Pair has different frozen {key}')
    if len(left.trace)!=len(right.trace) or not math.isclose(left.dt,right.dt,abs_tol=1e-9):
        raise ValueError('Comparison cells have different execution budgets')


def load_font(size,explicit=None):
    choices=[Path(explicit)] if explicit else [Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),Path('/usr/share/fonts/truetype/arphic/uming.ttc')]
    for path in choices:
        if path.is_file():return ImageFont.truetype(str(path),size),path
    raise ValueError('A Chinese font is required; provide --font')


def final_label(episode):
    result=episode.result
    if not episode.neural:
        return '工程资格通过' if result.get('qualified') is True else '工程资格未通过'
    if result.get('validated_success') is True:return '任务完成，机械检查有效'
    if result.get('mechanical_valid') is False:return '机械异常，未计为有效成功'
    return '任务未完成'


def panel(episode,index,font,small,terminal=False):
    canvas=Image.new('RGB',(896,688),'#101c26');draw=ImageDraw.Draw(canvas)
    color='#ffc36b' if not episode.neural else '#8bd7f0'
    identity='ORACLE 工程演示｜使用目标真值｜非 Adapter 成绩' if not episode.neural else f'{ARMS.get(episode.arm,episode.arm)}｜{REPS.get(episode.rep,episode.rep)}'
    if episode.pre.get('synthetic_test_fixture'):
        identity='脚本自检｜合成占位图｜不是机器人执行'
    draw.text((16,9),identity,font=font,fill=color)
    split='ID' if episode.split=='id' else 'OOD'
    rebuilt='reconstruction' in episode.verified
    provenance='封存状态重建回放' if rebuilt else '原始执行录像'
    draw.text((16,42),f'{provenance} | {TASKS[episode.task]} · {split} · 条件 {episode.seed}',font=small,fill='#ffc36b' if rebuilt else '#c0d0da')
    raw=episode.frames[min(index,len(episode.frames)-1)]
    canvas.paste(Image.fromarray(raw).resize((896,448),Image.Resampling.LANCZOS),(0,78))
    annotation=episode.annotation(index)
    if terminal:
        draw.rectangle((0,526,896,688),fill='#162b34')
        label='轨迹结束｜封存终态评估：'+final_label(episode)
        draw.text((16,538),label,font=font,fill='#ffffff')
        picture=('背景由封存终态重建；不是当时保存的原始 RGB' if rebuilt else '背景为封存终态画面') if episode.terminal_frame else '背景为最后原始画面；工程源未保存终态 RGB'
        draw.text((16,574),picture,font=small,fill='#b6cad4')
        draw.text((16,608),'此结论来自封存后 RESULT；未提前显示在执行片段中',font=small,fill='#b6cad4')
    else:
        draw.text((16,537),f'{annotation["elapsed_seconds"]:5.2f} 秒  |  已执行：{annotation["stage"]}',font=font,fill='white')
        if episode.neural:
            line=f'已发生 z 梯度更新：{annotation["gradient_events"]} 次事件 / {annotation["gradient_steps"]} 步'
            if episode.arm=='multiview':line=f'已发生多视图拟合：{annotation["fit_events"]} 次；没有 z 梯度更新'
            if annotation['current_completed_gradient_steps']:
                line+=f'  · 本次 Δz={annotation["current_completed_latent_change"]:.3g}'
        else:line='控制器读取工程 Oracle；没有在线 z 适配'
        draw.text((16,575),line,font=small,fill='#a9e6ef')
        if annotation['estimate_frozen']:
            line='TRACE 已记录闭合阶段锁定：后续目标估计冻结'
        else:line='目标估计尚未进入闭合后的冻结阶段' if episode.neural else '阶段信息来自封存控制轨迹'
        draw.text((16,607),line,font=small,fill='#c5d6df')
    assumption='从开始即假设已牢固持物；物块与槽壁保留真实接触' if episode.task=='insert' else '真实机械臂与手指接触；没有物体到手指的隐藏吸附'
    if rebuilt and episode.task!='insert':assumption='状态重建：零积分；派生量可能与原始画面相差约 1 毫秒'
    draw.text((16,650),assumption,font=small,fill='#e2b980')
    return canvas,annotation


def render(episodes,out,selection_reason,view='recorded',font_path=None):
    out=Path(out).resolve()
    if any(out==episode.path or episode.path in out.parents for episode in episodes):
        raise ValueError('Derivative output must be outside every original episode directory')
    if len(episodes)==2:validate_pair(*episodes)
    if out.exists():raise FileExistsError('Output directory already exists; originals/derivatives are never overwritten')
    font,font_file=load_font(25,font_path);small,_=load_font(20,font_path)
    out.mkdir(parents=True)
    inputs=[dict(path=str(e.path),identity=dict(task=e.task,seed=e.seed,split=e.split,arm=e.arm,rep=e.rep),verified=e.verified,frames=len(e.frames),trace_rows=len(e.trace),terminal_frame_available=e.terminal_frame,terminal_rgb_sealed=e.terminal_frame and 'reconstruction' not in e.verified) for e in episodes]
    manifest=dict(schema='prove.p45-video-derivative.v1',status='started',inputs=inputs,
                  script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),font=dict(path=str(font_file),sha256=sha(font_file)),
                  selection_reason=selection_reason,representativeness='deliberately selected examples; not an unbiased success-rate sample',
                  pixels=('synthetic test fixture; not robot execution' if any(e.pre.get('synthetic_test_fixture') for e in episodes) else
                          're-rendered sealed physics states; not original execution RGB' if any('reconstruction' in e.verified for e in episodes) else
                          'sealed real-execution RGB; no simulator rendering or generation'),view=view,
                  annotation_time_rule='frame i shows only completed actions with t < i; outcome appears only on final result card',
                  result_binding='RESULT hash bound to derivative; legacy seals do not hash RESULT and do not prove wall-clock ordering')
    write_json(out/'MANIFEST.json',manifest)
    import imageio.v2 as imageio
    events=[];fps=1/episodes[0].dt;count=max(len(e.frames) for e in episodes)
    with imageio.get_writer(out/'annotated.mp4',fps=fps,codec='libx264',quality=8,macro_block_size=16) as writer:
        for index in range(count):
            images=[];annotations=[]
            for episode in episodes:
                frame,annotation=panel(episode,index,font,small);images.append(frame);annotations.append(annotation)
            merged=Image.new('RGB',(896*len(images),688))
            for i,frame in enumerate(images):merged.paste(frame,(896*i,0))
            if index==0:merged.save(out/'poster.png')
            writer.append_data(np.asarray(merged));events.append(annotations)
        cards=[panel(e,len(e.trace),font,small,terminal=True)[0] for e in episodes]
        merged=Image.new('RGB',(896*len(cards),688))
        for i,frame in enumerate(cards):merged.paste(frame,(896*i,0))
        merged.save(out/'result.png')
        for _ in range(round(2*fps)):writer.append_data(np.asarray(merged))
    write_json(out/'ANNOTATIONS.json',events)
    manifest.update(status='complete',fps=fps,raw_display_frames=count,result_card_seconds=round(2*fps)/fps,
                    output_sha256={name:sha(out/name) for name in ['annotated.mp4','poster.png','result.png','ANNOTATIONS.json']})
    provenance=('画面是封存状态重建回放，不是当时保存的原始 RGB；零物理积分，但前向重建的派生量可能与原始画面相差一个积分步。' if any('reconstruction' in e.verified for e in episodes) else '画面来自封存的真实执行 RGB。')
    lines=['# P45 视频个例','',f'选择理由：{selection_reason}','',provenance+'阶段与更新只显示该画面之前已执行的记录，最终评价只出现在结束卡。','',
           '- [播放注释视频](annotated.mp4)','- [来源、哈希与校验清单](MANIFEST.json)','- [逐帧注释依据](ANNOTATIONS.json)','']
    for episode in episodes:
        label='Oracle 工程演示，不属于 Adapter 成绩' if not episode.neural else ARMS[episode.arm]
        lines.append(f'- {TASKS[episode.task]}，{episode.split}，条件 {episode.seed}：{label}。')
    lines.extend(['','这里只是明确选择的个例，不能代替完整统计。插入演示假设开始时已经牢固持物。',
                  '校验可确认帧/轨迹内容与 seal 一致，并确认绑定 runner 在评价前写 seal；旧文件格式不提供不可伪造的墙钟时序证明。'])
    (out/'INDEX_ZH.md').write_text('\n'.join(lines)+'\n')
    manifest['output_sha256']['INDEX_ZH.md']=sha(out/'INDEX_ZH.md')
    write_json(out/'MANIFEST.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    for name in ['single','pair']:
        p=sub.add_parser(name)
        if name=='single':p.add_argument('--episode',required=True)
        else:p.add_argument('--fixed',required=True);p.add_argument('--continuous',required=True)
        if name=='single':p.add_argument('--reconstruction')
        else:p.add_argument('--fixed-reconstruction');p.add_argument('--continuous-reconstruction')
        p.add_argument('--out',required=True);p.add_argument('--selection-reason',required=True)
        p.add_argument('--view',choices=['recorded','detail'],default='recorded');p.add_argument('--font')
    args=parser.parse_args();paths=[args.episode] if args.mode=='single' else [args.fixed,args.continuous]
    reconstructions=[args.reconstruction] if args.mode=='single' else [args.fixed_reconstruction,args.continuous_reconstruction]
    manifest=render([Episode.load(path,args.view,replay) for path,replay in zip(paths,reconstructions)],args.out,args.selection_reason,args.view,args.font)
    print(json.dumps(dict(status=manifest['status'],out=str(Path(args.out).resolve()),inputs=len(manifest['inputs'])),ensure_ascii=False))


if __name__=='__main__':main()
