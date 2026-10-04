#!/usr/bin/env python3
"""Render the sealed 455406 position-only triplet with zero physics integration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prove.fast_adapter.robot_geometry_env_p45_v02 import RobotGeometrySkill, configuration, source_assets
from tools.fast_adapter.p45_replay_video import integration_guard, restore_integration
from tools.fast_adapter.p45_video import STAGES, load_font


ARMS = ('fixed', 'online', 'direct')
ARM_LABEL = {'fixed': '固定几何', 'online': '梯度在线更新', 'direct': '同射线直接解'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def validate_video_freeze(path):
    """Validate every runtime source and the selected model without requiring P47 bytes."""
    value = read(path)
    if (value.get('schema') != 'prove.p58-cup-development-evaluation.v1'
            or value.get('status') != 'frozen' or value.get('steps') != 240):
        raise ValueError('Invalid P58 cup development freeze')
    for name, digest in value['sources'].items():
        if sha(name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    selected = value['models']['position_only']
    if sha(selected['path']) != selected['sha256']:
        raise ValueError('Selected position-only model changed')
    if value.get('robot_assets') != source_assets():
        raise ValueError('Robot assets changed')
    return value


def verify_episode(path, freeze, freeze_path, expected_index, arm):
    path = Path(path)
    pre, seal, result = [read(path / name) for name in ('PRE_RUN.json', 'SEAL.json', 'RESULT.json')]
    expected = dict(condition_seed=455406, representation='position_only', arm=arm)
    if pre.get('index') != expected_index or pre.get('cell') != expected or result.get('cell') != expected:
        raise ValueError('Representative cell identity changed')
    if pre.get('freeze_sha256') != sha(freeze_path) or freeze['cells'][expected_index] != expected:
        raise ValueError('Representative cell is not bound to the development freeze')
    if seal.get('schema') != 'prove.p58-cup-trajectory-seal.v1':
        raise ValueError('Unsupported P58 cup seal')
    for name, digest in seal['files'].items():
        if sha(path / name) != digest:
            raise ValueError('Sealed source changed: ' + str(path / name))
    if seal['neural_parameters_before'] != seal['neural_parameters_after']:
        raise ValueError('Source execution changed neural weights')
    if result.get('neural_weights_unchanged') is not True or type(result.get('validated_success')) is not bool:
        raise ValueError('Source result is incomplete')
    trace = np.load(path / 'TRACE.npy', allow_pickle=True).tolist()
    terminal = np.load(path / 'TERMINAL.npy', allow_pickle=True).item()
    if len(trace) != 240 or [row['t'] for row in trace] != list(range(240)):
        raise ValueError('Representative trace is incomplete')
    return dict(path=path, arm=arm, pre=pre, seal=seal, result=result, trace=trace, terminal=terminal)


def annotation(episode, frame_index):
    completed = min(frame_index, len(episode['trace']))
    previous = episode['trace'][completed - 1] if completed else None
    stage = '初始观察姿态' if previous is None else STAGES.get(previous['stage'], previous['stage'])
    record = None if previous is None else previous.get('adaptation')
    updates = 0 if record is None else int(record.get('update_events', 0))
    support = 0 if record is None else int(record.get('S', 0))
    frozen = previous is not None and previous['stage'] in (
        'close physical fingers', 'lift with finger contacts')
    return dict(frame_index=frame_index, completed_actions=completed, stage=stage,
                update_events=updates, support_views=support, estimate_frozen=frozen)


def panel(rgb, episode, frame_index, font, small, terminal=False):
    canvas = Image.new('RGB', (896, 688), '#101c26')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 9), ARM_LABEL[episode['arm']] + '｜抓持中心＋训练平均朝向', font=font, fill='#8bd7f0')
    draw.text((16, 42), '封存状态重建回放｜杯柄抓取 · 开发条件 455406', font=small, fill='#ffc36b')
    canvas.paste(Image.fromarray(rgb).resize((896, 448), Image.Resampling.LANCZOS), (0, 78))
    note = annotation(episode, frame_index)
    if terminal:
        draw.rectangle((0, 526, 896, 688), fill='#162b34')
        outcome = ('任务完成，机械检查有效' if episode['result']['validated_success'] else
                   '机械异常，未计为有效成功' if not episode['result']['mechanical_valid'] else '任务未完成')
        draw.text((16, 538), '轨迹结束｜封存终态：' + outcome, font=font, fill='white')
        draw.text((16, 578), '最终结果仅在结束卡显示；开发个例不代替确认集统计', font=small, fill='#b6cad4')
    else:
        draw.text((16, 537), f"已执行 {note['completed_actions']:3d}/240 步｜{note['stage']}", font=font, fill='white')
        if episode['arm'] == 'fixed':
            update_line = f"固定 z；公开射线历史 {note['support_views']} 帧"
        else:
            update_line = f"已发生 {note['update_events']} 次 z 更新；公开射线历史 {note['support_views']} 帧"
        draw.text((16, 578), update_line, font=small, fill='#a9e6ef')
        state = '闭合前目标估计已冻结' if note['estimate_frozen'] else '目标估计仍处于闭合前公开观察阶段'
        draw.text((16, 612), state, font=small, fill='#c5d6df')
    draw.text((16, 650), '逐帧恢复封存状态；物理积分为零；画面不是执行时保存的原始 RGB', font=small, fill='#e2b980')
    return canvas, note


def render(development, output, font_path=None):
    development = Path(development).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    freeze_path = development / 'EVALUATION_FREEZE.json'
    freeze = validate_video_freeze(freeze_path)
    episodes = [verify_episode(development / 'cells' /
        f'{index:02d}_455406_position_only_{arm}', freeze, freeze_path, index, arm)
        for index, arm in enumerate(ARMS)]
    output.mkdir(parents=True)
    manifest = dict(schema='prove.p58-cup-state-replay-video.v1', status='started',
        source_freeze=dict(path=str(freeze_path), sha256=sha(freeze_path)),
        source_episodes=[dict(path=str(item['path']), arm=item['arm'],
            files={name: sha(item['path'] / name) for name in
                   ('PRE_RUN.json', 'SEAL.json', 'TRACE.npy', 'OBS.npy', 'TERMINAL.npy', 'RESULT.json')})
            for item in episodes],
        selection='seed 455406 position-only fixed/online/direct; fixed fails while both updated arms succeed',
        representation='grasp center plus training-mean orientation; not a complete pose',
        original_seals_unchanged=True, physics_steps_executed=0,
        frame_identity='sealed-state reconstruction; not original saved RGB')
    write(output / 'MANIFEST.json', manifest)
    import imageio.v2 as imageio
    import mujoco
    env = None
    frames = [[] for _ in episodes]
    try:
        with integration_guard(mujoco) as receipt:
            env = RobotGeometrySkill(configuration('cup', 455406, 'geometry'), render=True)
            receipt['constructor_active'] = False
            env.step = lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError('Physics integration forbidden in sealed replay'))
            for episode_index, episode in enumerate(episodes):
                for state in [row['state'] for row in episode['trace']] + [episode['terminal']]:
                    restore_integration(mujoco, env.model, env.data, state)
                    env.renderer.disable_segmentation_rendering()
                    env.renderer.update_scene(env.data, camera='global')
                    global_rgb = env.renderer.render().copy()
                    env.renderer.update_scene(env.data, camera='wrist')
                    wrist_rgb = env.renderer.render().copy()
                    frames[episode_index].append(np.concatenate([global_rgb, wrist_rgb], axis=1))
            manifest.update({key: value for key, value in receipt.items() if key != 'constructor_active'})
        font, font_file = load_font(25, font_path)
        small, _ = load_font(20, font_path)
        annotations = []
        video = output / 'position_only_455406_fixed_online_direct.mp4'
        with imageio.get_writer(video, fps=25, codec='libx264', quality=8, macro_block_size=16) as writer:
            for frame_index in range(240):
                panes, notes = zip(*(panel(frames[i][frame_index], episode, frame_index, font, small)
                                     for i, episode in enumerate(episodes)))
                combined = Image.new('RGB', (896 * 3, 688))
                for i, pane in enumerate(panes):
                    combined.paste(pane, (896 * i, 0))
                if frame_index == 0:
                    combined.save(output / 'poster.png')
                writer.append_data(np.asarray(combined))
                annotations.append(notes)
            panes = [panel(frames[i][-1], episode, 240, font, small, terminal=True)[0]
                     for i, episode in enumerate(episodes)]
            combined = Image.new('RGB', (896 * 3, 688))
            for i, pane in enumerate(panes):
                combined.paste(pane, (896 * i, 0))
            combined.save(output / 'result.png')
            for _ in range(50):
                writer.append_data(np.asarray(combined))
        write(output / 'ANNOTATIONS.json', annotations)
        manifest.update(status='complete', font=dict(path=str(font_file), sha256=sha(font_file)),
            frames_per_episode=241, displayed_action_frames=240, result_card_seconds=2,
            output_sha256={name: sha(output / name) for name in
                ('position_only_455406_fixed_online_direct.mp4', 'poster.png', 'result.png', 'ANNOTATIONS.json')})
        write(output / 'MANIFEST.json', manifest)
        return manifest
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--font')
    args = parser.parse_args()
    value = render(args.development, args.output, args.font)
    print(json.dumps(dict(status=value['status'], physics_steps_executed=value['physics_steps_executed']), indent=2))


if __name__ == '__main__':
    main()
