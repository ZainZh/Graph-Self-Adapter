#!/usr/bin/env python3
"""Render the sealed P53 complete online episode; never step the simulator.

The prefix and Code handoff are redrawn from sealed qpos with mj_fwdPosition.
The neural micro stage uses the RGB that the episode recorded during execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.task_graph.p52_render_full_online import choose_font, stage_of


STAGES = {'s1': '移动', 's2': '接近', 's3': '抓取', 's4': '抬起',
          's5': '搬运', 's6': '对齐', 's7': '代码对齐', 's8': '代码近接触',
          's9': '局部对齐与放入'}
EXPLANATIONS = {
    's1': '从原始初态接近书本', 's2': '调整末端，进入抓取位置',
    's3': '固定抓取技能闭合夹爪', 's4': '持书抬离桌面',
    's5': '将书本运到收纳盒附近', 's6': '固定技能完成原图前缀对齐',
    's7': '公开本体反馈驱动代码对齐', 's8': '代码接近盒口并交接短技能',
    's9': '冻结神经策略执行短阶段；仅 z 随反馈更新',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def checked_source(root, seal, relative, expected=None):
    path = root / relative
    digest = sha(path)
    if digest != seal['files'][relative] or (expected is not None and digest != expected):
        raise ValueError('Recorded source differs from seal: ' + relative)
    return path, digest


def frame(rgb, stage, step, total_actions, update_count, origin, font_path, result=None, replay=False):
    canvas = Image.new('RGB', (896, 736), (17, 23, 34))
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(str(font_path), 29)
    body = ImageFont.truetype(str(font_path), 22)
    small = ImageFont.truetype(str(font_path), 18)
    heading = '短技能回看（慢放，仅重复已记录帧）' if replay else 'P53 原始 LIBERO：抓书 → 放入收纳盒'
    draw.text((18, 10), heading, font=title, fill='white')
    route = '冻结神经策略 + 在线 z' if stage == 's9' else ('公开输入代码' if stage in ('s7', 's8') else '原图 Code／固定技能')
    draw.text((18, 53), f'{stage.upper()} {STAGES[stage]}  |  {route}', font=body, fill=(130, 220, 255))
    draw.text((18, 87), EXPLANATIONS[stage], font=small, fill=(221, 225, 233))
    for i, label in enumerate(('外部相机 agentview', '腕部相机')):
        canvas.paste(Image.fromarray(np.asarray(rgb[i], dtype=np.uint8)), (i * 448, 125))
        draw.rectangle((i * 448, 125, i * 448 + 220, 152), fill=(20, 25, 32))
        draw.text((i * 448 + 9, 128), label, font=small, fill='white')
    for i, (name, caption) in enumerate((('原图前缀', '抓取 · 抬起 · 搬运'), ('新代码', '对齐 · 近入口'), ('短技能', '局部对齐 · 放入'))):
        x = i * 298 + 7
        active = i == (0 if stage in ('s1', 's2', 's3', 's4', 's5', 's6') else 1 if stage in ('s7', 's8') else 2)
        draw.rectangle((x, 582, x + 286, 629), fill=(48, 145, 220) if active else (61, 68, 80))
        draw.text((x + 10, 585), name, font=body, fill='white')
        draw.text((x + 106, 592), caption, font=small, fill='white')
    progress = f'回看短阶段帧 {step + 1}；仅重复画面' if replay else f'执行动作 {min(step, total_actions)}/{total_actions}'
    draw.text((18, 638), f'{progress}  |  当前短阶段 z 更新 {update_count if stage == "s9" else "未启用"}', font=small, fill=(218, 223, 234))
    draw.text((18, 665), origin, font=small, fill=(212, 215, 220))
    if result is None:
        draw.text((18, 699), '单初态工程演示；固定 z 与在线 z 均成功，不能据此声称在线增益', font=small, fill=(240, 201, 126))
    else:
        draw.text((18, 692), f'终态：官方成功={result["official_success"]}  图完成={result["graph_complete"]}  持书接触连续3帧', font=small, fill=(111, 232, 155))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-metadata-only', action='store_true')
    parser.add_argument('--font', type=Path)
    parser.add_argument('--fps', type=int, default=20)
    args = parser.parse_args()
    root = args.episode.resolve()
    if args.fps <= 0:
        raise ValueError('fps must be positive')
    result = json.loads((root / 'RESULT.json').read_text())
    binding = json.loads((root / 'BINDING.json').read_text())
    seal = json.loads((root / 'SEAL.json').read_text())
    if result['cell']['arm'] != 'online' or result['cell']['zmode'] != 'observed_feedback':
        raise ValueError('Expected the sealed P53 online full-task cell')
    if not (result['official_success'] and result['graph_complete'] and result['network_weights_unchanged']):
        raise ValueError('Source is not a sealed complete success with frozen weights')
    if sha(root / 'SEAL.json') != result['seal_sha256'] or sha(root / 'BINDING.json') != seal['binding_sha256']:
        raise ValueError('Result, binding or seal changed')
    bound = {}
    for relative in ('RESULT.json', 'BINDING.json', 'SEAL.json', 'prefix/TRACE.jsonl', 'code/TRACE.jsonl',
                     'public/ACTIONS.jsonl', 'PREFIX_RESULT.json', 'CODE_RESULT.json', 'STAGE_TRANSITIONS.json'):
        if relative in seal['files']:
            path, digest = checked_source(root, seal, relative)
        else:
            path, digest = root / relative, sha(root / relative)
        bound[relative] = digest
    prefix = read_lines(root / 'prefix/TRACE.jsonl')
    code = read_lines(root / 'code/TRACE.jsonl')
    micro = read_lines(root / 'public/ACTIONS.jsonl')
    if not prefix[-1].get('ready') or prefix[-1].get('action') is not None:
        raise ValueError('Expected terminal prefix handoff observation')
    if code[-1].get('stop_reason') != 'near_ready' or code[-1].get('action') is not None:
        raise ValueError('Expected terminal Code handoff observation')
    if micro[-1]['executed'] or not all(row['executed'] for row in micro[:-1]):
        raise ValueError('Expected executed neural actions followed by one terminal observation')
    if (len(prefix) - 1, len(code) - 1, len(micro) - 1) != (result['prefix_actions'], result['code_actions'], result['actions']):
        raise ValueError('Trace/action counts differ from RESULT')
    if result['total_actions'] != result['prefix_actions'] + result['code_actions'] + result['actions'] or result['updates'] != micro[-1]['updates']:
        raise ValueError('Total actions or z updates differ')
    code_s7_actions = sum(row['stage_id'] == 's7' for row in code[:-1])
    if any(row['stage_id'] not in ('s7', 's8') for row in code[:-1]) or code_s7_actions == 0 or code_s7_actions == result['code_actions']:
        raise ValueError('Code trace does not contain both declared stages')
    transition_pairs = [(x['after_actions'], x['from_stage']) for x in json.loads((root / 'STAGE_TRANSITIONS.json').read_text())]
    expected_pairs = [(result['prefix_actions'], 's6'), (result['prefix_actions'] + code_s7_actions, 's7'),
                      (result['prefix_actions'] + result['code_actions'], 's8'), (result['total_actions'], 's9')]
    if transition_pairs != expected_pairs:
        raise ValueError('Graph transition boundaries differ')
    if args.check_metadata_only:
        print(json.dumps({'metadata_ok': True, 'prefix_actions': result['prefix_actions'],
                          'code_actions': result['code_actions'], 'neural_actions': result['actions'],
                          'source_observations': len(prefix) - 1 + len(code) - 1 + len(micro),
                          'transitions': transition_pairs, 'z_updates': result['updates']}))
        return
    if args.output_dir is None:
        raise ValueError('--output-dir is required when rendering')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source_rows = []
    for folder, rows, key in (('prefix', prefix[:-1], 'file'), ('code', code[:-1], 'file'), ('public', micro, 'observation')):
        for row in rows:
            relative = f'{folder}/{row[key]}'
            _, digest = checked_source(root, seal, relative, row['sha256'])
            bound[relative] = digest
            source_rows.append((folder, row, relative))
    if len(source_rows) != result['total_actions'] + 1:
        raise ValueError('Video must show every action state and the terminal observation')
    font = choose_font(args.font)
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    model = mujoco.MjModel.from_xml_string(binding['environment']['relocated_xml'])
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=448, width=448)
    option = mujoco.MjvOption()
    option.geomgroup[0] = 0
    option.geomgroup[1] = 1
    args.output_dir.mkdir(parents=True)
    video = args.output_dir / 'online_full_task_annotated.mp4'
    cover = args.output_dir / 'cover.png'
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '896x736',
               '-r', str(args.fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    written = 0
    replay_source = []
    try:
        for folder, row, relative in source_rows:
            if folder == 'public':
                with np.load(root / relative, allow_pickle=False) as public:
                    rgb = public['rgb'].copy()
                stage = 's9'
                updates = row['updates']
                origin = '短技能：执行时保存的原始 RGB 与动作／z 日志'
                terminal = result if not row['executed'] else None
                replay_source.append((rgb, row, terminal))
            else:
                with np.load(root / relative, allow_pickle=False) as state:
                    data.qpos[:] = state['supervisor_qpos']
                mujoco.mj_fwdPosition(model, data)
                rgb = []
                for camera in binding['public']['cameras']:
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    rgb.append(renderer.render().copy())
                stage = stage_of(row) if folder == 'prefix' else row['stage_id']
                updates = 0
                origin = '前缀／代码阶段：封存 qpos 重绘；无物理积分或新动作'
                terminal = None
            if stage not in STAGES:
                raise ValueError('Unknown source graph stage: ' + stage)
            picture = frame(rgb, stage, written, result['total_actions'], updates, origin, font, terminal)
            if written == 0:
                picture.save(cover)
            process.stdin.write(np.asarray(picture).tobytes())
            written += 1
        main_frames = written
        for i, (rgb, row, terminal) in enumerate(replay_source):
            picture = frame(rgb, 's9', i, result['total_actions'], row['updates'],
                            '执行时保存的 RGB 慢放；重复画面不是新增动作', font, terminal, replay=True)
            for _ in range(4):
                process.stdin.write(np.asarray(picture).tobytes())
                written += 1
        replay_frames = written - main_frames
        for _ in range(args.fps * 2):
            process.stdin.write(np.asarray(picture).tobytes())
            written += 1
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('Video encoder failed')
    finally:
        renderer.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    manifest = {
        'schema': 'p53.full-online-annotated-video.v1',
        'scope': 'Sealed single-initial-state success visualization; no rollout or physical integration',
        'source_episode': str(root),
        'result_sha256': sha(root / 'RESULT.json'),
        'seal_sha256': sha(root / 'SEAL.json'),
        'source_bindings': bound,
        'source_frames': len(source_rows),
        'main_frames': main_frames,
        'replay_frames': replay_frames,
        'terminal_hold_frames': args.fps * 2,
        'replay_source': '15 already recorded neural observations repeated four times each; no new actions',
        'source_action_segments': {'prefix': result['prefix_actions'], 'code': result['code_actions'], 'neural': result['actions']},
        'terminal_observations': 1,
        'neural_z_updates': result['updates'],
        'fps': args.fps,
        'video_frames': written,
        'video': {'file': video.name, 'sha256': sha(video)},
        'cover': {'file': cover.name, 'sha256': sha(cover)},
        'visualization': 'prefix and Code mj_fwdPosition from sealed qpos; neural uses recorded public RGB',
        'claim_limit': 'BC, fixed-z and online-z all succeeded on the same initial state; no online-z incremental benefit shown',
    }
    (args.output_dir / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'video': str(video), 'cover': str(cover), 'source_frames': len(source_rows), 'video_frames': written}))


if __name__ == '__main__':
    main()
