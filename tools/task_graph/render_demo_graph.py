"""Render original demo RGB with explicit human/root annotations, never inferred stages.

Annotation JSON: {"stages": [{"start":0,"stop":30,"stage":"Move",
"roles":"book -> caddy"}], "status":"provisional root decomposition"}.
Intervals use original action indices, half open; RGB at t is post-action t.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .inspect_demo import sha256_file


def validate_annotations(annotation: dict, count: int) -> list[dict]:
    stages = annotation['stages']
    end = 0
    for stage in stages:
        start, stop = stage['start'], stage['stop']
        if not isinstance(start, int) or not isinstance(stop, int) or start != end or not start < stop <= count:
            raise ValueError('stages must cover contiguous half-open action intervals')
        if not isinstance(stage['stage'], str) or not stage['stage'].strip():
            raise ValueError('stage label required')
        if not isinstance(stage.get('roles', ''), str):
            raise ValueError('roles must be text')
        end = stop
    if end != count:
        raise ValueError('stages must cover the complete demo')
    return stages


def frame_label(frame: int, count: int, stages: list[dict], language: str = "en") -> list[str]:
    stage = next(stage for stage in stages if stage['start'] <= frame < stage['stop'])
    if language == 'zh':
        labels = {'Move': '移动', 'Approach': '接近', 'Grasp': '抓取', 'Lift': '抬起',
                  'Align': '对齐', 'Insert': '放入后格（非销孔插入技能）'}
        text = ['官方单条演示；未调用Fast Adapter',
                f'动作 {frame} / {count - 1}；图像为该动作执行后',
                f"阶段：{labels.get(stage['stage'], stage['stage'])}",
                '执行者：Panda机械臂  操作物：黑色书本  目标：收纳盒后格']
        if frame == count - 1:
            text.append('尚未验证物理成功；源终点标记不能作成功依据')
        return text
    text = ['Original official demo playback / no Adapter',
            f'Action {frame} / {count - 1}; image after action',
            f"Stage: {stage['stage']}  |  {stage.get('roles', '')}"]
    if frame == count - 1:
        text.append('End outcome: unverified (dataset terminal flag is not success evidence)')
    return text


def render_demo(dataset: Path, output: Path, annotations: Path, *, group='data/demo_0', fps=20, language='zh',
                font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc') -> dict:
    if output.exists() or output.with_suffix('.manifest.json').exists():
        raise FileExistsError('refusing to overwrite existing video evidence')
    if fps <= 0:
        raise ValueError('fps must be positive')
    font = ImageFont.truetype(font_path, 26)
    annotation = json.loads(annotations.read_text())
    with h5py.File(dataset, 'r') as stream:
        demo = stream[group]
        count = len(demo['actions'])
        stages = validate_annotations(annotation, count)
        agent, wrist = demo['obs/agentview_rgb'], demo['obs/eye_in_hand_rgb']
        if len(agent) != count or len(wrist) != count:
            raise ValueError('image/action length mismatch')
        output.parent.mkdir(parents=True, exist_ok=True)
        width, height = 1024, 768
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                   '-threads', '1', '-pix_fmt', 'yuv420p', str(output)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame in range(count):
                canvas = Image.new('RGB', (width, height), '#141a22')
                for x, rgb in [(0, agent), (512, wrist)]:
                    arr = np.asarray(rgb[frame])
                    if arr.dtype != np.uint8 or arr.shape[-1] != 3:
                        raise ValueError('expected uint8 RGB')
                    canvas.paste(Image.fromarray(arr[::-1]).resize((512, 512)), (x, 0))
                draw = ImageDraw.Draw(canvas)
                for x, title in [(0, '外部视角 · 原始128×128'), (512, '腕部视角 · 原始128×128')]:
                    draw.rectangle((x, 0, x + 511, 40), fill='#141a22')
                    draw.text((x + 12, 4), title, fill='white', font=font)
                lines = frame_label(frame, count, stages, language)
                lines.append('主Agent离线审阅证据并提出阶段；尚无闭环验证' if language == 'zh'
                             else 'Provisional root annotation; no closed-loop validation')
                for line, text in enumerate(lines):
                    draw.text((12, 519 + line * 38), text, fill='white', font=font)
                process.stdin.write(canvas.tobytes())
            process.stdin.close()
            error = process.stderr.read().decode()
            if process.wait() != 0:
                raise RuntimeError(error)
        except BaseException:
            process.kill()
            process.wait()
            raise
    manifest = {'format': 'prove.annotated-original-demo.v1', 'source_kind': 'human_teleoperation',
                'dataset': str(dataset), 'dataset_sha256': sha256_file(dataset), 'group': group,
                'annotations': str(annotations), 'annotation_sha256': sha256_file(annotations),
                'video_sha256': sha256_file(output), 'frame_count': count, 'fps': fps,
                'image_transform': 'OpenGL vertical flip, side-by-side agentview and wrist',
                'annotation_method': 'explicit root labels; no learned or heuristic semantic parsing',
                'outcome': 'unverified; source synthetic terminal flag does not verify success',
                'runtime': 'offline original demo video; Fast Adapter not invoked',
                'language': language, 'font_path': font_path, 'font_size': 26}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--group', default='data/demo_0')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--language', choices=['zh', 'en'], default='zh')
    parser.add_argument('--font-path', default='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    print(json.dumps(render_demo(**vars(parser.parse_args())), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
