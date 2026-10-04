#!/usr/bin/env python3
"""Summarize sealed P60 results and render prespecified three-arm videos.

This utility only reads sealed arrays.  It never imports or advances physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ARMS = ('corner_fixed', 'corner_continuous', 'corner_direct')
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     sort_keys=True, allow_nan=False) + '\n')


def load_complete(freeze_path):
    frozen = json.loads(freeze_path.read_text())
    if (frozen.get('schema') != 'prove.p60-precision-development-freeze.v1'
            or len(frozen.get('cells', [])) != 24):
        raise ValueError('Expected frozen 24-cell P60 development batch')
    records = []
    for index, cell in enumerate(frozen['cells']):
        folder = freeze_path.parent / 'cells' / (
            f"{index:02d}_{cell['split']}_{cell['condition_seed']}_{cell['arm']}")
        if not (folder / 'RESULT.json').is_file() or not (folder / 'SEAL.json').is_file():
            raise ValueError(f'All 24 sealed results required: {folder.name}')
        result = json.loads((folder / 'RESULT.json').read_text())
        seal = json.loads((folder / 'SEAL.json').read_text())
        if (result.get('cell') != cell or result.get('cell_index') != index
                or result.get('seal_sha256') != sha(folder / 'SEAL.json')):
            raise ValueError(f'Result/cell/seal mismatch: {folder.name}')
        if (seal.get('freeze', {}).get('sha256') != sha(freeze_path)
                or seal.get('neural_parameters_before') != seal.get('neural_parameters_after')):
            raise ValueError(f'Freeze or model hash mismatch: {folder.name}')
        for name, digest in seal.get('files', {}).items():
            if sha(folder / name) != digest:
                raise ValueError(f'Sealed file mismatch: {folder / name}')
        records.append(dict(cell=cell, result=result, directory=str(folder),
                            result_sha256=sha(folder / 'RESULT.json'),
                            seal_sha256=sha(folder / 'SEAL.json')))
    return frozen, records


def counts(records):
    groups, paired = [], []
    for split in ('id', 'geometry'):
        table = {}
        for arm in ARMS:
            rows = [row for row in records
                    if row['cell']['split'] == split and row['cell']['arm'] == arm]
            if len(rows) != 4 or len({row['cell']['condition_seed'] for row in rows}) != 4:
                raise ValueError('Each P60 split/arm requires four distinct conditions')
            table[arm] = {row['cell']['condition_seed']:
                          bool(row['result']['validated_success']) for row in rows}
            groups.append(dict(
                split=split, arm=arm, n=4,
                validated_success=sum(table[arm].values()),
                native_success=sum(bool(row['result']['success']) for row in rows),
                mechanical_anomalies=sum(not row['result']['mechanical_valid'] for row in rows),
                gradient_steps=sum(int(row['result']['gradient_steps']) for row in rows),
                direct_solves=sum(int(row['result']['direct_solves']) for row in rows)))
        fixed = table['corner_fixed']
        for arm in ('corner_continuous', 'corner_direct'):
            candidate = table[arm]
            if set(candidate) != set(fixed):
                raise ValueError('Paired P60 seeds differ')
            paired.append(dict(
                split=split, candidate=arm, n=4,
                rescued=sum(not fixed[seed] and candidate[seed] for seed in fixed),
                harmed=sum(fixed[seed] and not candidate[seed] for seed in fixed),
                both_success=sum(fixed[seed] and candidate[seed] for seed in fixed),
                both_failure=sum(not fixed[seed] and not candidate[seed] for seed in fixed)))
    return dict(groups=groups, paired=paired)


def fit(canvas, rgb, box):
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected recorded uint8 RGB')
    image = Image.fromarray(rgb)
    image.thumbnail((box[2], box[3]), Image.Resampling.LANCZOS)
    canvas.paste(image, (box[0] + (box[2] - image.width) // 2,
                         box[1] + (box[3] - image.height) // 2))


def render_triplet(records, output):
    loaded = []
    for record in records:
        folder = Path(record['directory'])
        with np.load(folder / 'FRAMES.npz') as archive:
            frames = archive['frames'].copy()
        trace = np.load(folder / 'TRACE.npy', allow_pickle=True).tolist()
        if len(trace) != 240 or len(frames) != 241:
            raise ValueError('Expected 240 actions and one terminal frame')
        loaded.append((frames, trace))
    width, height, column_width = 1440, 720, 480
    title = ImageFont.truetype(FONT, 21)
    text = ImageFont.truetype(FONT, 17)
    process = subprocess.Popen(
        ['ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo',
         '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '20', '-i', '-',
         '-an', '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p',
         '-movflags', '+faststart', str(output)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    names = {'corner_fixed': '固定 z', 'corner_continuous': '持续梯度更新 z',
             'corner_direct': '同信息 Direct 解 z'}
    try:
        for t in range(241):
            canvas = Image.new('RGB', (width, height), '#101823')
            draw = ImageDraw.Draw(canvas)
            split = 'ID' if records[0]['cell']['split'] == 'id' else '几何 OOD'
            seed = records[0]['cell']['condition_seed']
            draw.text((16, 10), f'P60 DEVELOPMENT · {split} · 条件 {seed} · 同一冻结 P50 小动作模型',
                      font=title, fill='white')
            for column, (record, arrays) in enumerate(zip(records, loaded)):
                x = column * column_width
                frames, trace = arrays
                draw.text((x + 12, 50), names[record['cell']['arm']], font=title, fill='#9bd7f5')
                fit(canvas, frames[t], (x + 8, 88, column_width - 16, 390))
                row = trace[min(t, 239)]
                phase = '终态' if t == 240 else str(row['stage'])
                draw.text((x + 12, 495), f'真实动作步：{min(t, 240)}/240', font=text, fill='white')
                draw.text((x + 12, 524), f'阶段：{phase}', font=text, fill='white')
                if t == 240:
                    verdict = ('严格成功' if record['result']['validated_success'] else
                               ('机械异常' if not record['result']['mechanical_valid'] else '严格失败'))
                    update = f'终态：{verdict}'
                elif record['cell']['arm'] == 'corner_fixed':
                    update = '当前视觉读出；z 不更新'
                elif record['cell']['arm'] == 'corner_continuous':
                    adaptation = row.get('adaptation')
                    update = f'本步梯度 U={adaptation["U"]}' if adaptation else '本步无有效更新'
                else:
                    adaptation = row.get('adaptation')
                    update = ('本步 Direct solve' if adaptation and adaptation.get('solver') else
                              '本步无有效 Direct solve')
                draw.text((x + 12, 553), update, font=text, fill='#ffd09c')
                draw.text((x + 12, 582), '动作头与几何网络权重均冻结', font=text, fill='white')
            draw.text((16, 646), '预先指定每个 split 的首条件；只组合封存真实帧，不重放或新积分。',
                      font=title, fill='#bccddd')
            draw.text((16, 680), '开发证据，不是独立确认；成功/失败按原 P47 严格评价。',
                      font=title, fill='#bccddd')
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode()
        if process.wait():
            raise RuntimeError(error)
    except BaseException:
        process.kill(); process.wait(); raise
    manifest = dict(
        schema='prove.p60-prespecified-triplet-video.v1', video_sha256=sha(output),
        frame_count=241, fps=20, display_fps_not_physics_timing=True,
        selection='prespecified first condition in the split; no outcome-based selection',
        sources=[dict(cell=row['cell'], result_sha256=row['result_sha256'],
                      seal_sha256=row['seal_sha256'],
                      files={name: sha(Path(row['directory']) / name)
                             for name in ('FRAMES.npz', 'TRACE.npy')}) for row in records])
    dump(output.with_suffix('.manifest.json'), manifest)
    return manifest


def present(freeze_path, output):
    _, records = load_complete(freeze_path)
    summary = counts(records)
    output.mkdir(parents=True, exist_ok=False)
    summary.update(schema='prove.p60-reference-presentation.v1', phase='development',
                   freeze_sha256=sha(freeze_path), episodes=24,
                   independent_success_rate_claim=False, records=records)
    dump(output / 'SUMMARY.json', summary)
    for split, seed in (('id', 605100), ('geometry', 605200)):
        triplet = {row['cell']['arm']: row for row in records
                   if row['cell']['split'] == split and row['cell']['condition_seed'] == seed}
        render_triplet([triplet[arm] for arm in ARMS], output / f'{split}_{seed}_three_arms_zh.mp4')
    dump(output / 'MANIFEST.json', dict(
        schema='prove.p60-presentation-manifest.v1', freeze_sha256=sha(freeze_path),
        presenter_sha256=sha(__file__),
        files={path.name: sha(path) for path in output.iterdir() if path.is_file()}))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = present(args.freeze.resolve(), args.output.resolve())
    print(json.dumps(dict(groups=result['groups'], paired=result['paired']),
                     ensure_ascii=False, indent=2))
