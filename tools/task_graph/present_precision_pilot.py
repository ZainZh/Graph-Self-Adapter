#!/usr/bin/env python3
"""P50 offline DEVELOPMENT result page and prespecified neural comparison videos.

Refuses partial/missing or unsealed results. Never imports a physics environment.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from PIL import Image, ImageDraw, ImageFont
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_complete(freeze_path):
    frozen = json.loads(freeze_path.read_text())
    if frozen.get('schema') != 'prove.p50-physical-development-freeze.v1' or len(frozen['cells']) != 32:
        raise ValueError('expected frozen 32-cell P50 development pilot')
    records = []
    for i, cell in enumerate(frozen['cells']):
        path = freeze_path.parent / 'cells' / f"{i:02d}_{cell['split']}_{cell['condition_seed']}_{cell['controller']}_{cell['arm']}"
        # Deliberately no partial publication: parent requested all 32 sealed results.
        if not (path / 'RESULT.json').is_file() or not (path / 'SEAL.json').is_file():
            raise ValueError(f'all 32 sealed results required: {path.name}')
        result = json.loads((path / 'RESULT.json').read_text())
        seal = json.loads((path / 'SEAL.json').read_text())
        if result['cell'] != cell or result['cell_index'] != i or result['seal_sha256'] != sha(path / 'SEAL.json'):
            raise ValueError('result/cell/seal mismatch')
        if seal['freeze']['sha256'] != sha(freeze_path) or seal['neural_parameters_before'] != seal['neural_parameters_after']:
            raise ValueError('freeze or frozen weights mismatch')
        for name, digest in seal['files'].items():
            if sha(path / name) != digest:
                raise ValueError(f'sealed file mismatch: {path / name}')
        records.append(dict(cell=cell, result=result, directory=str(path),
                            result_sha256=sha(path / 'RESULT.json'), seal_sha256=sha(path / 'SEAL.json')))
    return frozen, records


def counts(records):
    groups, pairs = [], []
    for split in ('id', 'geometry'):
        for controller in ('neural', 'scripted'):
            selected = [r for r in records if r['cell']['split'] == split and r['cell']['controller'] == controller]
            by_arm = {}
            for arm in ('fixed', 'continuous'):
                rows = [r for r in selected if r['cell']['arm'] == arm]
                if len(rows) != 4 or len({r['cell']['condition_seed'] for r in rows}) != 4:
                    raise ValueError('four distinct conditions required in every cell group')
                by_arm[arm] = {r['cell']['condition_seed']: bool(r['result']['validated_success']) for r in rows}
                groups.append(dict(split=split, controller=controller, arm=arm, n=4,
                                   validated_success=sum(by_arm[arm].values()),
                                   native_success=sum(bool(r['result']['success']) for r in rows),
                                   mechanical_anomalies=sum(not r['result']['mechanical_valid'] for r in rows)))
            fixed, continuous = by_arm['fixed'], by_arm['continuous']
            if set(fixed) != set(continuous):
                raise ValueError('paired condition seeds differ')
            pairs.append(dict(split=split, controller=controller, n=4,
                              rescued=sum(not fixed[k] and continuous[k] for k in fixed),
                              harmed=sum(fixed[k] and not continuous[k] for k in fixed),
                              both_success=sum(fixed[k] and continuous[k] for k in fixed),
                              both_failure=sum(not fixed[k] and not continuous[k] for k in fixed)))
    return dict(groups=groups, paired=pairs)


def fit_image(canvas, rgb, box):
    """Letterbox display preserving every source pixel and original aspect ratio."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError('expected recorded uint8 RGB')
    image = Image.fromarray(rgb)
    image.thumbnail((box[2], box[3]), Image.Resampling.LANCZOS)
    canvas.paste(image, (box[0] + (box[2]-image.width)//2, box[1] + (box[3]-image.height)//2))


def render_pair(fixed, continuous, output):
    sources = [fixed, continuous]
    loaded = []
    for record in sources:
        path = Path(record['directory'])
        with np.load(path / 'FRAMES.npz') as z:
            frames = z['frames'].copy()
        with np.load(path / 'DETAIL_FRAMES.npz') as z:
            detail = z['frames'].copy()
        trace = np.load(path / 'TRACE.npy', allow_pickle=True).tolist()
        if len(trace) != 240 or len(frames) != 241 or len(detail) != 241:
            raise ValueError('expected complete 240-action and terminal-frame video')
        loaded.append((frames, detail, trace))
    font = ImageFont.truetype(FONT, 22)
    small = ImageFont.truetype(FONT, 19)
    proc = subprocess.Popen(['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
                             '-s','1440x950','-r','20','-i','-','-an','-c:v','libx264','-threads','1',
                             '-pix_fmt','yuv420p','-movflags','+faststart',str(output)],
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for t in range(241):
            canvas = Image.new('RGB', (1440,950), '#111924')
            draw = ImageDraw.Draw(canvas)
            split = 'ID' if fixed['cell']['split'] == 'id' else '几何OOD'
            draw.text((18, 12), f'P50 DEVELOPMENT · {split} · 条件{fixed["cell"]["condition_seed"]} · 已持物Panda局部插入（非LIBERO）',font=font,fill='white')
            for column, (record, arrays) in enumerate(zip(sources, loaded)):
                x = column*720
                frames, detail, trace = arrays
                arm = '固定 z' if column == 0 else '持续更新 z'
                draw.text((x+16,52), f'{arm} · 同一冻结MLP动作头 / 同一冻结几何网络',font=font,fill='#93cff0')
                fit_image(canvas, frames[t], (x+10,90,700,300))
                fit_image(canvas, detail[t], (x+10,430,700,300))
                draw.text((x+16,395), '上：记录的全局＋观测视角；下：记录的细节视角',font=small,fill='white')
                row = trace[min(t,239)]
                if t < 12:
                    phase = '共同前缀：公开观察运动（前12动作）'
                else:
                    phase = '阶段：' + {'observe':'观察','align':'对齐','insert':'插入','settle':'停稳','done':'完成控制阶段', 'shared observation motion':'共同观察运动', 'neural align':'神经动作对齐', 'neural approach':'神经动作接近', 'neural insert':'神经动作插入', 'neural hold':'神经动作保持'}.get(row['stage'],str(row['stage']))
                adaptation = row.get('adaptation')
                update_text = f'本步梯度更新 U={adaptation["U"]}' if adaptation is not None else '本步无latent梯度更新'
                if column == 0:
                    update_text = '固定 z（梯度不更新）'
                if t == 240:
                    phase = '终态（最后动作之后）'
                    update_text = '通过物理有效成功判据' if record['result']['validated_success'] else ('机械异常，保留失败' if not record['result']['mechanical_valid'] else '任务失败，保留记录')
                draw.text((x+16,748),phase,font=font,fill='white')
                draw.text((x+16,791),update_text,font=font,fill='#ffce96' if t==240 else 'white')
                if t < 240:
                    draw.text((x+16,833),f'动作前帧 {t}/240；MLP权重始终冻结',font=small,fill='white')
                else:
                    draw.text((x+16,833),'结果仅于此终态显示；不以离线拟合门作为任务成功',font=small,fill='white')
            draw.text((18,900),'预先指定首个ID/OOD条件；没有按成功筛选视频。仅离线显示缩放，保留原宽高比。',font=font,fill='#b7c5d5')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close(); error=proc.stderr.read().decode()
        if proc.wait():
            raise RuntimeError(error)
    except BaseException:
        proc.kill(); proc.wait(); raise
    manifest = dict(schema='prove.p50-prespecified-paired-video.v1', video_sha256=sha(output),
                    frame_count=241,fps=20,display_fps_not_physics_timing=True,
                    selection='neural; prespecified first condition of each split, independent of outcome',
                    views='FRAMES combined global/public and DETAIL_FRAMES; preserve aspect ratio; no synthetic frames',
                    font=FONT, presenter_sha256=sha(__file__),
                    sources=[dict(cell=r['cell'],result_sha256=r['result_sha256'],seal_sha256=r['seal_sha256'],
                                  files={name:sha(Path(r['directory'])/name) for name in ('FRAMES.npz','DETAIL_FRAMES.npz','TRACE.npy')}) for r in sources])
    dump(output.with_suffix('.manifest.json'),manifest)
    return manifest


def present(freeze_path, output):
    frozen, records = load_complete(freeze_path)
    summary = counts(records)
    output.mkdir(parents=True,exist_ok=False)
    summary.update(schema='prove.p50-development-presentation.v1',phase='DEVELOPMENT',
                   freeze_sha256=sha(freeze_path),n_per_split=4,episodes=32,
                   domain='local Panda preheld insert; not LIBERO; not full manipulation task',
                   offline_qualification_is_task_success=False,independent_success_rate_claim=False,
                   records=records)
    dump(output/'SUMMARY.json',summary)
    for split, seed in [('id',505100),('geometry',505200)]:
        pair = {r['cell']['arm']:r for r in records if r['cell']['split']==split and r['cell']['condition_seed']==seed and r['cell']['controller']=='neural'}
        render_pair(pair['fixed'],pair['continuous'],output/f'{split}_{seed}_neural_pair_zh.mp4')
    group_rows=''.join(f'<tr><td>{"ID" if r["split"]=="id" else "几何OOD"}</td><td>{"冻结MLP" if r["controller"]=="neural" else "脚本控制"}</td><td>{"固定z" if r["arm"]=="fixed" else "持续更新z"}</td><td>{r["validated_success"]}/4</td><td>{r["mechanical_anomalies"]}/4</td></tr>' for r in summary['groups'])
    pair_rows=''.join(f'<tr><td>{"ID" if r["split"]=="id" else "几何OOD"}</td><td>{r["controller"]}</td><td>{r["rescued"]}</td><td>{r["harmed"]}</td><td>{r["both_success"]}</td><td>{r["both_failure"]}</td></tr>' for r in summary['paired'])
    freeze_link=os.path.relpath(freeze_path,output)
    document=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>P50 小型神经动作模型开发试验</title><style>body{{max-width:1200px;margin:32px auto;padding:20px;font:18px/1.7 sans-serif;color:#24333f}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ccd;text-align:left}}video{{width:100%;background:#111}}a{{color:#1769a6}}</style><h1>P50 小型神经动作模型：DEVELOPMENT</h1><p>本地Panda从已牢固持物开始的局部插入，非LIBERO、非完整抓取任务。每个ID/OOD分布仅4个配对条件；两控制器×两种latent更新方式，共32回合。所有普通失败与机械异常均计入，不声称独立成功率确证。</p><p>离线动作近似资格只允许进入物理开发检查，不能当作任务成功。部署中MLP与几何网络权重均冻结；持续更新只作用于latent z。</p><h2>物理有效成功（分母各4）</h2><table><tr><th>分布</th><th>控制器</th><th>latent</th><th>成功</th><th>机械异常</th></tr>{group_rows}</table><h2>同条件持续更新相对固定z（各4对）</h2><table><tr><th>分布</th><th>控制器</th><th>救回</th><th>误伤</th><th>均成功</th><th>均失败</th></tr>{pair_rows}</table><p>救回=固定失败、持续成功；误伤=固定成功、持续失败。小样本开发计数，不代表总体显著优势。</p><h2>预先指定ID 505100：冻结MLP</h2><video controls preload="metadata" src="id_505100_neural_pair_zh.mp4"></video><h2>预先指定几何OOD 505200：冻结MLP</h2><video controls preload="metadata" src="geometry_505200_neural_pair_zh.mp4"></video><p>每组左侧固定z，右侧持续更新z；前12个动作采用共同观察运动。视频仅使用4条预定回合的封存真实图像，结果仅在末帧显示。</p><p><a href="SUMMARY.json">完整计数与逐回合结果来源</a> · <a href="{html.escape(freeze_link)}">结果前冻结</a> · <a href="id_505100_neural_pair_zh.manifest.json">ID视频来源</a> · <a href="geometry_505200_neural_pair_zh.manifest.json">OOD视频来源</a></p></html>'''
    (output/'index.html').write_text(document)
    dump(output/'MANIFEST.json',dict(schema='prove.p50-presentation-manifest.v1',freeze_sha256=sha(freeze_path),
        presenter_sha256=sha(__file__),files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=present(args.freeze.resolve(),args.output.resolve())
    print(json.dumps(dict(groups=result['groups'],paired=result['paired']),ensure_ascii=False,indent=2))
