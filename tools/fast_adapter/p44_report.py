"""Compact Chinese development report and original-RGB Panda comparisons."""
from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.fast_adapter.p44_audit import check_seal
from tools.fast_adapter.p44_evaluate import js, sha


LABELS = {'fixed': '固定 z', 'continuous': '持续更新 z'}
TASKS = {'cup': 'Panda 机械臂抓持杯柄并抬升', 'insert': 'Panda 机械臂插入 · 初始已牢固持物'}
SPLITS = {'id': '训练范围内 ID', 'geometry': '几何范围外 OOD'}
STAGES = {
    'shared observation motion': '共同小幅观察',
    'target not observed: hold': '目标不可见，保持',
    'align held peg': '对齐已持物块的位置与朝向',
    'approach slot': '靠近槽口',
    'insert along estimated axis': '沿估计槽轴插入',
    'align outside cup handle': '在杯柄外侧对齐',
    'approach handle with open fingers': '张开夹爪，靠近杯柄',
    'close physical Panda fingers': '闭合 Panda 实体夹爪',
    'lift with finger contacts': '保持指面接触并抬升',
}


def font(size):
    path = subprocess.check_output(['fc-match', '-f', '%{file}', 'Noto Sans CJK SC'], text=True).strip()
    return ImageFont.truetype(path, size)


def cell_path(frozen, pair, arm):
    return (Path(frozen['execution']['out']) / f'{pair["task"]}_seed{pair["model_seed"]}'
            / f'{pair["split"]}_{pair["seed"]}_{arm}')


def render_pair(frozen, pair, output):
    rows = []
    for arm in frozen['arms']:
        cell = cell_path(frozen, pair, arm)
        pre, seal = check_seal(cell)
        if 'frames' not in seal:
            raise ValueError(f'No sealed live RGB for preselected pair: {cell}')
        result = json.loads((cell / 'RESULT.json').read_text())
        frames = np.load(cell / 'FRAMES.npz', allow_pickle=False)['frames']
        detail = np.load(cell / 'DETAIL_FRAMES.npz', allow_pickle=False)['frames']
        trace = np.load(cell / 'TRACE.npy', allow_pickle=True)
        if frames.shape != (pre['steps'] + 1, 448, 896, 3):
            raise ValueError('Original RGB must cover all pre-action states and terminal')
        if detail.shape != (pre['steps'] + 1, 448, 448, 3):
            raise ValueError('Original external closeup must cover every displayed state')
        rows.append(dict(arm=arm, cell=cell, frames=frames, detail=detail, trace=trace, result=result))
    steps = frozen['steps_per_episode']
    task = pair['task']
    width, height, fps, hold = 1792, 1008, 15, 45
    large, medium, small = font(32), font(28), font(22)
    stem = f'{task}_{pair["split"]}_{pair["seed"]}_seed{pair["model_seed"]}'
    video = output / f'{stem}.mp4'
    cover = output / f'{stem}.jpg'
    encoder = subprocess.Popen([
        'ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
        '-threads', '2', '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
    try:
        for t in range(steps + 1):
            canvas = Image.new('RGB', (width, height), '#111a26')
            draw = ImageDraw.Draw(canvas)
            draw.text((24, 10), TASKS[task], font=large, fill='#f1f6ff')
            draw.text((24, 57), f'开发迁移 · {SPLITS[pair["split"]]} · 条件 {pair["seed"]} · 模型种子 {pair["model_seed"]} · 预定样例',
                      font=small, fill='#bbcee1')
            draw.text((24, 91), '真实关节驱动的仿真机械臂｜原始执行 RGB｜同模型、同初态、同预算',
                      font=small, fill='#bbcee1')
            for col, row in enumerate(rows):
                x = 896 * col
                draw.rectangle((x + 2, 135, x + 894, 180), fill='#223248')
                draw.text((x + 20, 140), LABELS[row['arm']], font=medium, fill='#f1f6ff')
                frame = row['frames'][t]
                canvas.paste(Image.fromarray(frame[:, :448]), (x, 188))
                canvas.paste(Image.fromarray(row['detail'][t]), (x + 448, 188))
                draw.rectangle((x + 7, 195, x + 218, 230), fill='#172333')
                draw.text((x + 14, 199), '全臂与任务工作区', font=small, fill='white')
                draw.rectangle((x + 455, 195, x + 718, 230), fill='#172333')
                draw.text((x + 462, 199), '任务近景 · 展示相机', font=small, fill='white')
                draw.text((x + 24, 650), '腕部输入（缩略图）', font=small, fill='#c8daef')
                canvas.paste(Image.fromarray(frame[:, 448:]).resize((224, 224)), (x + 24, 688))
                trace = row['trace'][min(t, steps - 1)]
                stage = '全部动作后的封存终态' if t == steps else STAGES.get(trace['stage'], trace['stage'])
                draw.text((x + 278, 658), f'{min(t + 1, steps):03d}/{steps} · {stage}', font=small, fill='#eef5ff')
                adaptation = trace['adaptation']
                info = '角点不可用，保留已有几何估计'
                if adaptation:
                    info = f'观察 {adaptation["S"]} 帧 · 本帧 z 更新 {adaptation["U"]} 次'
                if task == 'cup' and t >= frozen.get('cup_close_step', 105):
                    info = '闭合后冻结几何估计，保持抓持并抬升'
                draw.text((x + 278, 706), info, font=small, fill='#b5c9de')
                if t == steps:
                    result = row['result']
                    draw.text((x + 278, 758), '末 10 状态：' + ('通过' if result['success'] else '未通过'),
                              font=medium, fill='#79e4bf' if result['success'] else '#ffb4a7')
                    detail = (f'抬升 {1000 * result["lift"]:.1f} mm · 双侧接触 {"是" if result["bilateral"] else "否"}'
                              if task == 'cup' else
                              f'插入深度 {1000 * result["depth"]:.1f} mm · 横向 {1000 * result["lateral"]:.1f} mm')
                    draw.text((x + 278, 810), detail, font=small, fill='#d0dfef')
                    draw.text((x + 278, 852), '终态结果保留全部成功与失败', font=small, fill='#a9bed3')
                else:
                    criterion = '检查双侧抓持与持续抬升' if task == 'cup' else '检查持续插入深度、横向误差与朝向'
                    draw.text((x + 278, 772), criterion, font=small, fill='#a9bed3')
                    draw.text((x + 278, 820), '完成全部动作后才显示结果', font=small, fill='#a9bed3')
            draw.line((896, 135, 896, 930), fill='#61768c', width=2)
            draw.text((24, 949), '本视频为预定的单个开发样例，仅展示实际完成的执行；不代表独立验证或统计显著性。',
                      font=small, fill='#a9bed3')
            encoder.stdin.write(np.asarray(canvas).tobytes())
            if t == steps:
                canvas.save(cover, quality=94)
                for _ in range(hold - 1):
                    encoder.stdin.write(np.asarray(canvas).tobytes())
    finally:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Annotated video encoding failed')
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'stream=width,height,nb_frames:format=duration',
        '-of', 'json', str(video)], text=True))
    if (int(probe['streams'][0]['nb_frames']) != steps + hold
            or probe['streams'][0]['width'] != width or probe['streams'][0]['height'] != height):
        raise ValueError('Annotated frame count differs from declared timing')
    return dict(pair=pair, video=str(video), video_sha256=sha(video), cover=str(cover),
                cover_sha256=sha(cover), probe=probe,
                layout='Desktop landscape: each arm has full-robot and detail views side by side, wrist-input thumbnail below',
                selection='Result-free protocol media_pairs; retain all paired outcomes',
                image_timing='Live pre-action RGB at t0..steps-1 plus live post-action terminal RGB; outcome label only at terminal',
                source_kind='Original execution images, not reconstructed or decorative robot animation',
                sources=[dict(arm=r['arm'], cell=str(r['cell']),
                              hashes={name: sha(r['cell'] / name) for name in
                                      ('PRE_RUN.json', 'SEAL.json', 'RESULT.json', 'TRACE.npy', 'FRAMES.npz', 'DETAIL_FRAMES.npz')})
                         for r in rows])


def summarize_results(frozen, results):
    """Descriptive counts with task conditions kept distinct from model repeats."""
    summary, split_summary = [], []
    for task in frozen['tasks']:
        task_rows = [row for row in results if row['task'] == task]
        complete = len(task_rows) == 16
        counts = lambda rows: {arm: dict(success=sum(bool(r['success']) for r in rows if r['arm'] == arm),
                                         total=sum(r['arm'] == arm for r in rows))
                               for arm in frozen['arms']}
        summary.append(dict(task=task, complete=complete, completed=len(task_rows), expected=16,
                            counts=counts(task_rows) if complete else None))
        for split in ('id', 'geometry'):
            rows = [row for row in task_rows if row['split'] == split]
            cfg = frozen['tasks'][task]
            seeds = [seed for seed, kind in zip(cfg['seeds'], cfg['splits']) if kind == split]
            expected = {(seed, model_seed, arm) for seed in seeds
                        for model_seed in frozen['model_seeds'] for arm in frozen['arms']}
            actual = {(r['seed'], r['model_seed'], r['arm']) for r in rows}
            complete = len(rows) == 8 and len(seeds) == 2 and actual == expected
            paired = []
            if complete:
                lookup = {(r['seed'], r['model_seed'], r['arm']): r for r in rows}
                for seed in seeds:
                    for model_seed in frozen['model_seeds']:
                        fixed = bool(lookup[seed, model_seed, 'fixed']['success'])
                        continuous = bool(lookup[seed, model_seed, 'continuous']['success'])
                        category = ('both_pass' if fixed and continuous else 'both_fail'
                                    if not fixed and not continuous else 'rescue' if continuous else 'harm')
                        paired.append(dict(seed=seed, model_seed=model_seed, fixed=fixed,
                                           continuous=continuous, category=category))
            split_summary.append(dict(task=task, split=split, complete=complete,
                                      completed=len(rows), expected=8,
                                      independent_conditions=len(seeds), fixed_model_repeats=2,
                                      completed_independent_conditions=len({r['seed'] for r in rows}),
                                      completed_model_seeds=sorted({r['model_seed'] for r in rows}),
                                      paired_model_condition_comparisons=4 if complete else None,
                                      counts=counts(rows) if complete else None,
                                      paired_counts={name: sum(p['category'] == category for p in paired)
                                                     for name, category in [('rescues', 'rescue'), ('harms', 'harm'),
                                                                            ('both_pass', 'both_pass'), ('both_fail', 'both_fail')]}
                                      if complete else None,
                                      paired_outcomes=paired,
                                      inference='Descriptive only; the two model repeats are not independent task conditions'))
    return summary, split_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--media', action='store_true')
    args = parser.parse_args()
    frozen = json.loads(Path(args.freeze).read_text())
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    results, missing, aborts = [], [], []
    for task, cfg in frozen['tasks'].items():
        for model_seed in frozen['model_seeds']:
            for seed, split in zip(cfg['seeds'], cfg['splits']):
                pair = dict(task=task, model_seed=model_seed, seed=seed, split=split)
                for arm in frozen['arms']:
                    cell = cell_path(frozen, pair, arm)
                    if (cell / 'ABORT.json').exists():
                        aborts.append(dict(cell=str(cell), **json.loads((cell / 'ABORT.json').read_text())))
                    if not (cell / 'RESULT.json').exists():
                        missing.append(str(cell))
                        continue
                    pre, _ = check_seal(cell)
                    if pre['bindings']['freeze']['sha256'] != sha(args.freeze):
                        raise ValueError('Mixed pilot protocols in report')
                    result = json.loads((cell / 'RESULT.json').read_text())
                    results.append(dict(result, cell=str(cell), result_sha256=sha(cell / 'RESULT.json')))
    summary, split_summary = summarize_results(frozen, results)
    videos = []
    if args.media:
        for pair in frozen['media_pairs']:
            if all((cell_path(frozen, pair, arm) / 'RESULT.json').exists() for arm in frozen['arms']):
                videos.append(render_pair(frozen, pair, output))
    record = dict(schema='prove.p44-development-report.v1', freeze=dict(path=args.freeze, sha256=sha(args.freeze)),
                  scope='Frozen P43 checkpoints transferred to articulated Panda simulation; development only',
                  complete=len(results) == 32 and not aborts, cells=len(results), expected=32,
                  task_summary=summary, split_summary=split_summary,
                  negative_cases=[{key: row[key] for key in ('task', 'split', 'seed', 'model_seed', 'arm', 'cell')}
                                  for row in results if not row['success']],
                  missing=missing, aborts=aborts, results=results,
                  no_new_rollouts=True, no_statistical_generalization_claim=True, videos=videos)
    js(output / 'REPORT.json', record)
    js(output / 'MEDIA_MANIFEST.json', dict(generator=str(Path(__file__)), generator_sha256=sha(__file__), videos=videos,
                                            new_physics_steps=0, new_network_calls=0))
    lines = ['# P44：真实机械臂形态的仿真开发检查', '',
             'Panda 关节驱动，插入初始已牢固持物。复用 P43 冻结模型，仅 z 可在线更新。', '',
             '这是机械臂形态迁移的开发检查，不是新的独立测试或真实硬件结果。', '',
             f'已完成 {len(results)}/32 个预定执行；工程中止 {len(aborts)} 个。', '']
    for task in summary:
        if not task['complete']:
            lines.append(f'- {TASKS[task["task"]]}：{task["completed"]}/16，尚未完成配对比较。')
        else:
            counts = task['counts']
            lines.append(f'- {TASKS[task["task"]]}：固定 z {counts["fixed"]["success"]}/8；持续更新 z {counts["continuous"]["success"]}/8。')
    lines += ['', '预定每个任务的 ID、OOD 各只有 **2 个独立条件**，每条件复用 2 个固定模型。每个分布的 4 个模型—条件配对不构成 4 个独立条件；仅描述计数，不作统计显著性结论。未完成分布不报告整批成功计数。', '',
              '| 任务 / 分布 | 固定 z | 持续更新 z | 救回 | 退步 | 双通过 | 双失败 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in split_summary:
        label = f'{"杯柄" if row["task"] == "cup" else "插入"} / {SPLITS[row["split"]]}'
        if row['complete']:
            counts, paired = row['counts'], row['paired_counts']
            lines.append(f'| {label} | {counts["fixed"]["success"]}/4 | {counts["continuous"]["success"]}/4 | {paired["rescues"]} | {paired["harms"]} | {paired["both_pass"]} | {paired["both_fail"]} |')
        else:
            lines.append(f'| {label}（{row["completed"]}/8 执行，{row["completed_independent_conditions"]}/2 条件） | 待完成 | 待完成 | — | — | — | — |')
    lines += ['', '救回：同初态、同模型的固定 z 失败而持续更新成功；退步：固定 z 成功而持续更新失败。完整记录保留全部失败与配对，不按结果挑选视频。']
    if results:
        lines += ['', '已完成执行的终态判据如下。判据未通过不自动归因为神经方法失败；工程有效性由独立资格与停止记录判断。', '',
                  '| 任务 / 分布 | 条件 | 模型种子 | 方法 | 终态判据 |',
                  '|---|---:|---:|---|---|']
        for row in results:
            label = f'{"杯柄" if row["task"] == "cup" else "插入"} / {SPLITS[row["split"]]}'
            lines.append(f'| {label} | {row["seed"]} | {row["model_seed"]} | {LABELS[row["arm"]]} | {"通过" if row["success"] else "未通过"} |')
    if aborts:
        lines += ['', '工程中止记录按执行/实现缺口保留，不计作神经方法负结果。']
    (output / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n')
    page = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>P44 机械臂开发检查</title>',
            '<style>body{max-width:1792px;margin:32px auto;padding:0 24px;background:#111a26;color:#eef5ff;font:18px/1.7 sans-serif}video{width:100%;aspect-ratio:16/9}a{color:#8ddbc7}pre{white-space:pre-wrap}table{border-collapse:collapse;width:100%;max-width:1280px}th,td{padding:10px 18px;text-align:left;border-bottom:1px solid #3b4f65}th{background:#223248}.note{color:#b8cce0}</style>',
            '<h1>P44：Panda 机械臂抓杯与插入</h1>',
            '<p>关节驱动的仿真机械臂；插入从已牢固抓持物块开始。使用现有冻结神经模型，比较固定 z 与持续更新 z。属于开发迁移检查。</p>',
            '<pre>' + html.escape('\n'.join(lines[6:10])) + '</pre>',
            '<p class="note">预定每任务的 ID、OOD 各仅 <strong>2 个独立条件</strong>，每条件复用 2 个固定模型。每分布 4 个模型—条件配对，不计作 4 个独立条件；仅描述计数，不作统计显著性结论。未完成分布不报告整批成功计数。</p>',
            '<table><thead><tr><th>任务 / 分布</th><th>固定 z</th><th>持续更新 z</th><th>救回</th><th>退步</th><th>双通过</th><th>双失败</th></tr></thead><tbody>']
    for row in split_summary:
        label = f'{"杯柄" if row["task"] == "cup" else "插入"} / {SPLITS[row["split"]]}'
        if row['complete']:
            counts, paired = row['counts'], row['paired_counts']
            cells = [label, f'{counts["fixed"]["success"]}/4', f'{counts["continuous"]["success"]}/4',
                     paired['rescues'], paired['harms'], paired['both_pass'], paired['both_fail']]
        else:
            cells = [label + f'（{row["completed_independent_conditions"]}/2 条件）',
                     f'{row["completed"]}/8 已完成', '待完成', '—', '—', '—', '—']
        page.append('<tr>' + ''.join('<td>' + html.escape(str(cell)) + '</td>' for cell in cells) + '</tr>')
    page += ['</tbody></table>', '<p class="note">救回：固定 z 失败而持续更新成功；退步：固定 z 成功而持续更新失败。保留全部失败记录，视频按预定条件展示。</p>']
    if aborts:
        page.append('<p>工程中止按执行/实现缺口保留，不计作神经方法负结果。</p>')
    for entry in videos:
        pair = entry['pair']
        page.append(f'<h2>{TASKS[pair["task"]]} · 条件 {pair["seed"]}</h2><video controls preload="metadata" src="{Path(entry["video"]).name}" poster="{Path(entry["cover"]).name}"></video>')
    if results:
        page += ['<h2>实际已完成执行</h2>',
                 '<p class="note">终态判据未通过不自动归因为神经方法失败；工程有效性由独立资格与停止记录判断。以下保留全部已完成结果。</p>',
                 '<table><thead><tr><th>任务 / 分布</th><th>条件</th><th>模型种子</th><th>方法</th><th>终态判据</th></tr></thead><tbody>']
        for row in results:
            cells = [f'{"杯柄" if row["task"] == "cup" else "插入"} / {SPLITS[row["split"]]}',
                     row['seed'], row['model_seed'], LABELS[row['arm']], '通过' if row['success'] else '未通过']
            page.append('<tr>' + ''.join('<td>' + html.escape(str(cell)) + '</td>' for cell in cells) + '</tr>')
        page.append('</tbody></table>')
    page.append('<p><a href="REPORT_ZH.md">开发报告</a> · <a href="REPORT.json">完整记录</a> · <a href="MEDIA_MANIFEST.json">视频来源与哈希</a></p></html>')
    (output / 'index.html').write_text('\n'.join(page))
    print(json.dumps(dict(cells=len(results), aborts=len(aborts), videos=len(videos), out=str(output))), flush=True)


if __name__ == '__main__':
    main()
