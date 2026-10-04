#!/usr/bin/env python3
"""Offline P49 engineering review and real-frame Chinese video; no simulator."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tools.task_graph.collect_libero_demo_gt import sha, dump

LABELS = dict(Move='移动', Approach='接近', Grasp='抓取', Lift='抬起', Align='对齐', Insert='放入后格')
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def load_run(path):
    binding = json.loads((path / 'BINDING.json').read_text())
    result = json.loads((path / 'RESULT.json').read_text())
    audit = json.loads((path / 'TRACE_AUDIT.json').read_text())
    rows = [json.loads(x) for x in (path / 'FRAMES.jsonl').read_text().splitlines()]
    if sha(path / 'BINDING.json') != result['binding_sha256'] or len(rows) != result['frames']:
        raise ValueError('result binding/frame count mismatch')
    graph = json.loads(Path(binding['graph']).read_text())
    if sha(binding['graph']) != binding['implementation'][binding['graph']]:
        raise ValueError('graph binding mismatch')
    table = []
    for node in graph['nodes']:
        events = [(row['frame'], event['judge']) for row in rows for event in row['events']
                  if event['judge']['stage_id'] == node['id']]
        acts = sum(event.get('action') is not None for _, event in events)
        table.append(dict(stage=node['id'], skill=node['skill'], entry_frame=events[0][0] if events else None,
                          completion_frame=next((frame for frame, event in events if event['state'] == 'complete'), None),
                          actions=acts, step_budget=node['step_budget'], no_progress_budget=node['no_progress_budget'],
                          final_state=events[-1][1]['state'] if events else 'not_entered'))
    for row in rows:
        if sha(path / row['file']) != row['sha256']:
            raise ValueError('frame hash mismatch')
    return dict(version=path.name, result=result, audit=audit, stages=table,
                final_controller=rows[-1]['events'][-1].get('controller'),
                final_supervisor=rows[-1]['events'][-1]['supervisor'],
                inputs={name: sha(path / name) for name in ('BINDING.json', 'RESULT.json', 'FRAMES.jsonl', 'TRACE_AUDIT.json')}), rows


def render(path, rows, output):
    """Reuse render_demo_graph's Noto/Pillow/raw RGB FFmpeg evidence convention."""
    if output.exists():
        raise FileExistsError(output)
    font = ImageFont.truetype(FONT, 24)
    graph = json.loads(Path(json.loads((path / 'BINDING.json').read_text())['graph']).read_text())
    skills = {n['id']: n['skill'] for n in graph['nodes']}
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1024x800', '-r', '20', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '1',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for row in rows:
            event = row['events'][-1]
            stage = event['judge']['stage_id']
            canvas = Image.new('RGB', (1024, 800), '#141a22')
            draw = ImageDraw.Draw(canvas)
            with np.load(path / row['file']) as z:
                for x, key in [(0, 'agentview_image'), (512, 'robot0_eye_in_hand_image')]:
                    rgb = z['observation_' + key]
                    if rgb.shape != (128, 128, 3) or rgb.dtype != np.uint8:
                        raise ValueError('expected actual 128px uint8 RGB')
                    canvas.paste(Image.fromarray(rgb[::-1]).resize((512, 512)), (x, 0))
            # Labels sit below images, preserving all real scene pixels.
            draw.text((12, 515), '外部视角（原始128×128）                       腕部视角（原始128×128）', font=font, fill='white')
            controller = event.get('controller', {})
            goal = '成立' if row['supervisor']['success'] else '未成立'
            lines = [f'P49 工程执行 v02 · 同一已知初态 · 帧 {row["frame"]}/{len(rows)-1}',
                     f'实时阶段：{stage} {LABELS[skills[stage]]}   控制：{event.get("backend_kind", "无动作")}',
                     f'固定单条示范位姿 + 当前EEF反馈；参考点 {controller.get("source_index", "—")}',
                     f'官方目标实时监督：{goal}；未调用 Fast Adapter / 神经动作模型',
                     '图像为当前动作前状态；双视角均来自实际物理记录']
            if row is rows[-1]:
                lines[-1] = '结束：放入阶段反复无进展；任务图未完成，官方目标未成立'
            for i, line in enumerate(lines):
                draw.text((12, 560 + 43*i), line, font=font, fill='#ffbb88' if row is rows[-1] and i == 4 else 'white')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        error = proc.stderr.read().decode()
        if proc.wait():
            raise RuntimeError(error)
    except BaseException:
        proc.kill(); proc.wait(); raise
    manifest = dict(schema='prove.libero-graph-real-frame-video.v1', source=str(path),
                    source_frames_sha256=sha(path / 'FRAMES.jsonl'), video_sha256=sha(output),
                    frame_count=len(rows), fps=20, synthetic_frames=0,
                    image_transform='display-only vertical flip and resize; two original recorded RGB views',
                    source_resolution=[128,128], annotation='Chinese live recorded stage/controller/goal; end outcome only last frame',
                    runtime='offline rendering only; no simulator or Adapter invocation', font_path=FONT,
                    renderer_sha256=sha(__file__))
    dump(output.with_suffix('.manifest.json'), manifest)
    return manifest


def summarize(root):
    if (root / 'ROOT_REVIEW_DRAFT.json').exists():
        raise FileExistsError('review already exists')
    runs, all_rows = [], []
    for version in ('engineering_v01', 'engineering_v02'):
        run, rows = load_run(root / version)
        runs.append(run); all_rows.append(rows)
    review = dict(schema='prove.libero-graph-root-review-draft.v1', status='draft_for_root_review',
                  runs=runs, engineering_versions_used=2, engineering_versions_unused=1,
                  further_physics_authorized=False, full_task_success=False, adapter_executed=False,
                  evidence_scope='opened-development same-initial-state engineering; no independent generalization evidence',
                  interpretation=['v01 Lift budget stop while attached and physically progressing',
                                  'v02 budget-only revision passes Lift and Align, stops Insert with no progress',
                                  'v02 waypoint204 EEF residual remains just above fixed tolerance; recorded book/caddy contact',
                                  'contact-constrained tracking is a diagnosis hypothesis, not a proven single cause'],
                  final_neural_policy_requirement='not fulfilled by these fixed demo engineering controllers')
    dump(root / 'ROOT_REVIEW_DRAFT.json', review)
    lines=['# P49 工程执行审阅草稿', '', '两次同初态真实Task Graph执行均未完成任务，官方目标均未成立。第三个工程版本未使用；停止物理追加。', '',
           '控制为单条示范派生的固定EEF位姿反馈，实时只读取机器人自身位姿；GT仅由独立监督侧判断阶段。没有执行Fast Adapter或神经动作模型，也不是原始动作序列回放。', '',
           '| 版本 | 阶段 | 动作数 / 阶段预算 | 完成帧 | 状态 |', '|---|---|---:|---:|---|']
    for run in runs:
        for stage in run['stages']:
            lines.append(f'| {run["version"]} | {stage["stage"]} {LABELS[stage["skill"]]} | {stage["actions"]} / {stage["step_budget"]} | {stage["completion_frame"] if stage["completion_frame"] is not None else "—"} | {stage["final_state"]} |')
    lines += ['', 'v01执行183动作，Lift达到原51动作预算时仍双指持物并上升，底部净空0.116002米。v02只将所有阶段动作预算翻倍，谓词、停滞预算、控制参数、校准、初态和700总动作预算不变。', '',
              'v02执行429动作，完成至Align，在Insert反复无进展而停止。最后控制参考点为204，EEF位置残差6.28755毫米，高于冻结容差6.11144毫米；仍保持双指持物和对齐，记录到书本与收纳盒接触。该证据提示接触约束下的轨迹跟踪停滞，不能当作唯一根因已经证实。没有放宽完成谓词或追加容差搜索。', '',
              'v01全部184帧、v02全部430帧的文件摘要通过；183和429个动作均由公开EEF状态及冻结配置精确复算，最大绝对误差为0。全部相邻物理状态发生变化。该审计不包含重新进行物理积分。', '',
              '这只支持真实分阶段控制与失败停止的工程证据，不支持完整任务成功、新初态/物体泛化或Adapter收益。用户指定的简单阶段API控制、精确阶段离线训练小型动作模型、部署期冻结并由Adapter适配的最终架构尚未由本工程控制器实现。', '',
              '[v02完整双视角注释视频](engineering_v02_zh.mp4) · [机器可读审阅草稿](ROOT_REVIEW_DRAFT.json) · [监督离线检查](SUPERVISOR_OFFLINE_REPORT.json)', '']
    (root / 'ROOT_REVIEW_DRAFT_ZH.md').write_text('\n'.join(lines))
    video = render(root / 'engineering_v02', all_rows[-1], root / 'engineering_v02_zh.mp4')
    (root / 'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>P49 工程执行审阅</title><style>body{max-width:1080px;margin:40px auto;padding:0 20px;font:18px/1.7 sans-serif;color:#233}video{width:100%}a{color:#1762a3}</style><h1>P49 真实任务图执行：工程失败记录</h1><p>两次执行均未完成任务；v02到达放入阶段后因反复无进展停止。固定one-shot技能，非神经动作模型，未调用Fast Adapter；同一已知初态，不是独立泛化成绩。</p><video controls preload="metadata" src="engineering_v02_zh.mp4"></video><p>完整430帧，双视角，20fps；只做显示翻转和放大，没有生成场景或补帧。</p><ul><li><a href="ROOT_REVIEW_DRAFT_ZH.md">中文审阅草稿与阶段表</a></li><li><a href="ROOT_REVIEW_DRAFT.json">结构化审阅草稿</a></li><li><a href="engineering_v01/RESULT.json">v01结果</a> / <a href="engineering_v02/RESULT.json">v02结果</a></li><li><a href="engineering_v01/TRACE_AUDIT.json">v01审计</a> / <a href="engineering_v02/TRACE_AUDIT.json">v02审计</a></li><li><a href="SUPERVISOR_OFFLINE_REPORT.json">监督判断依据</a></li><li><a href="PLAN.md">输入边界与Adapter接入门</a></li><li><a href="engineering_v02_zh.manifest.json">视频来源与完整性</a></li></ul><p>尚缺：精确阶段的小型神经动作模型与真实可见几何输入接入；本视频不证明Adapter已经接入。</p></html>''')
    return dict(review=str(root/'ROOT_REVIEW_DRAFT.json'), video=video)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args=parser.parse_args()
    print(json.dumps(summarize(args.root), ensure_ascii=False, indent=2))
