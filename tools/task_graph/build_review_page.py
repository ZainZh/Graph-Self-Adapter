"""Build a small offline review page; no result inference or Dashboard mutation."""
from __future__ import annotations
import argparse
import html
import json
import os
from pathlib import Path

from .inspect_demo import sha256_file


def build_page(root: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f'preserve existing review page: {output}')
    def link(path: Path, title: str) -> str:
        if not path.is_file():
            return html.escape(title) + '（尚未生成）'
        return f'<a href="{html.escape(os.path.relpath(path, output.parent))}">{html.escape(title)}</a>'
    video = root / 'demo_review_v03/ORIGINAL_DEMO_ZH_PROVISIONAL.mp4'
    graph = root / 'graph_v01/TASK_GRAPH.json'
    report = root / 'ROOT_REPORT.json'
    rows = [('P1 演示→阶段', '主Agent离线审阅证据并提出分段；非自动解析验证'),
            ('P2 Task Graph', '单元测试；尚未验证完整任务执行'),
            ('P3 演示→机器人桥接', '存在兼容性缺口；等待实际采集器结果'),
            ('P4 阶段调度', '单元测试；尚无在线闭环证据'),
            ('P5 自动拆解→执行', '自动链路尚未验证')]
    report_body = '<p>根报告尚未生成；本页不报告最终结果。</p>'
    if report.is_file():
        facts = json.loads(report.read_text())
        rows = [(module['id'] + ' ' + module['name'], module['state']) for module in facts['modules']]
        report_body = '<p>' + link(report, '根报告（以其事实为准）') + '</p>'
        teacher = facts.get('teacher_capture', {})
        if teacher.get('live_final_goal') is True:
            report_body += f'<p>独立于原演示视频的一次教师动作物理回放：{teacher.get("replay_actions")}个动作，{teacher.get("captured_frames")}个记录帧，最终目标谓词为真。此结果来自本次GT回放，不是原视频的终点标记，也不是Task Graph自主执行或Fast Adapter成绩。</p>'
        report_body += '<details><summary>根报告原始事实</summary><pre>' + html.escape(json.dumps(facts, ensure_ascii=False, indent=2)) + '</pre></details>'
    links = [link(report, '根报告 JSON'), link(root / 'ROOT_REPORT_ZH.md', '中文根报告'),
             link(root / 'gt_replay_v01/RESULT.json', '单次GT教师回放记录'),
             link(root / 'gt_replay_review_v01/DIAGNOSTIC.json', '原演示与GT回放诊断'),
             link(root / 'gt_replay_review_v01/CONTACT_SHEET.png', 'GT回放与原演示对照图'),
             link(graph, 'Task Graph JSON'), link(root / 'demo_review_v03/ROOT_PROVISIONAL_ANNOTATIONS.json', '主Agent提出的阶段标注'),
             link(root / 'demo_review_v01/INVENTORY.json', '演示字段与来源'),
             link(Path('data/sources/libero_one_shot_p48/ACQUISITION.json').resolve(), '官方来源版本与哈希'),
             link(video.with_suffix('.manifest.json'), '视频证据清单')]
    body = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>P48 单条演示与任务图审阅</title><style>body{max-width:1120px;margin:36px auto;font:18px/1.6 sans-serif;color:#17212c;background:#f5f7f9}h1{font-size:30px}video{width:1024px;background:#111}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border:1px solid #ccd4dc;text-align:left}a{color:#075aa8}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.note{padding:16px;background:#e7edf4}</style><h1>P48：官方单条演示 → 暂定任务图</h1>'
    body += '<p>LIBERO-10：抓起书本并放入收纳盒后格。官方训练演示 demo_0，234步，原始双视角各128×128。</p><p class="note">官方单条演示；未调用Fast Adapter。阶段由主Agent离线审阅证据并提出，尚未形成经过验证的自动闭环。源终点标记不能作为物理成功依据。</p>'
    if video.is_file():
        body += '<video controls preload="metadata" src="' + html.escape(os.path.relpath(video, output.parent)) + '"></video>'
    body += '<p>暂定阶段：移动 → 接近 → 抓取 → 抬起 → 移动 → 对齐 → 放入后格。演示未显示松爪；“放入”不表示已具备销孔插入技能。</p><h2>管线状态</h2><table><tr><th>环节</th><th>当前证据边界</th></tr>'
    body += ''.join('<tr><td>'+html.escape(a)+'</td><td>'+html.escape(b)+'</td></tr>' for a,b in rows)
    body += '</table><h2>来源与复核</h2><ul>' + ''.join('<li>'+item+'</li>' for item in links) + '</ul>' + report_body + '</html>'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body)
    sources = [report, root / 'ROOT_REPORT_ZH.md', graph, video, root / 'gt_replay_review_v01/DIAGNOSTIC.json']
    manifest = {'format': 'prove.p48-review-page.v1', 'sources': {str(path): sha256_file(path) for path in sources if path.is_file()}, 'page_sha256': sha256_file(output)}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(build_page(args.root.resolve(), args.output.resolve()))


if __name__ == '__main__':
    main()
