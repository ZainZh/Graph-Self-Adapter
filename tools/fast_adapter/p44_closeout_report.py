"""Build the local P44 closeout from recorded JSON/media, with standard library only.

No simulator, neural library, shell process, training or rollout is imported or
started. Historical artifacts are read only; this writes three derived summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path


DEFAULT_ROOT = Path('evaluations/fast_adapter_p44_robot_geometry_v01')
ARMS = ('fixed', 'continuous')
SPLITS = {'id': 'ID（训练范围内）', 'geometry': 'OOD（几何范围外）'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    return dict(path=str(path), sha256=sha(path))


def identity(row):
    return row['task'], row['split'], row['seed'], row['model_seed'], row['arm']


def load_result(path):
    row = read(path)
    cell = path.parent
    pre, seal = read(cell / 'PRE_RUN.json'), read(cell / 'SEAL.json')
    if sha(cell / 'PRE_RUN.json') != seal['pre_run']:
        raise ValueError(f'PRE_RUN seal mismatch: {cell}')
    if row.get('neural_weights_unchanged') is not True:
        raise ValueError(f'Frozen-weight confirmation missing: {cell}')
    cfg = pre['cfg']
    if ((row['task'], row['split'], row['seed'], row['arm'], row['model_seed'])
            != (cfg['task'], cfg['split'], cfg['seed'], pre['arm'], pre['bindings']['model_seed'])):
        raise ValueError(f'Result identity differs from sealed run: {cell}')
    return dict(row, cell=str(cell), evidence={name: binding(cell / name)
                for name in ('RESULT.json', 'PRE_RUN.json', 'SEAL.json')})


def require_exact(rows, expected, label):
    keys = [identity(row) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f'{label}: incomplete/extra/duplicate results; expected {len(expected)}, found {len(keys)}')


def split_counts(rows, split):
    selected = [row for row in rows if row['split'] == split]
    keys = sorted({(row['seed'], row['model_seed']) for row in selected})
    lookup = {(r['seed'], r['model_seed'], r['arm']): r for r in selected}
    paired = []
    for seed, model_seed in keys:
        fixed, continuous = (bool(lookup[seed, model_seed, arm]['success']) for arm in ARMS)
        category = ('both_pass' if fixed and continuous else 'both_fail'
                    if not fixed and not continuous else 'rescue' if continuous else 'harm')
        paired.append(dict(seed=seed, model_seed=model_seed, fixed=fixed,
                           continuous=continuous, category=category))
    return dict(split=split, independent_geometry_conditions=len({r['seed'] for r in selected}),
                model_seeds=sorted({r['model_seed'] for r in selected}),
                executions=len(selected), model_condition_pairs=len(paired),
                counts={arm: dict(successes=sum(bool(r['success']) for r in selected if r['arm'] == arm),
                                  total=sum(r['arm'] == arm for r in selected)) for arm in ARMS},
                paired_counts={name: sum(row['category'] == category for row in paired)
                               for name, category in [('rescues', 'rescue'), ('harms', 'harm'),
                                                      ('both_pass', 'both_pass'), ('both_fail', 'both_fail')]},
                paired_outcomes=paired)


def fraction(row, arm):
    counts = row['counts'][arm]
    return f'{counts["successes"]}/{counts["total"]}'


def href(path, output):
    return html.escape(os.path.relpath(path, output), quote=True)


def video_section(path, output, title, note, poster=None):
    safe_title, safe_note = html.escape(title), html.escape(note)
    if not path.is_file():
        return f'<section><h2>{safe_title}</h2><p class="notice">视频尚未回传。</p><p>{safe_note}</p></section>'
    poster_attr = f' poster="{href(poster, output)}"' if poster and poster.is_file() else ''
    return (f'<section><h2>{safe_title}</h2><p class="notice">{safe_note}</p>'
            f'<video controls preload="metadata"{poster_attr} src="{href(path, output)}"></video></section>')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--v05-results', type=Path, help='Optional exact directory containing repaired cup RESULT files')
    parser.add_argument('--v05-comparison', type=Path, help='Existing repaired-cup comparison index.html')
    args = parser.parse_args()
    root = args.root
    output = args.out or root / 'summary'
    freeze_path, amendment_path = root / 'PILOT_FREEZE.json', root / 'REPAIR_AMENDMENT.json'
    freeze, amendment = read(freeze_path), read(amendment_path)
    original = [load_result(p) for p in sorted((root / 'development').glob('*/*/RESULT.json'))]
    if any(row['scope'] != 'development_embodiment_transfer' for row in original):
        raise ValueError('Original pilot contains a different evidence scope')
    insert = [row for row in original if row['task'] == 'insert']
    invalid_cup = [row for row in original if row['task'] == 'cup']
    cfg = freeze['tasks']['insert']
    require_exact(insert, {('insert', split, seed, model, arm)
                           for seed, split in zip(cfg['seeds'], cfg['splits'])
                           for model in freeze['model_seeds'] for arm in ARMS}, 'Original insertion')
    repair_scope = amendment['new_cup_scope']
    cup_expected = {('cup', condition['split'], condition['seed'], model, arm)
                    for condition in repair_scope['conditions']
                    for model in repair_scope['model_seeds'] for arm in repair_scope['arms']}
    require_exact(invalid_cup, cup_expected, 'Excluded original cup')
    search_root = args.v05_results or root
    repaired = []
    for path in sorted(search_root.rglob('RESULT.json')):
        record = read(path)
        if record.get('scope') == 'development_embodiment_repair_recheck':
            row = load_result(path)
            pre = read(path.parent / 'PRE_RUN.json')
            if 'prove/fast_adapter/robot_geometry_env_v05.py' not in pre['sources']:
                raise ValueError(f'Repair result does not bind the v05 backend: {path}')
            repaired.append(row)
    require_exact(repaired, cup_expected, 'v05 repaired cup recheck')
    if not repair_scope.get('opened_conditions'):
        raise ValueError('Repaired cup conditions must remain labelled already opened')
    insert_splits = [split_counts(insert, split) for split in SPLITS]
    cup_splits = [split_counts(repaired, split) for split in SPLITS]
    engineering_review = root / 'engineering/V05_MECHANICAL_REPAIR_REVIEW.json'
    if read(engineering_review).get('passed') is not True:
        raise ValueError('v05 engineering repair review is not passed')

    oracle_video = root / 'engineering/render_v05_cup_id440000/annotated.mp4'
    rescue_video = output / 'insert_rescue_reconstruction/insert_rescue.mp4'
    oracle_note = 'Oracle 工程演示：读取目标真值验证真实夹爪接触和抬升能力；不计入神经模型成绩。'
    rescue_note = '封存状态重建，非原始 RGB；事后按救回结果选样。没有新增物理 rollout 或神经推理，不代替整批统计。'
    media = [dict(role='oracle_engineering_demo', path=str(oracle_video), note=oracle_note),
             dict(role='posthoc_sealed_state_reconstruction', path=str(rescue_video), note=rescue_note)]
    for entry in media:
        path = Path(entry['path'])
        entry['available'] = path.is_file()
        if path.is_file():
            entry['sha256'] = sha(path)
    reconstruction_manifest = rescue_video.parent / 'MANIFEST.json'
    if rescue_video.is_file():
        manifest = read(reconstruction_manifest)
        if (manifest.get('video_sha256') != sha(rescue_video)
                or manifest.get('original_rgb') is not False
                or manifest.get('physics_steps_executed') != 0
                or manifest.get('network_calls') != 0):
            raise ValueError('Reconstruction provenance does not support its presentation label')
    repair_media_path = root / 'summary/repair_v05/MEDIA_MANIFEST.json'
    repair_media = read(repair_media_path)['videos']
    for entry in repair_media:
        if sha(entry['video']) != entry['video_sha256']:
            raise ValueError('Repaired cup comparison video changed')
    repair_comparison_href = href(args.v05_comparison, output) if args.v05_comparison else '#cup-v05'

    original_decisions_path = root / 'ALL_COMPLETED_DECISION_AUDIT.json'
    original_physics_path = root / 'FIRST_PAIR_REPLAY_AUDIT.json'
    repair_audit_path = root / 'REPAIR_V05_REPLAY_AUDIT.json'
    original_decisions, original_physics, repair_audit = (
        read(path) for path in (original_decisions_path, original_physics_path, repair_audit_path))
    result_hashes = {row['cell']: row['evidence']['RESULT.json']['sha256'] for row in original + repaired}
    for audit in original_decisions['rows'] + original_physics + repair_audit['records']:
        if audit['result_sha256'] != result_hashes.get(audit['cell']):
            raise ValueError('Audit receipt no longer binds the recorded result')
    exceptions = [row for row in original_physics if not row['passed']]
    if (not original_decisions['passed'] or not repair_audit['all_passed']
            or any(row['cell'] not in {r['cell'] for r in invalid_cup} for row in exceptions)):
        raise ValueError('An audit failure affects the admitted comparison set')
    verification = dict(original_decision_replays=original_decisions['cells'],
                        original_decisions_passed=original_decisions['passed'],
                        original_insert_full_physics_replays=sum('/insert_' in row['cell'] for row in original_physics),
                        repaired_full_physics_replays=len(repair_audit['records']),
                        repaired_maximum_error=max(max(row['maximum_error'].values()) for row in repair_audit['records']),
                        excluded_original_cup_physics_exceptions=exceptions)
    acceptance_path = root / 'summary/ROOT_ACCEPTANCE.json'
    mechanical_review = None
    warning_text = '全轨迹机械资格由主 Agent 验收文件记录；验收尚未生成时，不默认所有执行均无机械异常。'
    if acceptance_path.is_file():
        acceptance = read(acceptance_path)
        inventory = {row['cell']: row for row in acceptance['inventory']}
        trajectory_limits = {row['cell']: row for row in acceptance['mechanical_trajectory_limits']}
        for cell, row in inventory.items():
            if row['result_sha256'] != result_hashes.get(cell):
                raise ValueError('Root acceptance no longer binds the recorded result')
        id_warnings = [trajectory_limits[row['cell']] for row in insert
                       if row['split'] == 'id' and trajectory_limits[row['cell']]['mechanical_warning']]
        ood_warning_count = sum(trajectory_limits[row['cell']]['mechanical_warning']
                                for row in insert if row['split'] == 'geometry')
        peaks = '、'.join(f'{1000 * row["finger_joint_limit_violation_m"]:.3f}' for row in id_warnings)
        warning_text = (f'插入 ID 中 {len(id_warnings)} 个在线失败伴随瞬时手指软限位异常（{peaks} mm）；'
                        '保留完整 0/4 对 0/4 分母，不能把这些失败当作干净的适配负结果。'
                        f'插入 OOD 的同类异常为 {ood_warning_count} 个，救回样例不受此问题影响。')
        for split_row in insert_splits:
            cells = [row['cell'] for row in insert if row['split'] == split_row['split']]
            split_row['mechanical_warning_cells'] = [cell for cell in cells if trajectory_limits[cell]['mechanical_warning']]
            split_row['full_denominator_preserved'] = True
        mechanical_review = dict(source=binding(acceptance_path),
                                 inventory_pointer='/inventory', limits_pointer='/mechanical_trajectory_limits',
                                 insertion_id_warning_pointer='/insertion_id_warning',
                                 insertion_id_warning_cells=[row['cell'] for row in id_warnings],
                                 insertion_ood_warning_count=ood_warning_count,
                                 v05_max_finger_overtravel_mm=1000 * max(trajectory_limits[row['cell']]['finger_joint_limit_violation_m'] for row in repaired),
                                 mechanically_unflagged_episodes=acceptance['mechanically_unflagged_episodes'],
                                 full_report_denominator_unchanged=True)

    evidence = [binding(path) for path in (freeze_path, amendment_path, engineering_review,
                root / 'REPAIR_FREEZE_V05.json', repair_media_path, original_decisions_path,
                original_physics_path, repair_audit_path)]
    for path in (root / 'summary/original_pilot/REPORT.json', reconstruction_manifest,
                 root / 'summary/repair_v05/REPORT.json', acceptance_path):
        if path.is_file():
            evidence.append(binding(path))
    summary = dict(schema='prove.p44-closeout.v1', scope='simulation_development_only',
                   result_set_complete=True, root_acceptance_owned_separately=True,
                   total_neural_executions=len(original) + len(repaired),
                   full_report_execution_count=len(insert) + len(repaired),
                   original_insert=dict(version='original_frozen_backend', executions=len(insert),
                                        split_summary=insert_splits, results=insert),
                   repaired_cup=dict(version='v05', executions=len(repaired), conditions_already_opened=True,
                                     fresh_held_out=False, split_summary=cup_splits, results=repaired),
                   excluded_original_cup=dict(executions=len(invalid_cup),
                       cancelled_remaining=amendment['original_pilot']['cancelled_cup_remainder'],
                       reason=amendment['original_pilot']['reason'], included_in_effect_comparison=False,
                       results=invalid_cup),
                   statistics='Descriptive only. Model repeats do not increase independent geometry-condition count.',
                   limits=['Simulation, not hardware', 'Insertion assumes a secure initial grasp',
                           'P43 checkpoint/PCA transferred without retraining',
                           'Cup engineering Oracle is not neural success evidence',
                           'v05 cup reuses opened conditions; no fresh held-out claim',
                           'ID insertion failures retained; no general adaptation success claim',
                           'No isolated DINO contribution or complete RSI claim'],
                   media=media, repaired_cup_media=repair_media, verification=verification,
                   mechanical_review=mechanical_review,
                   source_bindings=evidence, generator=binding(Path(__file__)),
                   operations=dict(new_rollouts=0, neural_inference=0, simulation_steps=0,
                                   original_artifacts_modified=False))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    rows = [('插入（原版）', row) for row in insert_splits] + [('杯柄（v05）', row) for row in cup_splits]
    id_row, ood_row = insert_splits
    report = ['# P44：完整 Panda 机械臂抓杯与插入', '',
              f'插入几何 OOD：固定 z {fraction(ood_row, "fixed")}，持续更新 z {fraction(ood_row, "continuous")}；'
              f'ID：固定 z {fraction(id_row, "fixed")}，持续更新 z {fraction(id_row, "continuous")}。', '',
              warning_text, '',
              '杯柄已在修复后的真实夹爪上完成 Oracle 工程抓持；冻结神经模型的 v05 开发复查仍未达到抓取成功判据。'
              if not any(r['success'] for r in repaired) else
              '杯柄 v05 开发复查的实际神经结果见下表；Oracle 工程展示单独标记。', '',
              '这是仿真开发结果。插入每档只有 2 个几何条件 × 2 个固定模型；4 个模型—条件配对不是 4 个独立物体。'
              '杯柄 v05 每档仅 1 个已开放条件 × 模型种子 11，不是 fresh 测试。', '',
              '| 任务 / 分布 | 几何条件 × 模型 | 固定 z | 持续更新 z | 救回 | 退步 | 双失败 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for label, row in rows:
        counts = row['paired_counts']
        report.append(f'| {label} / {SPLITS[row["split"]]} | {row["independent_geometry_conditions"]} × {len(row["model_seeds"])} | '
                      f'{fraction(row, "fixed")} | {fraction(row, "continuous")} | {counts["rescues"]} | {counts["harms"]} | {counts["both_fail"]} |')
    report += ['', '救回与退步按同条件、同模型配对计算；这里只描述小样本计数，不作统计显著性或普遍泛化主张。', '',
               f'原版杯柄 {len(invalid_cup)} 次执行因指部内部碰撞及软限位失效而工程污染，另行封存、排除有效方法比较；'
               f'原定其余 {amendment["original_pilot"]["cancelled_cup_remainder"]} 次取消。v05 修复复查未改旧结果。', '',
               f'[真实夹爪抓杯工程视频]({os.path.relpath(oracle_video, output)})：{oracle_note}', '',
               f'[插入救回展示]({os.path.relpath(rescue_video, output)})：{rescue_note}', '',
               '限制：插入初始已牢固持物；两任务均使用 P43 已冻结模型和 PCA，未重训。'
               '工程可达性不证明模型抓取成功；OOD 的局部救回不能掩盖 ID 失败。'
               '没有单独识别 DINO 的贡献，也不证明完整 RSI 或真实硬件能力。', '',
               f'复算：原版 {original_decisions["cells"]} 回合神经决策一致；插入首个 ID 两臂完整物理回放通过；'
               f'v05 四回合完整物理回放通过，最大误差 {verification["repaired_maximum_error"]:.3g}。'
               f'原版已排除杯柄中保留 {len(exceptions)} 个物理回放超阈值例外，详情见 SUMMARY.json。', '',
               '[原版预定对比](original_pilot/index.html) · '
               '[v05 杯柄对比](index.html#cup-v05) · [完整机器记录](SUMMARY.json)', '']
    (output / 'ROOT_REPORT_ZH.md').write_text('\n'.join(report))

    page = ['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>P44 · Panda 机械臂抓杯与插入</title>',
            '<style>body{margin:0;background:#101823;color:#eef4fb;font:18px/1.65 system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:28px 32px 64px}h1{font-size:34px;margin-bottom:12px}h2{font-size:26px;margin:30px 0 10px}p{max-width:1400px}a{color:#8ddcc9}video{display:block;width:100%;max-height:76vh;background:#070b10;object-fit:contain;border-radius:10px}section{margin:26px 0 38px}.notice{color:#ffd397;background:#26303c;padding:12px 18px;border-radius:8px}.muted{color:#b5c8db}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:12px 15px;border-bottom:1px solid #3b4b60}th{background:#243248}.tablewrap{overflow-x:auto}.lead{font-size:23px;color:#b8efde}.links{display:flex;gap:22px;flex-wrap:wrap}</style><main>',
            '<h1>Panda 机械臂：抓杯与已持物插入</h1>',
            '<p class="lead">完整机械臂与真实夹爪已经接入。先看抓杯的工程演示，再看插入的封存结果。</p>',
            '<p class="muted">MuJoCo 仿真 · 七关节物理执行 · 插入假设开始时已牢固持物 · 开发检查</p>',
            video_section(oracle_video, output, '真实夹爪抓杯：工程演示', oracle_note,
                          oracle_video.parent / 'end.png'),
            video_section(rescue_video, output, '插入：一个固定 z 失败、在线更新救回的例子', rescue_note,
                          rescue_video.parent / 'cover.jpg'),
            '<h2>模型执行结果（保留完整分母）</h2>',
            '<p>插入每档 <strong>2 个几何条件 × 2 个固定模型</strong>；模型重复不计作更多独立物体。'
            '杯柄 v05 每档仅 <strong>1 个已开放条件 × 模型 11</strong>，属于修复后的开发复查。</p>',
            '<div class="tablewrap"><table><thead><tr><th>任务 / 分布</th><th>条件 × 模型</th><th>固定 z</th><th>持续更新 z</th><th>救回</th><th>退步</th><th>双失败</th></tr></thead><tbody>']
    for label, row in rows:
        counts = row['paired_counts']
        values = [label + ' / ' + SPLITS[row['split']],
                  f'{row["independent_geometry_conditions"]} × {len(row["model_seeds"])}',
                  fraction(row, 'fixed'), fraction(row, 'continuous'), counts['rescues'], counts['harms'], counts['both_fail']]
        page.append('<tr>' + ''.join('<td>' + html.escape(str(value)) + '</td>' for value in values) + '</tr>')
    page += ['</tbody></table></div>',
             '<p class="notice">' + html.escape(warning_text) + '</p>',
             '<p class="muted">救回：固定 z 失败而在线更新成功；退步：固定 z 成功而在线更新失败。全部失败保留；小样本计数不构成统计显著性结论。</p>',
             '<h2>保留的失败与边界</h2>',
             '<p>插入 ID 两臂均未通过；杯柄 v05 的模型复查仍未通过。Oracle 抓杯成功说明当前机构能够完成任务，不能算成神经模型的抓取成绩。</p>',
             f'<p>原版杯柄 <strong>{len(invalid_cup)} 次</strong>受指部内部碰撞与软限位失效污染，排除有效比较；其余 '
             f'<strong>{amendment["original_pilot"]["cancelled_cup_remainder"]} 次</strong>取消。旧轨迹和结果完整保留，v05 另立记录。</p>',
             '<p>这批结果只支持当前小样本插入 OOD 中的局部收益。P43 模型直接迁移没有普遍解决机械臂任务；ID 失败、杯柄失败及较小样本量限制了结论。'
             'ID 的机械异常限制了负结果归因。未隔离 DINO 的贡献，也不是完整 RSI 或真实硬件验证。</p>',
             '<p class="links"><a href="ROOT_REPORT_ZH.md">完整中文报告</a><a href="SUMMARY.json">全部结果与来源</a>'
             '<a href="original_pilot/index.html">原版预定对比</a>'
             f'<a href="{repair_comparison_href}">v05 杯柄对比</a></p>',
             f'<p class="muted">原版 {original_decisions["cells"]} 回合神经决策复算一致；插入首个 ID 双臂及 v05 四回合通过完整物理回放。'
             f'已排除的原版杯柄保留 {len(exceptions)} 个物理回放超阈值例外，详见机器记录。</p>',
             '<details id="cup-v05"><summary>查看 v05 杯柄的两个预定对比（均保留失败）</summary>']
    for entry in repair_media:
        pair = entry['pair']
        page.append(video_section(Path(entry['video']), output, f'v05 杯柄 · {SPLITS[pair["split"]]} · 条件 {pair["seed"]}',
                    '原始执行 RGB；同一已开放条件、模型 11，左侧固定 z、右侧在线更新。修复后开发复查，不是 fresh 测试。',
                    Path(entry['cover'])))
    page.append('</details>')
    optional = [(output / 'ROOT_ACCEPTANCE.json', '主 Agent 验收'),
                (output / 'RESOURCE_CLOSEOUT.json', '计算资源收尾')]
    available = [(path, label) for path, label in optional if path.is_file()]
    if available:
        page.append('<p class="links">' + ''.join(f'<a href="{href(path, output)}">{label}</a>'
                                                 for path, label in available) + '</p>')
    else:
        page.append('<p class="muted">最终验收与资源收尾文件由主 Agent 单独记录；本页不代替验收。</p>')
    page.append('</main></html>')
    (output / 'index.html').write_text('\n'.join(page))
    print(json.dumps(dict(output=str(output), original_insert=len(insert), repaired_cup=len(repaired),
                          excluded_original_cup=len(invalid_cup), new_rollouts=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
