"""Aggregate the complete P30-v2 development cohort without selecting models."""
import json
from pathlib import Path
import numpy as np
from tools.fast_adapter.p30v2_common import ROOT, TASKS, dump, digest

ARMS = ('base', 'fixed_mean', 'first3', 'gradient', 'history')
LABELS = {'base': '冻结基座', 'fixed_mean': '固定训练平均z',
          'first3': '前三步更新后固定', 'gradient': '每步持续梯度更新', 'history': '在线History'}
TASK_NAMES = {'mug-handle-lift': '杯柄抓取', 'hammer-block': '已持锤子作用木块'}
SPLITS = {'train': '同分布新物体', 'geometry_ood': '范围外几何新物体'}


def interval(delta):
    """Conditions, not repeated seed observations, are the resampling units."""
    rng = np.random.default_rng(303000)
    draws = rng.integers(len(delta), size=(20000, len(delta)))
    return (100 * np.quantile(delta[draws].mean(1), [.025, .975])).tolist()


def main():
    rows, bindings, audit_summary, times = [], [], [], []
    for task in TASKS:
        for seed in (11, 22):
            folder = ROOT/task/'development'/f'seed{seed}'
            p = folder/'RESULT.json'; ap = folder/'AUDIT.json'
            result, audit = json.loads(p.read_text()), json.loads(ap.read_text())
            assert result['phase'] == 'development' and result['network_weights_unchanged']
            assert result['episodes'] == 160 and audit['status'] == 'passed'
            assert audit['physical_states_replayed'] == 32000
            assert audit['max_action_error'] < 1e-4 and audit['max_latent_error'] < 1e-4
            # Recheck all raw hashes and equal initial physical states after transfer.
            import torch
            cache = {}
            for row in result['rows']:
                source = row['source']; key = source['path']
                if key not in cache:
                    assert digest(key) == source['sha256'], key
                    b = torch.load(key, weights_only=True)
                    cache[key] = {k: v[:, 0] for k, v in b['states'].items()}
                    times.append({'task': task, 'seed': seed, 'split': row['split'],
                                  'arm': row['arm'], 'block_step_seconds': float(b['step_seconds'].mean())})
            grouped = {}
            for row in result['rows']:
                k = (row['split'], row['condition'])
                state = {n: v[row['source_row']] for n, v in cache[row['source']['path']].items()}
                if k in grouped:
                    for n, v in state.items(): assert torch.equal(v, grouped[k][n]), (k, n)
                else: grouped[k] = state
            rows += result['rows']
            bindings += [{'path': str(x), 'sha256': digest(x)} for x in (p, ap, folder/'SOURCE_FREEZE.json')]
            audit_summary.append({'task': task, 'seed': seed, **{k: audit[k] for k in
                ('status', 'physical_states_replayed', 'max_action_error', 'max_latent_error', 'full_visual_reencoding')}})
    keys = [(r['task'], r['model_seed'], r['split'], r['condition'], r['arm']) for r in rows]
    expected = {(t, s, st, c, a) for t in TASKS for s in (11, 22)
                for st in SPLITS for c in range(16) for a in ARMS}
    assert len(rows) == 640 and len(set(keys)) == len(rows) and set(keys) == expected
    lookup = dict(zip(keys, rows)); tables, contrasts = [], []
    for task in TASKS:
        for split in SPLITS:
            arrays = {}
            for arm in ARMS:
                rr = [[lookup[(task, s, split, i, arm)] for i in range(16)] for s in (11, 22)]
                arrays[arm] = np.array([[r['success'] for r in ss] for ss in rr], dtype=float)
                terminal = np.array([[r['terminal_success'] for r in ss] for ss in rr], dtype=float)
                tables.append({'task': task, 'split': split, 'arm': arm,
                    'successes_by_seed': arrays[arm].sum(1).astype(int).tolist(),
                    'episodes_per_seed': 16, 'unique_conditions': 16,
                    'success_rate': float(arrays[arm].mean()), 'terminal_rate': float(terminal.mean()),
                    'terminal_by_seed': terminal.sum(1).astype(int).tolist(),
                    'mean_block_step_seconds': float(np.mean([v['block_step_seconds'] for v in times
                        if v['task'] == task and v['split'] == split and v['arm'] == arm]))})
            for compare in ('base', 'fixed_mean', 'first3', 'history'):
                d = arrays['gradient'] - arrays[compare]
                paired = d.mean(0)
                contrasts.append({'task': task, 'split': split, 'comparison': 'gradient-'+compare,
                    'delta_pp': float(100*d.mean()), 'condition_bootstrap_95ci_pp': interval(paired),
                    'delta_by_seed_pp': (100*d.mean(1)).tolist(),
                    'rescues_seed_condition_pairs': int((d > 0).sum()),
                    'harms_seed_condition_pairs': int((d < 0).sum()),
                    'tied_seed_condition_pairs': int((d == 0).sum()),
                    'independent_units': 16, 'scope': 'descriptive, fixed two checkpoints, no multiplicity adjustment'})
            for i in range(16):
                a, b = lookup[(task, 11, split, i, 'base')], lookup[(task, 22, split, i, 'base')]
                assert a['config'] == b['config'] and a['reset_seed'] == b['reset_seed']
    prep = [r['preparation']['steps'] for r in rows if r['preparation']]
    summary = {'schema': 'prove.p30.continuous.summary.v2', 'stage': 'development',
        'complete': True, 'physical_episodes': 640, 'unique_task_object_reset_conditions': 64,
        'model_seeds': [11, 22], 'horizon': 200, 'physical_states_replayed': 128000,
        'tables': tables, 'contrasts': contrasts, 'audits': audit_summary,
        'held_hammer_preparation': {'failed_conditions': len(list((ROOT/'initialization_failures').glob('*.json'))),
                                   'min_steps': min(prep), 'max_steps': max(prep)},
        'bindings': bindings, 'protocol': {'path': str(ROOT/'PROTOCOL.json'), 'sha256': digest(ROOT/'PROTOCOL.json')},
        'limits': ['Development evidence; no final holdout or promotion.',
                   'Two fixed trained checkpoints; repeated base is not an independent sample.',
                   'Offline chronological physical replay with privileged training labels, not on-policy meta-RL.',
                   'Procedural same-class geometry, fixed color crops, not arbitrary real object assets.',
                   'Held hammer initialization uses a privileged teacher; task policy does not read truth.',
                   'Hammer is horizontal tool contact and block displacement, not a downward nail strike.',
                   'Different arms acquire different feedback after their trajectories diverge.',
                   'Same GPU block timing is descriptive; hosts differ; no deployment latency claim.',
                   'Cup seed11 runs on local4090 and seed22 on remote3090. Same initial images/physics and identical visual weights produce non-bitwise-identical visual features. Seed contrasts are paired within host; cross-seed variation is not attributable to model seed alone.']}
    out = ROOT/'summary'; out.mkdir(exist_ok=True)
    dump(out/'SUMMARY.json', summary)
    lines = ['# P30-v2：执行中持续适配，完整开发验证', '',
             '完成两任务、两粒模型种子、五方法共640回合。每个任务每档16个新物体/初态，所有方法配对；两粒种子不会把独立物体数从16变成32。', '',
             '| 任务 | 条件 | 冻结基座 | 固定训练平均z | 前三步后固定 | 每步梯度更新 | 在线History |',
             '|---|---|---:|---:|---:|---:|---:|']
    for task in TASKS:
        for split in SPLITS:
            vals = [next(r for r in tables if (r['task'], r['split'], r['arm']) == (task, split, a)) for a in ARMS]
            cells = [f"{100*r['success_rate']:.2f}% ({'/'.join(map(str,r['successes_by_seed']))}/16)" for r in vals]
            lines.append('| '+TASK_NAMES[task]+' | '+SPLITS[split]+' | '+' | '.join(cells)+' |')
    lines += ['', '括号为种子11、22各自的成功数，不是两批独立物体。主端点为回合内达到完整任务判据；终态仍达标率保存在SUMMARY.json，视频末尾同时标注。', '',
              '## 配对差值', '', '| 任务/条件 | 每步梯度 相对 | 差值pp | 描述性95%区间 | 种子差值pp | 救回/误伤 |', '|---|---|---:|---|---|---|']
    for r in contrasts:
        lo, hi = r['condition_bootstrap_95ci_pp']; a = r['comparison'].removeprefix('gradient-')
        lines.append(f"| {TASK_NAMES[r['task']]}/{SPLITS[r['split']]} | {LABELS[a]} | {r['delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {r['delta_by_seed_pp']} | {r['rescues_seed_condition_pairs']}/{r['harms_seed_condition_pairs']} |")
    lines += ['', '区间按16个物体/初态簇重采样，保留同条件两个checkpoint结果；仅作开发描述，未作多重比较校正，不支持训练种子总体或论文确认性结论。', '',
              '## 实施与复核', '',
              '- DINOv3、LiLa、PCA、动作基座及外围网络在测试时全部冻结；只更新32维z。无机械臂动作扰动。',
              '- 每执行完一步，以最近8条已经发生的反馈更新一次z；200步回合不重置，更新后用于下一步。第200次更新不再影响已结束的控制。',
              '- 每个Gradient/History模型训练240批×8条完整200步轨迹，2400次外层更新；每20步只截断反传图，保持z值。历史对照每个反馈只输入GRU一次。',
              '- 杯子使用64训练物体的128条历史轨迹，新派生标签修正6443个已持握帧。锤子新采48条标准物体专家、32条本策略补充用于6000+3000次基座训练，然后冻结；另采64物体×2角色=128条适配训练轨迹。',
              '- 锤子从原生夹持开始，无焊接、无外力托举，任务内固定闭合夹爪；准备步骤成本单独记录。其数字不可直接与旧P29含抓取任务相减。',
              '- 所有640条动作、z与128000个物理状态的成功判据已复算；五臂初始物理状态一致、权重不变；没有重新提取所有DINO图像特征。',
              '- 结果是小型视觉策略的有界开发测试，不是π0.5、完整官方benchmark或真机。已打开数据不作新的独立测试。', '',
              '杯子跨机复核：初始物理状态和RGB逐位一致，DINO/LiLa权重哈希一致，但视觉特征和最初基座动作有数值差异；两粒模型种子分别在本机4090/服务器3090执行。所有方法在各自机器内配对，跨种子差异同时包含计算环境影响。详见 [HOST_PARITY](../mug-handle-lift/HOST_PARITY.json)。这不构成额外样本，也没有声称全视觉路径位级复现。', '',
              '代码：[连续更新](../../../prove/fast_adapter/continuous_state.py)、[网络与反馈](../../../prove/fast_adapter/continuous_adapter.py)、[完整轨迹元训练](../../../tools/fast_adapter/p30v2_train.py)。', '',
              '原始行、成对比较和来源哈希：[SUMMARY.json](SUMMARY.json)。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'tables': tables, 'contrasts': contrasts, 'audit': audit_summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
