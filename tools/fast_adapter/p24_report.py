"""Report every K and seed; no checkpoint or interaction-count selection."""
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tools.fast_adapter.p24_common import R, KS, SEEDS, load_adapter
from tools.fast_adapter.p18_run import save_json, digest

NAMES = {'id': '同分布', 'position': '槽位置变化', 'scale': '整体尺度变化'}


def paired(a, b):
    d = np.asarray(a, float)-np.asarray(b, float)
    if d.ndim == 1: d = d[None]
    rng = np.random.default_rng(240224)
    si = rng.integers(d.shape[0], size=(5000, d.shape[0]))
    di = rng.integers(d.shape[1], size=(5000, d.shape[1]))
    boot = d[si[:, :, None], di[:, None, :]].mean((1, 2))*100
    return {'delta_pp': float(d.mean()*100),
            'ci95_pp': np.quantile(boot, [.025, .975]).tolist(),
            'two_primary_endpoint_ci_pp': np.quantile(boot, [.0125, .9875]).tolist(),
            'seed_deltas_pp': (d.mean(1)*100).tolist(),
            'wins': int((d > 0).sum()), 'harms': int((d < 0).sum()),
            'unique_win_conditions': np.flatnonzero((d > 0).any(0)).tolist(),
            'unique_harm_conditions': np.flatnonzero((d < 0).any(0)).tolist(),
            'all_paired_outcomes_identical': bool(np.all(d == 0))}


def main():
    torch.set_num_threads(2); torch.backends.cudnn.enabled = False
    assert (R/'CAMPAIGN_DONE.json').exists()
    out = R/'summary'; out.mkdir(exist_ok=True)
    results = {}; bindings = []
    for st in NAMES:
        roots = [R/'final'/st/f'seed{s}' for s in (0, *SEEDS)]
        rows = [json.loads((p/'RESULT.json').read_text()) for p in roots]
        bindings += [{'path': str(p/'RESULT.json'), 'sha256': digest(p/'RESULT.json')} for p in roots]
        arrays = {arm: np.array(rows[0]['metrics'][arm]['success'], bool) for arm in ('base', 'oracle')}
        arms = [f'{name}_k{k}' for name in ('meta', 'history') for k in KS]+['swapped_k3']
        for arm in arms: arrays[arm] = np.array([r['metrics'][arm]['success'] for r in rows[1:]], bool)
        comparisons = {f'meta_k{k}_minus_{ref}': paired(arrays[f'meta_k{k}'], arrays[ref])
                       for k in KS for ref in ('meta_k0', f'history_k{k}', 'base')}
        comparisons['meta_k3_minus_swapped_k3'] = paired(arrays['meta_k3'], arrays['swapped_k3'])
        reference = torch.load(roots[0]/'base_RAW.pt', weights_only=False)
        diagnostics = []
        for seed, root in zip(SEEDS, roots[1:]):
            net = load_adapter(seed, device='cpu')
            trace = torch.load(root/'SUPPORT_SEALED.pt', weights_only=True)
            v = reference['visual'].float(); p = torch.tensor(reference['positions'], dtype=torch.float32)
            a = torch.tensor(reference['actions'], dtype=torch.float32)
            previous = torch.cat((torch.zeros_like(a[:, :1]), a[:, :-1]), 1)
            with torch.no_grad():
                u = net.encode(v[:, :-1], p[:, :-1], previous)
                next_u = net.encode(v[:, 1:], p[:, 1:], a)
            first0 = torch.load(root/'meta_k0_RAW.pt', weights_only=False)['actions'][:, 0]
            per_k = {}
            for k in KS:
                z = trace['latents'][f'meta_k{k}']
                with torch.no_grad():
                    pred = net.predict(u, a, z[:, None, :].expand(-1, 48, -1))
                    error = (pred-next_u).square().mean()
                raw = torch.load(root/f'meta_k{k}_RAW.pt', weights_only=False)
                per_k[str(k)] = {'z_rms_from_prior': float((z-trace['latents']['meta_k0']).square().mean().sqrt()),
                    'first_action_rms_from_k0': float(np.sqrt(np.mean((raw['actions'][:, 0]-first0)**2))),
                    'frozen_base_query_next_u_mse': float(error)}
            diagnostics.append({'seed': seed, 'alpha': float(net.alpha), 'per_k': per_k,
                'support_transition_loss_before': trace['per_transition_loss'][:, :, 0].mean(0).tolist(),
                'support_transition_loss_after': trace['per_transition_loss'][:, :, 1].mean(0).tolist()})
        results[st] = {'rates_percent': {a: float(v.mean()*100) for a, v in arrays.items()},
            'seed_rates_percent': {a: (v.mean(1)*100).tolist() for a, v in arrays.items() if v.ndim == 2},
            'independent_conditions': len(arrays['base']), 'comparisons': comparisons,
            'diagnostics': diagnostics}
    training = {str(s): json.loads((R/'models'/f'seed{s}/COMPLETE.json').read_text()) for s in SEEDS}
    save_json(out/'RESULTS.json', {'strata': results, 'K': list(KS), 'training': training,
        'source_bindings': bindings,
        'primary': 'K3 minus same-checkpoint K0, separately position and scale; adjusted97.5% marginal intervals',
        'interpretation': 'K is both observed-transition count and causal one-update-per-transition count; not pure optimization-count ablation. Shared meta-policy support controls information, not exploration-policy performance.',
        'diagnostic_boundary': 'Base-query next-U errors measured post-seal; never used for latent updates, selection or tuning.',
        'uncertainty': '5000 paired bootstrap over independent condition and model-seed axes. One fixed base checkpoint. Degenerate intervals from identical outcomes do not imply zero population effect.'})
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, (st, name) in zip(axes, NAMES.items()):
        r = results[st]['rates_percent']
        ax.plot(KS, [r[f'meta_k{k}'] for k in KS], '-o', color='#176b69', label='共同初值＋梯度适配')
        ax.plot(KS, [r[f'history_k{k}'] for k in KS], '--s', color='#c07434', label='同预算 History')
        ax.axhline(r['base'], color='#718096', ls=':', label='冻结视觉基座')
        ax.set_xticks(KS); ax.set_ylim(0, 105); ax.grid(alpha=.2)
        ax.set_title(name, fontproperties=font, fontsize=14)
        ax.set_xlabel('已观察反馈数 K（每次更新一次 z）', fontproperties=font)
    axes[0].set_ylabel('成功率（%）', fontproperties=font)
    axes[-1].legend(prop=font, loc='upper right')
    fig.suptitle('P24：共同初值、五万次元训练、冻结 LiLa 和基础技能', fontproperties=font, fontsize=16)
    fig.text(.08, .015, '每类64个全新条件，三模型种子的平均。K0 是同模型不更新 z；所有 K 均完整报告。', fontproperties=font)
    fig.tight_layout(rect=(0, .05, 1, .94)); fig.savefig(out/'success_vs_interactions.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for seed in SEEDS:
        records = [json.loads(x) for x in (R/'models'/f'seed{seed}/TRAIN.jsonl').read_text().splitlines()]
        steps = [x['step'] for x in records]
        axes[0].plot(steps, [x['meta']['action_loss'] for x in records], label=f'Meta seed{seed}')
        axes[0].plot(steps, [x['history']['action_loss'] for x in records], ls='--', alpha=.55)
        axes[1].plot(steps, [x['meta']['alpha'] for x in records], label=f'seed{seed}')
        axes[2].plot(steps, [x['meta']['representation_std'] for x in records], label=f'seed{seed}')
    axes[0].set_yscale('log'); axes[0].legend(fontsize=8)
    for ax, title in zip(axes, ['动作监督误差（虚线为 History）', '学习到的 inner 学习率 α', '交互表示在样本间的平均标准差']):
        ax.set_title(title, fontproperties=font); ax.set_xlabel('外层训练次数', fontproperties=font); ax.grid(alpha=.2)
        for step in (10000, 30000): ax.axvline(step, color='#9ca3af', alpha=.5, ls=':')
    fig.tight_layout(); fig.savefig(out/'training_curves.png', dpi=180); plt.close(fig)
    lines = ['# P24：共同初值的 Fast Adapter 重训', '',
        'LiLa、DINOv3、PCA和视觉基座完全冻结。新P/A/D、共同z0和alpha训练50000次；每条反馈引起一次z更新，训练随机K=1/2/3/5。History使用同数据和同外层次数。', '',
        '| 方法 | 同分布 | 槽位置变化 | 整体尺度变化 |', '|---|---:|---:|---:|']
    for arm in ['base']+[f'meta_k{k}' for k in KS]+[f'history_k{k}' for k in KS]+['swapped_k3', 'oracle']:
        lines.append('| '+arm+' | '+' | '.join(f'{results[st]["rates_percent"][arm]:.2f}%' for st in NAMES)+' |')
    lines += ['', '每类64个独立环境条件，三模型种子共享条件。每方法192次模型×条件执行不是192个独立环境。', '',
        '主比较为K3−K0；两档OOD分别检验，使用97.5%边际区间校正。所有K及种子完整保留。']
    for st, name in NAMES.items():
        c = results[st]['comparisons']['meta_k3_minus_meta_k0']
        h = results[st]['comparisons']['meta_k3_minus_history_k3']
        lines += [f'- {name}：K3−K0 {c["delta_pp"]:+.2f}pp，95%区间{c["ci95_pp"]}，校正区间{c["two_primary_endpoint_ci_pp"]}；K3−History {h["delta_pp"]:+.2f}pp，95%区间{h["ci95_pp"]}。']
    lines += ['', '![成功率](success_vs_interactions.png)', '', '![训练曲线](training_curves.png)', '',
        '训练起点为384个既有训练环境条件；每种子在1万和3万步各新增384条当前meta策略的物理轨迹，更新动作数据而不是扩大几何条件数量。两种方法读取相同回放；History参数更多，未声称参数量匹配。', '',
        'Outer loss采用专家动作监督和下一步U预测辅助误差，没有反传模拟器终点成功率。支持策略由meta方法生成，因此查询对照共享信息，但不是独立探索策略的效率比较。', '',
        '测试时仅z更新，查询前封存，48步查询内固定。未使用槽坐标、尺寸或GT框；物块XYZ仍是允许的本体输入。模型和所有最终结果冻结后不再调参。', '',
        '正式解释与完整审计见ROOT_ACCEPTANCE.json和AUDIT.json。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({st: {'rates': x['rates_percent'], 'k3_minus_k0': x['comparisons']['meta_k3_minus_meta_k0'],
                            'k3_minus_history': x['comparisons']['meta_k3_minus_history_k3']} for st, x in results.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
