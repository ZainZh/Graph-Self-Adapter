"""Post-seal statistics and standalone figures for every fixed P23 update count."""
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from prove.fast_adapter.visual_proprio_adapter import ProprioAdapter
from tools.fast_adapter.p18_run import save_json, digest

R = Path('evaluations/fast_adapter_p23_update_steps_v01')
NAMES = {'id': '同分布', 'position': '槽位置变化', 'scale': '整体尺度变化'}


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.ndim == 1:
        d = d[None, :]
    rng = np.random.default_rng(230223)
    si = rng.integers(d.shape[0], size=(5000, d.shape[0]))
    di = rng.integers(d.shape[1], size=(5000, d.shape[1]))
    boot = d[si[:, :, None], di[:, None, :]].mean((1, 2)) * 100
    return {'delta_pp': float(d.mean() * 100),
            'ci95_pp': np.quantile(boot, [.025, .975]).tolist(),
            'two_primary_endpoint_ci_pp': np.quantile(boot, [.0125, .9875]).tolist(),
            'wins': int((d > 0).sum()), 'harms': int((d < 0).sum()),
            'seed_deltas_pp': (d.mean(1) * 100).tolist(),
            'all_paired_outcomes_identical': bool(np.all(d == 0))}


def main():
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled = False
    assert (R / 'CAMPAIGN_DONE.json').exists()
    protocol = json.loads((R / 'PROTOCOL.json').read_text())
    steps = protocol['update_steps']
    seeds = protocol['model_seeds']
    out = R / 'summary'
    out.mkdir(exist_ok=True)
    results = {}
    bindings = []
    for st in NAMES:
        paths = [R / 'final' / st / f'seed{s}/RESULT.json' for s in [0] + seeds]
        rows = [json.loads(p.read_text()) for p in paths]
        bindings += [{'path': str(p), 'sha256': digest(p)} for p in paths]
        arrays = {arm: np.array(rows[0]['metrics'][arm]['success'], bool)
                  for arm in ('base', 'oracle')}
        for arm in ['history'] + [f'u{u}' for u in steps]:
            arrays[arm] = np.array([r['metrics'][arm]['success'] for r in rows[1:]], bool)
        supports = [torch.load(p.parent / 'SUPPORT_SEALED.pt', weights_only=True) for p in paths[1:]]
        raw_base = torch.load(paths[0].parent / 'base_RAW.pt', weights_only=False)
        raw_models = [{arm: torch.load(p.parent / f'{arm}_RAW.pt', weights_only=False)
                       for arm in ['history'] + [f'u{u}' for u in steps]} for p in paths[1:]]
        diagnostics = {}
        nets = []
        for seed in seeds:
            net = ProprioAdapter().eval().requires_grad_(False)
            net.load_state_dict(torch.load(R / 'models' / f'seed{seed}/hybrid.pt', weights_only=True)['state'])
            nets.append(net)
        for u in steps:
            arm = f'u{u}'
            query_prediction = []
            displacement_mse = []
            visual_mse = []
            for net, support in zip(nets, supports):
                # Same held-out base trajectory for every latent; diagnostic only.
                # Last query image was not cached, so use its first 47 transitions.
                v = raw_base['visual'].float()
                p = torch.tensor(raw_base['positions'][:, :48], dtype=torch.float32)
                a = torch.tensor(raw_base['actions'][:, :47], dtype=torch.float32)
                z = support['latents'][arm]
                with torch.no_grad():
                    pred = net.predict(v[:, :-1], p[:, :-1], a, z[:, None, :].expand(-1, 47, -1))
                    dp = ((pred[..., :3] - (p[:, 1:] - p[:, :-1]) / .008) ** 2).sum(-1).mean()
                    dv = ((pred[..., 3:] - (v[:, 1:] - v[:, :-1])) ** 2).mean()
                displacement_mse.append(float(dp))
                visual_mse.append(float(dv))
                query_prediction.append(float(.5 * dp + .05 * dv))
            first = np.stack([x[arm]['actions'][:, 0] for x in raw_models])
            first0 = np.stack([x['u0']['actions'][:, 0] for x in raw_models])
            first3 = np.stack([x['u3']['actions'][:, 0] for x in raw_models])
            diagnostics[arm] = {
                'support_loss': float(np.mean([r['gradient'][arm]['final_loss'] for r in rows[1:]])),
                'support_loss_by_seed': [r['gradient'][arm]['final_loss'] for r in rows[1:]],
                'z_rms_from_u0': float(torch.stack([(s['latents'][arm] - s['latents']['u0']).square().mean() for s in supports]).mean().sqrt()),
                'first_action_rms_from_u0': float(np.sqrt(np.mean((first - first0) ** 2))),
                'first_action_rms_from_u3': float(np.sqrt(np.mean((first - first3) ** 2))),
                'heldout_base_prediction_loss': float(np.mean(query_prediction)),
                'heldout_displacement_mse': float(np.mean(displacement_mse)),
                'heldout_visual_delta_mse': float(np.mean(visual_mse)),
                'adapt_batch32_seconds_by_seed': [r['gradient'][arm]['adapt_batch_seconds'] for r in rows[1:]],
            }
        results[st] = {
            'rates_percent': {k: float(v.mean() * 100) for k, v in arrays.items()},
            'seed_rates_percent': {k: (v.mean(1) * 100).tolist() for k, v in arrays.items() if v.ndim == 2},
            'independent_conditions': len(arrays['base']),
            'comparisons': {f'u{u}_minus_{ref}': paired(arrays[f'u{u}'], arrays[ref])
                            for u in steps for ref in ('u0', 'u3', 'history', 'base')},
            'diagnostics': diagnostics,
            'support_loss_curve_0_to_100': torch.cat([s['loss_curves']['u100'] for s in supports]).mean(0).tolist(),
        }
    save_json(out / 'RESULTS.json', {
        'update_steps': steps, 'strata': results, 'bindings': bindings,
        'primary': protocol['primary'], 'statistics': protocol['statistics'],
        'diagnostic_scope': 'Held-out frozen-base query transitions are opened after execution only, never used to update latent or select counts. Query PCA features cached fp16.',
        'bootstrap_caution': 'A zero-width interval from identical paired outcomes does not establish zero population effect.',
    })
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    x = np.arange(len(steps))
    for ax, (st, name) in zip(axes, NAMES.items()):
        r = results[st]
        ax.plot(x, [r['rates_percent'][f'u{u}'] for u in steps], '-o', color='#176b69', label='同一模型，改变更新次数')
        ax.axhline(r['rates_percent']['history'], color='#c07434', ls='--', label='普通 History')
        ax.axhline(r['rates_percent']['base'], color='#718096', ls=':', label='视觉基座')
        ax.set_xticks(x, steps)
        ax.set_ylim(0, 105)
        ax.set_title(name, fontproperties=font, fontsize=14)
        ax.set_xlabel('梯度更新次数（档位等距展示）', fontproperties=font)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('成功率（%）', fontproperties=font)
    axes[-1].legend(prop=font, loc='upper right', fontsize=9)
    fig.suptitle('固定模型、学习率与五次交互：增加更新次数是否改善控制？', fontproperties=font, fontsize=16)
    fig.text(.08, .018, '每类32个新条件，三个模型种子的均值。所有档位完整披露；不按结果选最好次数。', fontproperties=font)
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(out / 'success_vs_steps.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for st, name in NAMES.items():
        d = results[st]['diagnostics']
        initial = d['u0']['support_loss']
        axes[0].plot(steps, [d[f'u{u}']['support_loss'] / initial for u in steps], '-o', label=name)
        axes[1].plot(steps, [d[f'u{u}']['z_rms_from_u0'] for u in steps], '-o', label=name)
        axes[2].plot(steps, [d[f'u{u}']['first_action_rms_from_u0'] for u in steps], '-o', label=name)
    titles = ['适配损失 / 未更新时的损失', 'latent 相对 U0 的均方根变化', '首个动作相对 U0 的均方根变化']
    for ax, title in zip(axes, titles):
        ax.set_xscale('symlog', linthresh=1)
        ax.set_xticks(steps, steps)
        ax.set_title(title, fontproperties=font)
        ax.set_xlabel('梯度更新次数', fontproperties=font)
        ax.grid(alpha=.2)
    axes[0].legend(prop=font)
    fig.tight_layout()
    fig.savefig(out / 'loss_latent_action_vs_steps.png', dpi=180)
    plt.close(fig)
    lines = ['# P23：增加测试时梯度更新次数', '',
             '固定P22的模型、学习率、视觉前端、五次已完成交互和48步控制，只改变z优化次数。不重新训练。', '',
             '| 方法 | 同分布 | 槽位置变化 | 整体尺度变化 |', '|---|---:|---:|---:|']
    for arm in ['base', 'history'] + [f'u{u}' for u in steps] + ['oracle']:
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{results[st]["rates_percent"][arm]:.2f}%' for st in NAMES) + ' |')
    lines += ['', '每类32个全新独立环境条件；学习方法报告三模型种子的均值，不能将96次相关执行视为96个独立条件。', '',
              '主比较在结果前固定为U20−U3，分别报告两档OOD，其他档位比较为完整描述性消融。']
    for st in NAMES:
        c = results[st]['comparisons']['u20_minus_u3']
        lines.append(f'- {NAMES[st]}：{c["delta_pp"]:+.2f}个百分点，95%区间{c["ci95_pp"]}；两主端点校正区间{c["two_primary_endpoint_ci_pp"]}；救回{c["wins"]}、损害{c["harms"]}（模型×条件执行数）。')
    lines += ['', '![成功率](success_vs_steps.png)', '', '![损失与变化](loss_latent_action_vs_steps.png)', '',
              '全部U0…U100在每个条件中使用相同的历史、同一模型和学习率；z在控制阶段固定。模型训练时使用3步更新，因此更多步数属于推理外推。', '',
              '最终主Agent验收、限制与解释见ROOT_ACCEPTANCE.json。原始结果未按成功筛选。']
    (out / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({s: {'rates': r['rates_percent'], 'u20_minus_u3': r['comparisons']['u20_minus_u3']} for s, r in results.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
