"""Root-reviewed P24 closeout, preserving the unresolved precision audit exception."""
from pathlib import Path
import html
import json
import time
from tools.fast_adapter.p18_run import digest, save_json
from tools.fast_adapter.p24_common import R


def main():
    out = R/'summary'
    assert not (out/'ROOT_ACCEPTANCE.json').exists()
    result = json.loads((out/'RESULTS.json').read_text())
    audit = json.loads((out/'AUDIT_SUPPLEMENT.json').read_text())
    replay = json.loads((out/'PRECISION_REPLAY.json').read_text())
    assert audit['all_outcomes_recomputed'] and audit['query_episodes_recomputed'] == 6720
    assert len(audit['precision_exceptions']) == 1
    assert audit['precision_exceptions'][0]['indices'] == [[63, 7, 2]]
    assert not audit['passed'] and not replay['passed']
    analysis = json.loads((R/'ANALYSIS_FREEZE.json').read_text())
    for b in analysis['bindings'] + result['source_bindings']:
        assert digest(Path(b['path'])) == b['sha256'], b['path']
    videos = json.loads((out/'media/VIDEO_INDEX.json').read_text())['videos']
    for row in videos:
        assert digest(Path(row['file'])) == row['sha256']
        for b in row['sources']:
            assert digest(Path(b['path'])) == b['sha256']

    def bind(path):
        return {'path': str(path), 'sha256': digest(path)}

    aggregate = {
        'passed': False,
        'status': 'outcomes_and_support_verified_one_query_action_precision_exception_unresolved',
        'all_success_outcomes_recomputed': 6720,
        'source_export_bindings_verified': audit['source_export_bindings'],
        'analysis_bindings_verified': len(analysis['bindings']),
        'support_actions_recomputed': audit['support_actions_recomputed'],
        'maximum_latent_replay_error': audit['maximum_latent_replay_error'],
        'maximum_support_action_replay_error': audit['maximum_support_action_replay_error'],
        'precision_exceptions': audit['precision_exceptions'],
        'initial_failure': bind(out/'AUDIT_INITIAL_FAILURE.json'),
        'full_supplement': bind(out/'AUDIT_SUPPLEMENT.json'),
        'float32_reconstruction_attempt': bind(out/'PRECISION_REPLAY.json'),
        'original_threshold_relaxed': False,
        'limitation': 'Query visuals were cached fp16. One action component differs .00525045 versus the original .005 threshold. Deterministic action replay exactly reproduces positions/velocities and initial RGB, but re-encoded features are not bitwise identical and do not resolve the exception. Do not call the original full audit passed.',
        'scientific_scope': 'Success rates describe the original sealed trajectories and all success labels were independently reconstructed; fine-grained action reproducibility has the stated limitation.'}
    save_json(out/'AUDIT.json', aggregate)

    acceptance = {
        'status': 'completed_mixed_outcomes_verified_with_unresolved_precision_audit_exception',
        'accepted_at_unix': time.time(),
        'root_review': 'GPT-6 Astra; no delegated agents',
        'completion_accepted': True,
        'full_numerical_audit_passed': False,
        'candidate_promoted': False,
        'scope': 'User-proposed shared prior P/A/D meta-training; fixed DINOv3/LiLa/PCA and frozen base. Three seeds, 50000 full meta and 50000 History updates per seed. No history warmup.',
        'query_episodes': 6720,
        'unique_test_conditions': 192,
        'rates_and_comparisons': bind(out/'RESULTS.json'),
        'audit': bind(out/'AUDIT.json'),
        'supported_on_recorded_trajectories': [
            'Latent updates change actions and improve ID and scale sample means over the same model at K0.',
            'The prespecified scale K3-v-K0 adjusted bootstrap interval excludes zero; position does not.',
            'All K3 sample means remain below the unchanged base, with a negative scale comparison interval.',
            'Correct versus swapped K3 latents and Meta versus History show no stable advantage.'
        ],
        'not_established': [
            'Useful self-improvement beyond the frozen visual skill.',
            'Environment-specific latent adaptation rather than a broadly shared update offset.',
            'Superiority to History, robustness to goal-position OOD, or a complete RSI mechanism.',
            'Bitwise visual reproduction or a fully passing original numerical audit.'
        ],
        'training_diagnostic': bind(R/'diagnostics/TRAINING_RESIDUAL_CAP_ALL.json'),
        'diagnostic_limit': 'Some on-policy expert labels exceed the fixed residual action cap. This diagnoses supervised fit feasibility, not proof of task impossibility or sole cause.',
        'next_recommendation_not_started': 'Before scaling, preserve base behavior, test whether updates use environment-specific evidence, and align the adaptation objective with actionable correction; keep the visual frontend fixed for those tests.',
        'dashboard': 'Deferred per user instruction',
        'server_workers': 'Campaign ended; deterministic audit replay also ended. No persistent new service.'}
    save_json(out/'ROOT_ACCEPTANCE.json', acceptance)

    labels = [('base','原冻结视觉基座'),('meta_k0','新Meta，同模型不更新z'),
              ('meta_k1','新Meta，1反馈／1更新'),('meta_k2','新Meta，2反馈／2更新'),
              ('meta_k3','新Meta，3反馈／3更新'),('meta_k5','新Meta，5反馈／5更新'),
              ('history_k3','History，3反馈'),('swapped_k3','Meta，交换环境的z3'),('oracle','真值参考')]
    strata = [('id','同分布'),('position','槽位置变化'),('scale','整体尺度变化')]
    table = ['| 方法 | 同分布 | 槽位置变化 | 整体尺度变化 |', '|---|---:|---:|---:|']
    html_rows = []
    for arm, label in labels:
        rates = [f"{result['strata'][st]['rates_percent'][arm]:.2f}%" for st, _ in strata]
        table.append('| ' + ' | '.join([label, *rates]) + ' |')
        html_rows.append('<tr><th>'+label+'</th>'+''.join('<td>'+v+'</td>' for v in rates)+'</tr>')
    effects = []
    for st, label in strata:
        c = result['strata'][st]['comparisons']['meta_k3_minus_meta_k0']
        interval = c['ci95_pp'] if st == 'id' else c['two_primary_endpoint_ci_pp']
        level = '次要95%' if st == 'id' else '预定主比较97.5%'
        effects.append(f"- {label}：K3−K0 {c['delta_pp']:+.2f}个百分点，{level}区间 [{interval[0]:.2f}, {interval[1]:.2f}]；{c['wins']}次救回、{c['harms']}次损害（模型×条件执行）。")

    headline = '重新训练完成；有同模型更新收益，但仍低于基础控制，未证明环境专属适应或优于History。'
    audit_text = ('6,720条原始测试轨迹的成功／失败全部复算一致，301个训练／来源／回传绑定及3个分析脚本绑定通过；2,880个支持动作及全部latent重算通过。'
        '原审计有一项未过：scale/seed33/meta_k1，条件63、第7步、第三动作分量，从半精度视觉缓存复算的差值0.00525045，超过预设0.005。'
        '补充重放使用原服务器、原GPU、原64批量以及封存动作，物理位置／速度与初始RGB精确一致，但重新编码的视觉特征仍未位级一致；动作最大误差0.00660932，未能消除例外。'
        '原阈值没有放宽，原失败与补充失败均保留，完整数值审计标为未通过。成功率属于已封存实际执行结果；不能声称所有动作与视觉均已精确重现。')
    md = f'''# P24 主Agent结果报告

**{headline}**

按用户方案重新训练了三个种子。每个种子Meta和History各50,000次外层更新，无History预热。DINOv3、LiLa、原训练期PCA和视觉基础控制权重不变；新训练交互编码P、修正动作A、下一表示预测D、共同z0及正学习率alpha。Meta不再用GRU初始化z。

每个训练样本先抽取K∈{{1,2,3,5}}：每一条已完成反馈只更新一次z，外层梯度穿过这些更新，再用同环境另一初态的query动作标签训练全部元参数。外层采用技术文档允许的专家动作监督及预测辅助损失，没有反传模拟器最终成功。大部分外层迭代读取真实轨迹回放；不是每一次外层迭代都在线启动新物理回合。

初始材料覆盖384个训练条件；每个种子在10,000和30,000步分别补充384条当前Meta策略轨迹，共2,304条新训练轨迹，几何／动力学条件池仍是原384个。Meta与History读取相同材料、执行相同外层次数；History参数更多，未声称参数量匹配。主训练加全部独立测试耗时约34.5分钟，不含结果回传、审计与视频整理。

{chr(10).join(table)}

每档64个全新独立环境，共192条件。三模型种子共享条件，总计6,720次query执行；192次模型×条件执行／学习方法／档位，不等于192个独立环境。基础控制／真值参考各64次。每次query重置到独立起点，z封存后固定48步，成功需最后连续三帧满足插入判据。K同时表示支持反馈数和更新次数，不是固定数据上的纯优化次数消融。

{chr(10).join(effects)}

主端点为K3相对同checkpoint K0的两档OOD比较，使用按条件和模型种子重采样的预定bootstrap；尺度端点正向，位置端点跨零。只有三个训练种子，区间仍是有限重复的近似估计。其余K完整披露，不按测试结果重新选最优K。

三条结论必须同时保留：

1. z的梯度更新确实改变行为，并在同分布与尺度档改善相对自身K0的成功率；不是根本没有执行更新。
2. K3仍低于未加新Adapter的冻结基础控制；尺度差距−18.75个百分点，描述性95%区间[−30.21,−7.81]。这版训练出的外围模块没有实现总体提升。
3. K3相对同反馈History没有稳定优势；交换其他环境z3后，同分布53.13%、尺度28.65%，接近正确z的55.21%、28.13%。所有正确−交换的95%区间均包含或触及零，尚不能把K0→K3收益解释为已学会针对当前环境进行适配。

一个需要下一轮验证的解释是：训练只对K≥1的query优化，未约束K0保留基座能力；首次更新可能产生多数环境共用的状态偏移，后续反馈贡献很小。K1之后曲线基本持平，与这个解释相容，但不能据此断言原因已证明。

训练材料上的事后诊断还发现：新增轨迹中16.2%–30.9%的动作标签超出固定±0.35残差修正范围，原始示范约1.7%。这意味着部分监督目标在该参数化下无法精确拟合，也解释了为什么不能简单以训练误差降不下去判定梯度实现错误。它不证明任务不可解，亦非唯一原因的因果证据。诊断没有进入任何测试时输入，也没有据此修改模型。

任务仍是自建MuJoCo的双物体插入，物块直接受控移动，没有机械臂，不是官方LIBERO任务。输入只含全局RGB经固定LiLa/PCA所得表示、受控物块XYZ、已执行动作；槽的位置／尺度、真值框和隐藏动力学不提供给普通模型。训练专家可读取真值产生动作标签。LiLa仍含公开预训练的外部操作监督，输出是全景token，未宣称纯局部物体表示。

## 复算与限制

{audit_text}

原始审计：[失败记录](AUDIT_INITIAL_FAILURE.json)；[完整补充复算](AUDIT_SUPPLEMENT.json)；[精度重放尝试](PRECISION_REPLAY.json)；[统一审计状态](AUDIT.json)。该例外不改写任何成功标签，亦不作为挑选或重跑测试的理由。

## 曲线与视频

![全部K成功率](success_vs_interactions.png)

![训练轨迹](training_curves.png)

第10,000／30,000步引入新轨迹，训练曲线前后数据分布不同。误差跃升不能直接解释为模型突然变差。不同支持步骤的预测误差来自不同转移，也不能当成同一样本的连续优化曲线。

[打开全部六个中文注释视频](index.html#videos)。每档固定选一个K3救回和一个K3损害，四列显示基础控制／History／Meta K0／Meta K3。视频只是封存位置与速度的展示回放，显示query阶段，未冒称重放整个支持探索。几何误差为事后注释，不是模型输入；“当前已插入”仍可能因未满足最后连续三帧而最终失败。

## 后续建议，尚未执行

保持视觉前端不变，优先修复“加入Adapter反而损害基座”并检验更新是否使用环境专属反馈；同时检查固定动作修正范围与训练标签、适应目标是否匹配。下一轮应在训练开发材料上先确认这些环节，再冻结新比较和新独立测试，不继续在本批final上选结构或学习率。

本轮已结束，不晋升模型，不宣称Fast Adapter机制已经达到目标。spec/current-work更新完成状态；按用户要求暂缓Dashboard，未启动下一轮。
'''
    (out/'ROOT_REPORT_ZH.md').write_text(md)

    cards = []
    names = dict(strata)
    for video in videos:
        p = Path(video['file']).relative_to(out)
        description = '更新救回的例子' if video['kind'] == 'update_win' else '更新造成损害的例子'
        cards.append(f'<article><h3>{names[video["stratum"]]} · {description}</h3><p>种子{video["seed"]}，条件{video["domain"]}；由完整数据按固定规则选取。</p><video controls preload="none" poster="{p.with_suffix(".png")}" src="{p}"></video></article>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>P24共同初值重训结果</title>
<style>body{{font:17px/1.75 system-ui,sans-serif;max-width:1380px;margin:40px auto;padding:0 30px;color:#172a33;background:#f6f8f8}}h1{{font-size:30px}}h2{{margin-top:44px}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:10px 18px;border-bottom:1px solid #dce3e4;text-align:right}}th:first-child{{text-align:left}}img,video{{width:100%;display:block}}article{{background:white;padding:20px;margin:24px 0;border-radius:8px}}a{{color:#126d70}}.result{{font-size:21px;font-weight:650}}.note{{border-left:4px solid #bf782b;padding:10px 20px;background:#fff4e6}}</style>
<h1>P24 · 按共同z0方案重新训练</h1><p class="result">{headline}</p>
<p>LiLa / DINOv3 / PCA / 基础控制不变；3种子，每方法每种子5万次外层更新；训练K随机1/2/3/5。192个新独立条件，6,720次完整query。</p>
<p><a href="ROOT_REPORT_ZH.md">完整解释与实验限制</a> · <a href="RESULTS.json">机器结果</a> · <a href="ROOT_ACCEPTANCE.json">主Agent验收</a> · <a href="#videos">六个注释视频</a></p>
<table><thead><tr><th>方法</th><th>同分布</th><th>槽位置变化</th><th>整体尺度变化</th></tr></thead><tbody>{''.join(html_rows)}</tbody></table>
<p>K表示反馈数量，也是z更新次数；K0为同checkpoint不更新。测试query内z固定。各档64环境由三个模型种子共同使用。</p>
<p>预定尺度K3−K0区间正向，位置区间跨零；相对History、交换z均未建立稳定优势。K0未进入训练的query目标，不能仅凭它较差就声称自提升成功。</p>
<h2>完整K曲线与训练曲线</h2><img src="success_vs_interactions.png" alt="全部反馈数成功率"><img src="training_curves.png" alt="外层训练轨迹">
<p>1万与3万步补充当前策略的真实训练轨迹，前后数据分布不同。初始384个训练条件未扩增，新增2,304条轨迹使用新的起点。</p>
<h2>复算状态</h2><p class="note">{html.escape(audit_text)}</p><p><a href="AUDIT.json">查看审计限制和完整证据</a>。未修改阈值、模型、成功标签或任何原结果。</p>
<h2 id="videos">注释对比视频</h2><p>左起：冻结视觉基座、History K3、Meta K0、Meta K3。这里显示独立query阶段；几何误差为事后注释。当前帧插入不等于最终成功，最终需要末尾连续三帧满足判据。</p>{''.join(cards)}
<p>主Agent独立完成；本批关闭，模型不晋升，Dashboard按要求暂缓。</p></html>'''
    (out/'index.html').write_text(page)
    # All binary evidence remains local and on the authorized server; the index binds it.
    entries = [dict(bind(p), bytes=p.stat().st_size) for p in sorted(R.rglob('*'))
               if p.is_file() and p.name != 'EVIDENCE_INDEX.json']
    save_json(R/'EVIDENCE_INDEX.json', {'files': entries, 'count': len(entries),
        'root_status': acceptance['status'], 'large_binary_artifacts_in_git': False})
    print(json.dumps({'status': acceptance['status'], 'indexed_files': len(entries), 'videos': len(videos)}))


if __name__ == '__main__':
    main()
