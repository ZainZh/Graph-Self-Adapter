"""Local readable report assembled only from frozen statistics and audits."""
import html
import json
from tools.fast_adapter.p41_run import js,sha
from tools.fast_adapter.p42_batch import ROOT

ARMS=['single','fixed','prefix','gradient','multiview']
LABEL={'single':'单帧视觉','fixed':'同模型固定几何','prefix':'前12步后固定','gradient':'持续几何梯度','multiview':'同信息多视图'}
TASK={'cup':'杯柄','insert':'插入'}


def interval(values,scale=100):return '['+', '.join(f'{v*scale:.2f}' for v in values)+']'
def pformat(value):return f'{value:.3g}'


def video_note(video, rows):
    row=next(r for r in rows if (r['task'],r['split'],r['seed'],r['arm'])==(video['task'],'geometry',video['seed'],'gradient'))
    if video['kind']=='success':return '持续梯度达到并保持全部物理成功判据。'
    if video['task']=='cup':
        return f'此例末状态抬升{row["lift"]*1000:.1f}mm，但双侧接触为否；末10状态仅{row["last10_successes"]}/10满足原判据，因此保留为失败。'
    if row['valid_frames']==0:return '全回合没有准入的完整槽口四角轮廓，控制保持等待；全景中看见槽口不等于工具相机具备可用闭合轮廓。'
    return f'具备{row["valid_frames"]}帧可用轮廓，但末10状态仅{row["last10_successes"]}/10满足原判据。'


def main():
    out=ROOT/'summary'
    result=json.loads((out/'RESULTS.json').read_text())
    audit=json.loads((out/'AUDIT.json').read_text())
    videos=json.loads((out/'VIDEOS.json').read_text())
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    local=json.loads((out/'LOCAL_VERIFICATION.json').read_text())
    transfer=json.loads((out/'TRANSFER_VERIFICATION.json').read_text())
    resources=json.loads((out/'RESOURCE_EXIT.json').read_text())
    resources_exited=all(not p['exists'] for p in resources['processes'].values())
    primary=[p for p in result['paired'] if p['primary']]
    complete=sum(r['execution_status']=='complete' for r in result['rows'])
    verified=complete==1280 and audit['fully_replayed']==1280 and audit['all_source_and_trace_hashes'] and audit['common_prefix_equal'] and audit['outcomes_recomputed'] and local['status']=='passed' and not transfer['mismatches']
    accepted=verified and result['all_four_primary_positive'] and not result['exceptions']
    conclusion=('四项预注册主要比较全部支持：在这两个冻结任务的范围外几何分布中，持续几何更新优于同模型固定几何和逐帧单帧几何。'
                if accepted else '主要比较存在未通过或证据不完整的项目，以下逐项保留；不改门槛、不选择最佳子集。')
    lines=['# P42-A：冻结三维几何原型的扩大独立复验','',
        '本批是 P41 解析几何原型的独立条件复验，不训练新模型。P41 的环境、适配器、控制器和 rollout 保持原始字节；只增加结果前冻结的新条件与统计口径。','',
        '## 结论','',conclusion,'',
        '**这不是 DINO／LiLa 神经 Adapter 或元学习成功的证据。** 当前能力使用可见功能部位 mask、已知相机标定、工具本体位姿和解析几何先验；执行中的相机运动提供多视角几何信息。多视图拟合是次要强参考，本轮不要求梯度超过它，也不据此宣称梯度独有优势。','',
        f'预定并完成 {complete}/1280 回合：2任务 × 2设置 × 64新几何条件 × 5臂；每回合190动作。新种子为4211001–4211064，条件间重置latent，各臂配对使用同一几何与12步共同观察前缀。执行异常 {len(result["exceptions"])} 条。','',
        '## 全部成功率与 Wilson 95% 区间','',
        '| 任务／设置 | 方法 | 成功数 | 成功率 | Wilson 95% |','|---|---|---:|---:|---:|']
    for cell in result['cells']:
        setting='范围内' if cell['split']=='id' else '范围外几何'
        lines.append(f'| {TASK[cell["task"]]}／{setting} | {LABEL[cell["arm"]]} | {cell["successes"]}/{cell["n"]} | {100*cell["rate"]:.2f}% | {interval(cell["wilson_95"])}% |')
    lines+=['','## 四项主要配对比较','',
        '每行64个范围外几何条件；胜／负／平指持续梯度相对参照。差值和区间单位为百分点。95%区间按几何条件进行10万次配对bootstrap；98.75%区间按四比较Bonferroni给出。显著性使用精确单侧McNemar检验，并对四项做Holm校正。','',
        '| 任务 | 参照 | 差值 | 配对95%区间 | 家族98.75%区间 | 胜／负／平 | Holm p | 支持 |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for p in primary:
        lines.append(f'| {TASK[p["task"]]} | {LABEL[p["reference"]]} | {100*p["delta"]:+.2f} | {interval(p["paired_bootstrap_95"])} | {interval(p["paired_bootstrap_bonferroni_98_75"])} | {p["wins"]}/{p["losses"]}/{p["ties"]} | {pformat(p["holm_adjusted_p"])} | {"是" if p["positive_primary_comparison"] else "否"} |')
    lines+=['','通过规则在结果前固定为差值>0且Holm校正p<0.05；没有使用多视图优势、最佳种子、最佳设置或新阈值替代主要比较。有限离散样本的百分位bootstrap可能在所有配对差值一致时退化，不能把零宽区间理解为总体零不确定性。','',
        '## 次要配对参考','',
        '| 任务／设置 | 参照 | 梯度差值 | 配对95%区间 | 胜／负／平 |','|---|---|---:|---:|---:|']
    for p in result['paired']:
        if p['primary']:continue
        setting='范围内' if p['split']=='id' else '范围外几何'
        lines.append(f'| {TASK[p["task"]]}／{setting} | {LABEL[p["reference"]]} | {100*p["delta"]:+.2f} | {interval(p["paired_bootstrap_95"])} | {p["wins"]}/{p["losses"]}/{p["ties"]} |')
    lines+=['','这些比较是描述性参考，未被纳入四项主要检验的推断结论。','',
        '## 保留的失败与证据边界','']
    for task in ['cup','insert']:
        failures=[r for r in result['rows'] if r['task']==task and r['split']=='geometry' and r['arm']=='gradient' and not r['success']]
        invisible=sum(r.get('valid_frames')==0 for r in failures)
        lines.append(f'- {TASK[task]}范围外持续梯度保留 {len(failures)}/64 个失败，其中 {invisible} 个回合全程没有准入的四角轮廓；其余失败不能仅由这一计数归因。')
    lines+=['- 几何变化遵循 P41 原分布：杯柄尺寸、高度、位置与绕竖轴朝向；物块／槽口尺寸、三维位置与倾角。质量、摩擦、执行倍率、控制增益和固定开闭方式不变，不学习力度。',
        '- “同模型固定几何”是同一解析模型、名义几何先验和控制器固定latent；本原型没有神经checkpoint。',
        '- 普通运行不读取目标真实位姿、尺寸、接触力、成功标签；初始已持物块尺寸来自一次外部视角mask拟合。评价真值只在动作和观察封存后读取。',
        '- “持续”沿用P41实现：杯柄在前105个动作的观察阶段更新，形成抓持锚点后使用固定锚点与本体抬升；插入在全回合使用已执行动作后的可用观察。每次更新三步，12维角点latent采用解析阻尼预条件，不是元训练参数。',
        '- 浮动工具／夹爪可控六自由度，杯子为自由刚体、插槽有碰撞壁；仍是近距离局部技能，非完整机械臂、任意网格物体、真实场景或完整RSI。',
        '- 原P41 runner中scope=development的文字按冻结字节保留；独立复验身份由本批结果前PROTOCOL／FREEZE与逐回合P42_EPISODE_CONTRACT定义。',
        '- 本批结果不回流调参；不晋升、不追加种子、不覆盖P41和其他历史封存。','',
        '## 完整复核与视频','',
        f'全部 {audit["fully_replayed"]} 回合、{audit["actions"]} 动作已从封存观察重算，并从相同初态完整物理回放；动作、几何估计、位置、速度和本体观察的最大误差均为 {max(audit["max_errors"].values()):g}。源文件和轨迹哈希、共同前缀及全部末10状态成功判据复核一致。','',
        f'服务器的 {transfer["files"]} 个结果文件、{transfer["bytes"]} 字节已全部回传本机并逐文件核验SHA256。本机另从原始RESULT独立复算20格成功数／Wilson区间、16项胜负平／精确p值及四项Holm校正；256个实际几何配置与P41所有已保存配置无重合。','',
        '成功判据保持 P41 原定义：杯柄须真实双侧接触并抬升超过40mm；插入须深度超过34mm、横向各轴误差低于3mm、姿态误差低于0.1rad；最后10个状态全部满足且最大接触穿透小于2mm。','',
        '视频是封存后的派生展示，每任务按预注册规则选取首个梯度成功与首个梯度失败条件，并对比同条件单帧／固定／持续梯度。中文显示阶段、全景与工具视角、估计位置和朝向；不作为指标来源。','']
    missing=sum(v.get('rendered_corner_audit',{}).get('missingness_mismatches',0) for v in videos)
    coords=sum(v.get('rendered_corner_audit',{}).get('coordinate_mismatches',0) for v in videos)
    max_pixel=max([v.get('rendered_corner_audit',{}).get('max_coordinate_error_px',0) for v in videos]+[0])
    lines.append(f'派生视频按封存qpos/qvel重新执行mj_forward并渲染；相对原在线观察，可用轮廓准入状态改变 {missing} 帧，双方都有轮廓时坐标变化 {coords} 帧，按原角点顺序的最大坐标差 {max_pixel:g} 像素，详见VIDEOS.json。这是轮廓级核对，不是完整mask逐像素核对。重建视频不替代原观察，不改变任何动作、成功率或复核输入；不宣称逐像素重现。')
    lines.append('')
    for v in videos:
        kind='成功' if v['kind']=='success' else '失败'
        lines.append(f'- [{TASK[v["task"]]}／{kind}／{v["seed"]}]({v["file"]})：'+video_note(v,result['rows']))
    lines+=['',('回传验收时，本批两个执行worker与后处理进程均已退出，GPU0已释放。' if resources_exited else '进程退出状态仍需核对。')+'详见[资源退出记录](RESOURCE_EXIT.json)。']
    lines+=['','[全部机器结果](RESULTS.json) · [复核](AUDIT.json) · [视频来源](VIDEOS.json) · [协议](../PROTOCOL.json) · [冻结](../FREEZE.json) · [完整传输核验](TRANSFER_VERIFICATION.json)','']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines))
    js(out/'ACCEPTANCE.json',dict(status='completed_independent_replication_positive' if accepted else 'completed_independent_replication_mixed_or_qualified',
        complete_episodes=complete,expected_episodes=1280,exceptions=len(result['exceptions']),
        full_audit_verified=verified,all_four_primary_positive=result['all_four_primary_positive'],
        resources_exited=resources_exited,
        accepted_claim='bounded OOD analytic geometry adaptation relative to same-model fixed and single-frame controls' if accepted else 'see each predeclared comparison',
        not_claimed=['DINO/LiLa neural adapter validation','meta-learning','gradient superiority over multiview','full robot arm','complete RSI','promotion'],
        sources={name:sha(out/name) for name in ['RESULTS.json','AUDIT.json','VIDEOS.json','REPORT_ZH.md','LOCAL_VERIFICATION.json','TRANSFER_VERIFICATION.json','FRESHNESS.json','RESOURCE_EXIT.json']}))
    table=['<table><tr><th>任务／设置</th>'+''.join('<th>'+LABEL[a]+'</th>' for a in ARMS)+'</tr>']
    for task in ['cup','insert']:
        for split in ['id','geometry']:
            row='<tr><td>'+TASK[task]+'／'+('范围内' if split=='id' else '范围外几何')+'</td>'
            for arm in ARMS:
                c=next(c for c in result['cells'] if (c['task'],c['split'],c['arm'])==(task,split,arm))
                row+=f'<td>{c["successes"]}/64<br>{100*c["rate"]:.2f}%<br><small>95% {interval(c["wilson_95"])}</small></td>'
            table.append(row+'</tr>')
    table.append('</table>')
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>P42-A 独立几何复验</title>',
        '<style>body{font:18px system-ui;max-width:1400px;margin:36px auto;padding:0 20px;color:#173247;background:#edf2f5}p{line-height:1.7}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #ccd8dd;padding:12px}small{font-size:14px}video{width:100%}a{color:#08786e}</style>',
        '<h1>P42-A：冻结三维几何原型的扩大独立复验</h1><p>'+conclusion+'</p>',
        '<p>1280回合，256个新任务／设置／几何条件。使用允许的部件mask、标定与解析几何先验；不是DINO、元学习或完整机械臂证据。质量、摩擦与执行增益固定。</p>',
        *table,
        '<p>表中区间为Wilson 95%。四项主要配对检验与多重比较校正在完整报告中；前缀和多视图仅作次要参考。</p>']
    for v in videos:
        page += ['<h2>'+TASK[v['task']]+' · '+('首个梯度成功' if v['kind']=='success' else '首个梯度失败')+f' · {v["seed"]}</h2>',
            '<p>按结果前规则选取代表条件；同条件三种方法，封存状态回放。'+video_note(v,result['rows'])+'</p>',
            f'<video controls preload="metadata" poster="{html.escape(v["cover"])}" src="{html.escape(v["file"])}"></video>',
            f'<p><a href="{html.escape(v["file"])}">打开视频文件</a></p>']
    page+=['<p><a href="REPORT_ZH.md">完整报告</a> · <a href="RESULTS.json">全部结果</a> · <a href="AUDIT.json">全量复核</a> · <a href="VIDEOS.json">视频来源</a> · <a href="../PROTOCOL.json">结果前协议</a></p></html>']
    (out/'index.html').write_text('\n'.join(page))


if __name__=='__main__':main()
