"""Complete, descriptive report of the frozen resolution/view factorial experiment."""
import csv
import html
import numpy as np
import torch
from tools.fast_adapter.p35_common import *

LABELS={'g224':'224 全局','g448':'448 全局','g224w':'224 全局＋腕部','g448w':'448 全局＋腕部'}
ENC={'flat':'拼接','relation_attention':'关系注意力'}
CONTRASTS=(('分辨率／单视角','g224','g448'),('分辨率／双视角','g224w','g448w'),
           ('腕部／224','g224','g224w'),('腕部／448','g448','g448w'))


def main():
    out=ROOT/'summary';out.mkdir(exist_ok=True)
    forecasts=json.loads((ROOT/'validation/PREDICTION_RESULTS.json').read_text())
    branches=json.loads((ROOT/'branches/RESULTS.json').read_text())
    audit=json.loads((ROOT/'AUDIT.json').read_text());assert audit['passed']
    vision_audit=json.loads((ROOT/'VISION_AUDIT.json').read_text());assert vision_audit['passed']
    visibility=json.loads((ROOT/'VISIBILITY_DIAGNOSTIC.json').read_text())
    pred=torch.load(ROOT/'branches/PREDICTIONS.pt',weights_only=True)
    rows=[];lookup={}
    for task in TASKS:
        for view in VIEWS:
            for family in FAMILIES:
                for seed in (11,22):
                    ff=next(r for r in forecasts['rows'] if (r['task'],r['view'],r['family'],r['seed'])==(task,view,family,seed))
                    fixed=next(r for r in branches['rows'] if (r['task'],r['view'],r['family'],r['seed'],r['mode'])==(task,view,family,seed,'fixed'))
                    for mode in ('fixed','first3','continuous'):
                        bb=next(r for r in branches['rows'] if (r['task'],r['view'],r['family'],r['seed'],r['mode'])==(task,view,family,seed,mode))
                        row={k:bb[k] for k in ('task','view','family','seed','mode','progress_mm','regret_mm')}
                        row['prediction_mse']=ff['metrics'][mode]['all']['mean']
                        row['choice_changes_vs_fixed']=int(np.count_nonzero(np.array(bb['choices'])!=fixed['choices']))
                        row['progress_delta_vs_fixed_mm']=bb['progress_mm']-fixed['progress_mm']
                        row['progress_delta_vs_base_mm']=bb['progress_mm']-branches['base'][task]['progress_mm']
                        lookup[(task,view,family,seed,mode)]=row;rows.append(row)
    contrasts=[]
    for task in TASKS:
        for family in FAMILIES:
            for name,left,right in CONTRASTS:
                seeds=[]
                for seed in (11,22):
                    a=lookup[(task,left,family,seed,'continuous')];b=lookup[(task,right,family,seed,'continuous')]
                    seeds.append({'seed':seed,'prediction_mse_delta':b['prediction_mse']-a['prediction_mse'],
                                  'progress_mm_delta':b['progress_mm']-a['progress_mm']})
                contrasts.append({'task':task,'family':family,'name':name,'left':left,'right':right,'by_seed':seeds,
                    'mean_prediction_mse_delta':float(np.mean([r['prediction_mse_delta'] for r in seeds])),
                    'mean_progress_mm_delta':float(np.mean([r['progress_mm_delta'] for r in seeds])),
                    'both_seed_physical_improvement':all(r['progress_mm_delta']>1e-8 for r in seeds)})
    outcomes=torch.load(ROOT/'branches/OUTCOMES.pt',weights_only=True)['outcomes']
    diagnostics=[]
    for task in TASKS:
        for t in (20,40,60):
            oo=[o for o in outcomes if o['task']==task and o['t']==t];v=np.array([o['progress'] for o in oo])
            diagnostics.append({'task':task,'t':t,'snapshots':len(oo),'informative_over_2mm':int((np.ptp(v,axis=1)>.002).sum()),
                'base_progress_mm':float(v[:,0].mean()*1000),'best_progress_mm':float(v.max(-1).mean()*1000),
                'initial_positive_physical_value':sum(o['start_values'][0]>1e-8 for o in oo)})
    costs={}
    for task in TASKS:
        r=json.loads((ROOT/task/'data/RECEIPT.json').read_text())
        costs[task]={'elapsed_seconds':r['seconds'],'frames_per_stream':r['frames'],
                     'render_plus_encode_seconds_by_stream':r['render_and_encode_seconds_by_stream'],
                     'measurement':'wall clock under concurrent extraction; not an isolated latency benchmark'}
    verdict={}
    for task in TASKS:
        c=[r for r in contrasts if r['task']==task and r['both_seed_physical_improvement']]
        a=[(v,f) for v in VIEWS for f in FAMILIES
           if all(lookup[(task,v,f,s,'continuous')]['progress_delta_vs_fixed_mm']>1e-8 for s in (11,22))]
        verdict[task]={'visual_contrasts_positive_in_both_seeds':[(r['name'],r['family']) for r in c],
                       'continuous_over_fixed_positive_in_both_seeds':a}
    summary={'experiment':'P35','status':'complete_bounded_development','training_models':32,
             'fit_objects_per_task':24,'validation_objects_per_task':8,'source_episodes_per_task':64,
             'physical_branches':864,'full_task_rollouts':0,'base':branches['base'],'rows':rows,'contrasts':contrasts,
             'diagnostics_by_time':diagnostics,'extraction_cost':costs,'visibility':visibility['rows'],'descriptive_findings':verdict,
             'no_independent_final_or_task_success_claim':True,
             'bindings':bindings([ROOT/'PROTOCOL.json',ROOT/'validation/PREDICTION_RESULTS.json',ROOT/'branches/RESULTS.json',ROOT/'AUDIT.json',ROOT/'VISION_AUDIT.json',ROOT/'VISIBILITY_DIAGNOSTIC.json'])}
    dump(out/'SUMMARY.json',summary)
    with (out/'all_variants.csv').open('w') as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tables={}
    for task in TASKS:
        table=[]
        for family in FAMILIES:
            for view in VIEWS:
                r=[lookup[(task,view,family,s,'continuous')] for s in (11,22)]
                table.append([ENC[family],LABELS[view],f"{np.mean([x['prediction_mse'] for x in r]):.4f}",
                    f"{np.mean([x['progress_mm'] for x in r]):.3f}",
                    f"{np.mean([x['progress_delta_vs_fixed_mm'] for x in r]):+.3f}",
                    '/'.join(str(x['choice_changes_vs_fixed']) for x in r)])
        tables[task]=table
    header=['编码','视觉输入','预测误差↓','局部进展 mm↑','相对固定z mm','改变动作数 11/22']
    md=[]
    for task,table in tables.items():
        md += [f'### {task}','', '| '+' | '.join(header)+' |','|---|---|---:|---:|---:|---:|']
        md += ['| '+' | '.join(r)+' |' for r in table]
        md += ['',f"基座动作平均进展 {branches['base'][task]['progress_mm']:.3f} mm；48个状态中 {branches['base'][task]['informative_snapshots']} 个的九候选物理效果跨度超过2 mm。",'']
    findings=[]
    for task,v in verdict.items():
        visual='、'.join(f'{x[0]}（{ENC[x[1]]}）' for x in v['visual_contrasts_positive_in_both_seeds']) or '没有'
        adapt='、'.join(f'{LABELS[x[0]]}／{ENC[x[1]]}' for x in v['continuous_over_fixed_positive_in_both_seeds']) or '没有'
        findings.append(f'{task}：两个种子均改善局部进展的视觉对照：{visual}；持续更新相对同checkpoint固定z在两个种子都改善的配置：{adapt}。')
    report='''# P35：分辨率与腕部相机的受控比较

本轮32个模型、两任务、864条四步物理分支和冻结／因果复核完成。结果只说明旧开发材料上的预测与局部动作质量；没有新整任务成功率。下列同向比较均为描述性发现，没有多重比较校正或独立确认。

'''+ '\n\n'.join(findings)+'\n\n'+'\n'.join(md)+'''
表中为持续更新组、两个固定训练种子的均值；最后一列是各种子48个状态中相对同checkpoint固定z改变动作的数量。局部物理进展不是任务成功率。抓取进展是双侧杯柄接触成立时的非负抬升量变化，锤子进展是木块滑动距离；只由评价端在预测封存后读取。

## 做了什么

固定DINOv3与LiLa权重，使用原生224或448渲染，并按相同尺寸编码局部裁剪；不是将旧224图放大。相机固定在手腕上，使用机器人自身标定，不根据物体真值追踪。全局与腕部信息保留在不同输入槽，不平均不同相机的图像坐标。

参照LiLa官方论文：RoboTwin使用320×240外部视角，LIBERO使用256×256外部与腕部视角。本实验是224/448×单/双视角的受控比较，不声称完全复现LiLa设置。[官方论文](https://arxiv.org/html/2608.03701v1#S4.SS2)；[官方配置](https://github.com/teee000/LiLa-WAM/blob/main/configs/robotwin_all.yaml)。

每任务复用24个训练物体、8个验证物体，各有专家和基座轨迹，共64条200步记录。旧开发材料已被打开，不称fresh final或OOD。各视觉组分别训练拼接、关系注意力两个编码器，种子11/22；每个模型600次预测预训练、1000次外层更新。所有组输入槽与参数预算相同，未启用腕部时其槽为零；U256、z32不变。视觉和原动作基座冻结。

所有组使用完全相同的旧224全局反馈：128维物体特征变化、4维图像框变化、2维相对中心变化，权重0.5/0.25/0.25，依可见性归一化。每执行步用已完成四步转移、最近8条反馈更新z；对照为固定z、仅前三次更新、持续更新。没有读取未来反馈或用专家动作作为适配标签。

在每个验证物体的两条轨迹上，固定选第20、40、60步，共48个状态／任务。每个模型只从源轨迹前缀适配，再对同九候选作一次选择；每个候选在同状态保持执行四步。所有模型预测先封存，物理分支只执行一次后共享打分。不会在四步分支内重新选动作；864条分支不能算作864次完整任务。

## 可以和不能得出的结论

高分辨率与近距离视角确实增加目标的原生可见像素：代表性抓取状态中，杯子由全局224的约26×28像素，变为全局448的51×57、腕部448的124×91。这是输入可见性的证据，不等同于学习或控制收益。

固定z与持续更新的差值才是本checkpoint的在线适配贡献。高分辨率／腕部对照的差值属于视觉输入变化；不能把固定z已有的收益归给梯度。两个模型种子、每任务8个验证物体仍不足以支持普遍优势。

这次保持了4×4空间压缩、旧全局反馈和原物体运动打分。因此，负结果只约束这条固定控制链，不能证明高分辨率或腕部相机没有用；正结果也需完整闭环验证。尤其是尚未建立接触时，短时物体位移可能都为零，无法充分评价为后续抓取做准备的动作。色标ROI、预设角色和语义目标均不是自动部位识别或通用视觉理解。

## 材料

- [全部种子／模式／视觉组数值](all_variants.csv)
- [配对差值、按时刻诊断及耗时](SUMMARY.json)
- [因果与物理复核](../AUDIT.json)
- [原生像素对比](../visual_review/PIXEL_EXAMPLES.json)
- [预定范围](../PROTOCOL.json)
- [结果与注释短视频](index.html)

本批结束，不改写旧P34，不按本批结果调整超参数或继续追加回合。
'''
    (out/'REPORT_ZH.md').write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,task in enumerate(TASKS):
        for family,color in (('flat','#4684b5'),('relation_attention','#147d71')):
            for mode,style in (('fixed','--'),('continuous','-')):
                r=[[lookup[(task,v,family,s,mode)] for s in (11,22)] for v in VIEWS]
                for row,key in enumerate(('prediction_mse','progress_mm')):
                    axes[row,col].plot(range(4),[np.mean([x[key] for x in rs]) for rs in r],style,marker='o',color=color,label=f'{family} / {mode}')
        for ax in axes[:,col]:ax.set_xticks(range(4),['224','448','224 + wrist','448 + wrist']);ax.grid(alpha=.2)
        axes[0,col].set_title(task);axes[0,col].set_ylabel('Prediction MSE (lower better)');axes[1,col].set_ylabel('Four-step physical progress (mm)')
        axes[1,col].axhline(branches['base'][task]['progress_mm'],color='gray',linestyle=':',label='Frozen base action')
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8);fig.suptitle('P35: visual input and continuous adaptation; development, not task success')
    fig.tight_layout();fig.savefig(out/'comparison.svg');fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    video_file=out/'media/VIDEO_INDEX.json';media=''
    if video_file.exists():
        for v in json.loads(video_file.read_text())['videos']:
            media+=f'<article><h3>{html.escape(v["task"])} / {html.escape(v["kind"])}</h3><video controls preload="metadata" poster="media/{Path(v["cover"]).name}" src="media/{Path(v["file"]).name}"></video></article>'
    ht=''
    for task,table in tables.items():
        ht+=f'<h2>{task}</h2><table><tr>'+''.join(f'<th>{h}</th>' for h in header)+'</tr>'
        ht+=''.join('<tr>'+''.join(f'<td>{html.escape(c)}</td>' for c in r)+'</tr>' for r in table)+'</table>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>P35分辨率与腕部相机比较</title><style>
body{font:17px/1.65 system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 28px;color:#142934;background:#f6f9fb}
table{width:100%;border-collapse:collapse;background:white}td,th{padding:12px;border-bottom:1px solid #d8e1e8;text-align:left}
.notice{background:#fff0dc;border-left:5px solid #ce8b21;padding:18px}img,video{width:100%}article{padding:15px;background:white;margin:18px 0}a{color:#116e80}
</style><h1>P35：原生分辨率与腕部相机</h1><div class="notice">已完成32模型、864条四步分支；<b>此页不是完整任务成功率。</b>共同224反馈、4×4空间压缩及动作评分保持不变。</div>'''
    page+=''.join(f'<p>{html.escape(f)}</p>' for f in findings)+ht
    page+='<p>两种子平均。改变动作数分别列种子11/22；全部模式见CSV。</p><img src="comparison.svg"><h2>同状态原生像素比较</h2>'
    page+=''.join(f'<img src="../visual_review/{t}_comparison.jpg" alt="{t}原生相机比较">' for t in TASKS)
    page+='<p><a href="REPORT_ZH.md">完整说明</a> · <a href="all_variants.csv">逐种子数据</a> · <a href="SUMMARY.json">比较与诊断</a> · <a href="../AUDIT.json">复核</a> · <a href="../../adapter_dashboard/">Adapter Dashboard</a></p><h2>局部动作短片</h2>'+media+'</html>'
    (out/'index.html').write_text(page)
    dump(out/'ROOT_ACCEPTANCE.json',{'passed':True,'experiment':'P35','status':'complete_bounded_development',
        'no_promotion':True,'full_task_rollouts':0,'all_32_models_reported':True,'all_96_mode_cells_reported':len(rows)==96,
        'two_seeds_descriptive_not_confirmatory':True,'bindings':bindings([out/'SUMMARY.json',out/'REPORT_ZH.md',out/'index.html',ROOT/'AUDIT.json',Path(__file__)])})
    print('P35_REPORT_COMPLETE',verdict,flush=True)


if __name__=='__main__':main()
