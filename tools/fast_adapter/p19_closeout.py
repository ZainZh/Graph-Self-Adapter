"""Root closeout: interpret all frozen results, never retrain or select K/U."""
import json,datetime
from pathlib import Path
import numpy as np
import torch
from tools.fast_adapter.p19_run import ROOT,KINDS,SEEDS,digest,save_json
from tools.fast_adapter.p19_report import bootstrap

def main():
 torch.set_num_threads(2);out=ROOT/'summary';s=json.loads((out/'SUMMARY.json').read_text())
 additional={}
 for st in ['id','mild','strong']:
  additional[st]={}
  for kind in KINDS:
   raw=[torch.load(ROOT/'final'/st/f'{kind}_seed{seed}_RAW.pt',weights_only=False) for seed in SEEDS]
   d=np.stack([(r['gradient_s3_u20']['success'].float()-r['history_s3']['success'].float()).mean(1).numpy() for r in raw])
   additional[st][kind]={'U20_vs_history_success_delta':float(d.mean()),'CI95':bootstrap(d),'secondary_predeclared_U_not_reselected_primary':True}
 save_json(out/'SECONDARY_COMPARISONS.json',additional)
 # Recheck prediction seals, independent of the main aggregate code.
 seal_count=0
 for p in (ROOT/'final').glob('*/*_PREDICTIONS.pt'):
  assert digest(p)==json.loads(p.with_suffix('.json').read_text())['sha256'];seal_count+=1
 assert seal_count==30
 accept={
  'status':'completed_bounded_positive_structured_gradient_adaptation',
  'primary_setting':{'support_interactions':3,'gradient_steps':3,'primary_not_changed_after_results':True},
  'independent_final':{'domains_per_stratum':256,'queries_per_domain':4,'model_seeds':SEEDS,'strata':['id','mild','strong'],'inputs':KINDS},
  'supported':['Both numeric and supervised-DINO-geometry inputs improve versus same-model frozen z in all three strata; paired pointwise95% intervals exclude zero.','Correct z outperforms cross-domain swapped z in all three strata.','Updating the explicit response parameters changes actions materially.'],
  'not_supported':['The original unstructured P18 model has been repaired under otherwise identical conditions.','Gradient U3 consistently outperforms History: it is worse in all visual strata.','Meta-learning adds visual success gains over fixed-prior ordinary gradient U3: all pointwise intervals include zero.','GT-free vision, robotics/VLA performance, persistent RSI, universal advantage over classical system identification.'],
  'secondary':'At fixed S3, predeclared U20 improves strong-OOD visual success to80.46875%, but ID falls versus U3; do not choose a different best U separately for each test stratum.',
  'limits':['Extra supervised geometric readout; GT ROIs and query geometric feedback.','Known affine response form and analytic controller; a representation/prediction/control package repair, not a single-factor root-cause experiment.','History and meta parameter counts unequal; no equal-compute superiority claim.','History S0 was not part of the training S set; it is a diagnostic, not the primary baseline.','Gradient-zero-support numeric arm triggers the declared singular-model fallback; it is not a pure missing-visual interpretation.','All final data are now opened and may not be reused as fresh held-out.'],
  'verification':{'five_core_tests':True,'all20_models_trained':True,'three_remote_final_jobs_exit0':True,'audit':s['audit'],'final_prediction_seals_checked':seal_count},
  'next_step_proposed_not_started':'Retain structured response qualification; if extending, isolate observer/noise versus update objective and compare equal supervision, tune stopping only on new development, then new held-out. No VLA/robot launch.'
 }
 save_json(out/'ROOT_ACCEPTANCE.json',accept)
 lines=['## 主Agent最终判断','',
  '**本轮通过有界机制验证：明确动作响应的结构下，梯度适配能利用交互提高控制成功率。它没有证明原P18自由视觉latent设计本身已经修好。**','',
  '主端点固定为S3/U3：数值输入在ID/Mild为100%，Strong为75.68%；DINO几何输入在ID/Mild/Strong为94.08%/71.56%/43.93%。各档更新相对同模型冻结、正确相对交换z的配对95%区间均为正。',
  'DINO的History主端点为98.01%/86.13%/63.54%，全部高于Gradient U3，且差异区间不包含0。DINO元学习版本相对固定先验普通梯度U3没有明确成功率优势。数值输入在有限U3时从学习先验/步长获益，不能外推为视觉或普遍优势。',
  '', '预定U20消融只增加计算，不增加交互：DINO为87.19%/78.52%/80.47%。Strong高于History，但ID更低，不能在测试后逐分布挑选最好的U作为一个统一方法。ID上继续拟合与视觉误差/过拟合的解释相容，尚非独立噪声干预的因果结论。','',
  '此次修改同时引入显式物理latent、位移目标、直接外层响应训练和解析控制器，因此证明的是这套组合能工作，尚未确定P18失败的唯一原因。','',
  'History的S0不在训练S集合内，仅作辅助诊断；普通梯度、最小二乘和Oracle完整保留，不把经典辨识本来可实现的效果归为元学习独有贡献。','',
  '### 全部预定S/U曲线','']
 for kind in KINDS:
  lines += [f'**{kind}：S固定3，只改变计算更新次数U**','', '|U|ID|Mild|Strong|','|---:|---:|---:|---:|']
  for u in [0,1,3,10,20]:lines.append('|'+str(u)+'|'+'|'.join(f"{s['final'][st][kind]['metrics'][f'gradient_s3_u{u}']['success']*100:.2f}%" for st in ['id','mild','strong'])+'|')
  lines += ['',f'**{kind}：U固定3，只改变物理交互次数S**','', '|S|ID|Mild|Strong|','|---:|---:|---:|---:|']
  for k in [0,1,2,3,5]:lines.append('|'+str(k)+'|'+'|'.join(f"{s['final'][st][kind]['metrics'][f'gradient_s{k}_u3']['success']*100:.2f}%" for st in ['id','mild','strong'])+'|')
  lines.append('')
 lines += ['### 注释视频','', '[ID救回案例](media/id_rescue_annotated.mp4) · [Strong OOD失败案例](media/strong_failure_annotated.mp4)','', '主Agent已经检查视频最终画面。视频直接绘制封存轨迹，不重新推理或更改统计。']
 p=out/'REPORT_ZH.md';p.write_text(p.read_text()+'\n'+'\n'.join(lines)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for kind in KINDS:
  fig,axs=plt.subplots(2,3,figsize=(13,7));us=[0,1,3,10,20]
  for col,st in enumerate(['id','mild','strong']):
   m=s['final'][st][kind]['metrics']
   for row,key,title in [(0,'response_mse','Fresh-action prediction MSE'),(1,'error','Final goal error')]:
    ax=axs[row,col];vals=[m[f'gradient_s3_u{u}'][key] for u in us];ax.plot(us,vals,'-o');ax.set(title=st.upper()+' | '+title,xlabel='Gradient steps U (same S=3)');ax.grid(alpha=.2)
    if row==0:ax.set_yscale('log')
  fig.suptitle(kind+' | diagnostic predictions not used for adaptation');fig.tight_layout();fig.savefig(out/(kind+'_prediction_error.png'),dpi=150);plt.close(fig)
 page=out/'index.html';text=page.read_text();notice='<section style="background:#e5f4ee;padding:20px"><b>验收：结构化梯度适配有效，算法普遍优势未证实。</b><p>主设置S3/U3：DINO 94.08% / 71.56% / 43.93%，同模型冻结为28.26% / 23.36% / 27.29%。History为98.01% / 86.13% / 63.54%。</p><p>U20是预定计算量消融，不是测试后更换主端点。额外几何监督、明确仿射模型、解析控制和GT定位均为当前限制。</p><a href="ROOT_ACCEPTANCE.json">完整验收与限制</a></section>'
 text=text.replace('<h2>numeric</h2>',notice+'<h2>numeric</h2>',1)
 for kind in KINDS:text=text.replace('<img width="100%" src="'+kind+'_success.png">','<img width="100%" src="'+kind+'_success.png"><img width="100%" src="'+kind+'_prediction_error.png">')
 page.write_text(text)
 # Evidence index binds all metrics, predictions, raws, checkpoints and display derivatives.
 selected=[ROOT/'PROTOCOL.json',ROOT/'TRAIN_FREEZE.json',ROOT/'FINAL_FREEZE.json',ROOT/'QUALIFICATION.json',ROOT/'FINAL_RECEIPT.json',out/'SUMMARY.json',out/'REPORT_ZH.md',out/'ROOT_ACCEPTANCE.json',out/'AUDIT.json',out/'media/VIDEO_INDEX.json']
 selected+=list((ROOT/'final').glob('*/*.pt'))+list((ROOT/'train').glob('*/*/seed*/model.pt'))+list((out/'media').glob('*.mp4'))
 save_json(ROOT/'EVIDENCE_INDEX.json',{'format':'prove.p19-evidence-index.v1','artifacts':[{'ref':str(p),'sha256':digest(p)} for p in selected]})
 print(json.dumps(accept,ensure_ascii=False))
if __name__=='__main__':main()
