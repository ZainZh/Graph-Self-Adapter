"""Aggregate sealed P19 outputs; no fitting or test-set selection."""
from pathlib import Path
import json,hashlib,html
import numpy as np
import torch
from tools.fast_adapter.p19_run import ROOT,SEEDS,KINDS,digest,save_json

LABELS={'base':'Base','gradient_s3_u0':'冻结 z（同模型）','gradient_s3_u3':'梯度更新 z，U3','history_s3':'History','fixed_gradient_u3':'固定先验＋梯度 U3','fixed_gradient_u20':'固定先验＋梯度 U20','gradient_s3_u20':'学习先验＋梯度 U20','least_squares':'同观测最小二乘','oracle':'真动力学 Oracle','gradient_swapped':'交换 z','gradient_zero_support':'清零支持位移'}
PRIMARY=list(LABELS)

def bootstrap(x,seed=93404):
    # seed x domain; query resets already clustered inside each domain.
    rng=np.random.default_rng(seed);s,n=x.shape;vals=[]
    for _ in range(2000):
        ids=rng.integers(0,s,s);d=rng.integers(0,n,n)
        vals.append(float(x[np.ix_(ids,d)].mean()))
    return [float(v) for v in np.percentile(vals,[2.5,97.5])]

def collect(split,kind):
    files=[ROOT/split/f'{kind}_seed{s}_RAW.pt' for s in SEEDS]
    allraw=[torch.load(f,weights_only=False,map_location='cpu') for f in files]
    result={'metrics':{},'pairs':{},'raw_sha256':{str(f):digest(f) for f in files}}
    for arm in allraw[0]:
        success=np.stack([r[arm]['success'].float().mean(1).numpy() for r in allraw])
        error=np.stack([r[arm]['error'].mean(1).numpy() for r in allraw])
        result['metrics'][arm]={'success':float(success.mean()),'error':float(error.mean()),'response_mse':float(np.mean([r[arm]['response_mse'].mean() for r in allraw])),'per_seed_success':success.mean(1).tolist(),'fallback_domains_per_seed':[int(r[arm]['fallback'].any(1).sum()) for r in allraw]}
    comparisons=[('gradient_s3_u3','gradient_s3_u0'),('gradient_s3_u3','gradient_swapped'),('gradient_s3_u3','history_s3'),('gradient_s3_u3','fixed_gradient_u3'),('gradient_s3_u3','least_squares'),('gradient_s3_u20','gradient_s3_u0'),('history_s3','history_s0')]
    for left,right in comparisons:
        delta=np.stack([(r[left]['success'].float()-r[right]['success'].float()).mean(1).numpy() for r in allraw])
        error=np.stack([(r[left]['error']-r[right]['error']).mean(1).numpy() for r in allraw])
        result['pairs'][left+'__vs__'+right]={'success_delta':float(delta.mean()),'success_CI95':bootstrap(delta),'error_delta':float(error.mean()),'error_CI95':bootstrap(error),'per_seed_success_delta':delta.mean(1).tolist()}
    return result

def audit():
    issues=[];checks={};rows=0;queries=0
    for freeze in ['TRAIN_FREEZE.json','FINAL_FREEZE.json']:
        data=json.loads((ROOT/freeze).read_text())
        for f,h in data['files'].items():
            if digest(f)!=h:issues.append('changed:'+f)
    for split in ['qualification','final/id','final/mild','final/strong']:
      for kind in KINDS:
        files=list((ROOT/split).glob(f'{kind}_seed*_RAW.pt'))
        if not files:continue
        assert len(files)==5,(split,kind,len(files))
        reference=None
        for f in sorted(files):
            data=torch.load(f,weights_only=False,map_location='cpu')
            preds=torch.load(f.with_name(f.name.replace('_RAW','_PREDICTIONS')),weights_only=False,map_location='cpu')
            for arm,val in data.items():
                frame=val['frames'];error=(frame[-1]-(torch.load(ROOT/('qualification.pt' if split=='qualification' else 'final_'+split.split('/')[-1]+'.pt'),weights_only=False)['qg'])).norm(dim=-1)
                visible_failure=(frame[1:].abs()>1.546875).any(-1).any(0)
                success=(error<=.05)&~visible_failure
                if not torch.allclose(error,val['error'],atol=1e-7):issues.append(str(f)+arm+' error')
                if not torch.equal(success,val['success']):issues.append(str(f)+arm+' success')
                if arm not in ['base','oracle'] and not torch.equal(val['z'],preds[arm]):issues.append(str(f)+arm+' latent changed')
                rows+=len(error);queries+=error.numel()
            b=data['base']['error'];o=data['oracle']['error']
            if reference is None:reference=(b,o)
            elif not(torch.equal(b,reference[0]) and torch.equal(o,reference[1])):issues.append('baseline seed mismatch')
    checks.update(source_and_checkpoint_hashes=True,predictions_sealed_before_scoring=True,success_recomputed_from_frames=True,baseline_seed_invariance=True)
    result={'passed':not issues,'issues':issues,'checks':checks,'correlated_domain_rows':rows,'correlated_query_measurements':queries,'note':'Repeated methods/seeds/query resets are not independent physical domains.'}
    save_json(ROOT/'summary'/'AUDIT.json',result)
    if issues:raise AssertionError(issues)
    return result

def main():
    out=ROOT/'summary';out.mkdir(exist_ok=True)
    gates=json.loads((ROOT/'QUALIFICATION.json').read_text())
    summary={'scope':'P19 structured physical fast adaptation, GT ROI + supervised geometric readout + GT query localization','qualification':{},'final':{},'gate':gates}
    for kind in KINDS:summary['qualification'][kind]=collect('qualification',kind)
    for stratum in ['id','mild','strong']:
        summary['final'][stratum]={}
        for kind in KINDS:
            if (ROOT/'final'/stratum/f'{kind}_seed11_RAW.pt').exists():summary['final'][stratum][kind]=collect('final/'+stratum,kind)
    result=audit();summary['audit']=result
    save_json(out/'SUMMARY.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind in KINDS:
      if not all(kind in summary['final'][s] for s in ['id','mild','strong']):continue
      fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
      for ax,stratum in zip(axes,['id','mild','strong']):
        m=summary['final'][stratum][kind]['metrics'];us=[0,1,3,10,20]
        ax.plot(us,[100*m[f'gradient_s3_u{u}']['success'] for u in us],'-o',label='Meta gradient, S=3')
        for key,label in [('base','Base'),('history_s3','History'),('least_squares','Least squares')]:ax.axhline(100*m[key]['success'],linestyle='--',label=label)
        ax.set(title=stratum.upper(),xlabel='Gradient steps U (same 3 interactions)',ylim=(0,103));ax.grid(alpha=.2)
      axes[0].set_ylabel('Success (%)');axes[-1].legend(loc='lower right',fontsize=8);fig.suptitle(kind+' | shared GT-geometric controller');fig.tight_layout();fig.savefig(out/(kind+'_success.png'),dpi=150);plt.close(fig)
    lines=['# P19：结构化动作响应 Fast Adapter','',
      '本轮先将 latent 明确为六个动作响应参数，使用位移预测误差和共用解析控制器。它同时改变了 P18 的表示、预测和控制结构，是组合修订，不是单因素根因实验。',
      '', '全部方法 query 控制均使用模拟器真实几何位置；视觉版本仅从校准图像的冻结 DINO＋额外几何监督 readout 中估计动作响应。此结果不是无GT视觉控制、VLA或机器人证据。',
      '', '训练：数值/DINO两种输入 × meta-gradient/History两类 × 5个随机种子，每项1500更新。两种方法共享响应预测和oracle动作训练监督、相同观测及控制器；模型容量不同，不作同计算预算优势声明。旧P18结果保持。',
      '', '资格：256个已开放训练域（与P19训练域分开），5 seeds。达到预定门后才进入每档256新域×4新起点的独立测试。S为交互次数，U为计算更新次数；U20没有多收集17次交互。','']
    for kind in KINDS:
        g=gates['gates'][kind];lines += [f'## {kind}',f"资格门：{'通过' if g['passed'] else '未通过'}；更新相对冻结 {100*g['success_delta']:+.2f}pp，相对交换 {100*g['swap_delta']:+.2f}pp。",'']
        if not g['passed']:continue
        lines += ['|方法（S=3）|ID|Mild OOD|Strong OOD|','|---|---:|---:|---:|']
        for arm in PRIMARY:
            values=[summary['final'][s][kind]['metrics'][arm]['success'] for s in ['id','mild','strong']]
            lines.append('|'+LABELS[arm]+'|'+'|'.join(f'{v*100:.2f}%' for v in values)+'|')
        lines += ['',f'![更新次数与成功率]({kind}_success.png)','', '配对变化（95%逐点区间；按环境聚类并重采样训练种子）：']
        for s in ['id','mild','strong']:
            for pair in ['gradient_s3_u3__vs__gradient_s3_u0','gradient_s3_u3__vs__gradient_swapped','gradient_s3_u3__vs__history_s3','gradient_s3_u3__vs__fixed_gradient_u3']:
                v=summary['final'][s][kind]['pairs'][pair];lo,hi=v['success_CI95']
                lines.append(f"- {s} {pair}: {100*v['success_delta']:+.2f}pp [{100*lo:+.2f}, {100*hi:+.2f}]")
        lines.append('')
    lines += ['## 验收与解释边界','',f"原始轨迹复算与冻结审计：{'通过' if result['passed'] else '未通过'}。全部方法、种子和预定S/U均保存于SUMMARY及RAW文件。",'',
        '- 更新相对同模型冻结的改善支持该结构下的梯度适配；固定先验普通梯度也成功时，不能把全部收益归为元学习。',
        '- 同信息最小二乘是这个已知线性模型的重要参照；没有超过它，不宣称梯度算法优势或新颖系统辨识。',
        '- 额外几何监督、明确仿射模型与解析控制器均为额外结构，不与P18作等条件增益比较。',
        '- 当前只能证明这一局部机制，不证明原先自由视觉latent设计已修复或完成机器人RSI。','',
        '视频为封存轨迹的注释展示，不参与指标和样本选择。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    body='<h1>P19 结构化 Fast Adapter</h1><p>六维动作响应 · 额外几何监督 · GT定位辅助控制。S=物理交互次数；U=计算更新次数。</p>'
    for kind in KINDS:
      if not gates['gates'][kind]['passed']:body+=f'<p>{kind}未通过资格门，未进入final。</p>';continue
      body+=f'<h2>{kind}</h2><table><tr><th>方法 S=3</th><th>ID</th><th>Mild OOD</th><th>Strong OOD</th></tr>'
      for arm in PRIMARY:
        body+='<tr><td>'+LABELS[arm]+'</td>'+''.join('<td>'+f"{summary['final'][s][kind]['metrics'][arm]['success']*100:.2f}%"+'</td>' for s in ['id','mild','strong'])+'</tr>'
      body+='</table><img width="100%" src="'+kind+'_success.png">'
    body+='<p><a href="REPORT_ZH.md">完整报告</a> · <a href="SUMMARY.json">全部数据与区间</a> · <a href="AUDIT.json">原始轨迹审计</a></p><div id="videos"></div>'
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>P19结果</title><style>body{max-width:1200px;margin:35px auto;font:17px system-ui;line-height:1.6;color:#183434}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccd}td:not(:first-child){text-align:right}video{width:100%}</style>'+body+'</html>')
    print(json.dumps({'audit':result,'report':str(out/'REPORT_ZH.md')},ensure_ascii=False))
if __name__=='__main__':main()
