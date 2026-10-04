"""Aggregate every predeclared P20 cell, with paired domain/seed uncertainty."""
from pathlib import Path
import json,html
import numpy as np
import torch
from prove.fast_adapter.insert_adapter import InteractionAdapter
from tools.fast_adapter.p18_run import digest,save_json
R=Path('evaluations/fast_adapter_p20_insert3d_v01');SEEDS=[11,22,33]
LABELS={'base':'基础控制 Base','strong_fixed':'固定增强反馈','oracle':'真动力学 Oracle','history':'纯 History','hybrid_u0':'同模型冻结 z（U0）','hybrid_u3':'梯度适配（U3）','hybrid_swapped':'交换更新后的 z'}

def boot(x):
    rng=np.random.default_rng(97023);s,n=x.shape
    vals=[x[np.ix_(rng.integers(s,size=s),rng.integers(n,size=n))].mean() for _ in range(4000)]
    return np.percentile(vals,[2.5,97.5]).tolist()

def raw(stage,st,seed,arm):return torch.load(R/stage/st/f'seed{seed}'/f'{arm}_RAW.pt',weights_only=False,map_location='cpu')

def aggregate(stage,st):
    outcomes={};metrics={};diagnostics=[]
    for arm in LABELS:
        seeds=[0] if arm in ['base','strong_fixed','oracle'] else SEEDS
        values=np.stack([raw(stage,st,s,arm)['success'].astype(float) for s in seeds]);outcomes[arm]=values
        metrics[arm]={'rate':float(values.mean()),'n_domains':values.shape[1],'model_seeds':seeds,'per_seed_successes':values.sum(1).astype(int).tolist(),'per_seed_rates':values.mean(1).tolist()}
    pairs={}
    for other in ['base','strong_fixed','hybrid_u0','history','hybrid_swapped']:
        d=outcomes['hybrid_u3']-outcomes[other]
        pairs['u3_vs_'+other]={'delta':float(d.mean()),'CI95':boot(d),'per_seed_delta':d.mean(1).tolist(),'wins':(d>0).sum(1).tolist(),'harms':(d<0).sum(1).tolist()}
    for s in SEEDS:
        p=R/stage/st/f'seed{s}';seal=torch.load(p/'LATENTS_BEFORE_QUERY.pt',weights_only=False,map_location='cpu')
        assert digest(p/'LATENTS_BEFORE_QUERY.pt')==json.loads((p/'LATENTS_BEFORE_QUERY.json').read_text())['sha256']
        net=InteractionAdapter().eval();net.load_state_dict(torch.load(R/'models'/f'seed{s}/hybrid.pt',weights_only=False,map_location='cpu')['state'])
        data=raw(stage,st,s,'hybrid_u0');v=data['visuals'].float();b=torch.from_numpy(data['base_actions']).float();a=torch.from_numpy(data['actions']).float()
        z0=seal['latents']['hybrid_u0'];z3=seal['latents']['hybrid_u3'];sw=seal['latents']['hybrid_swapped']
        torch.testing.assert_close(sw,z3.roll(1,0),atol=0,rtol=0)
        with torch.no_grad():
            a0=net.action(v,b,z0[:,None,:].expand(-1,45,-1));a3=net.action(v,b,z3[:,None,:].expand(-1,45,-1))
            # Evaluate fresh-action prediction on the identical U0 trajectory.
            p0=net.predict(v[:,:-1],a[:,:-1],z0[:,None,:].expand(-1,44,-1))
            p3=net.predict(v[:,:-1],a[:,:-1],z3[:,None,:].expand(-1,44,-1))
        action_audit_error=float((a0-a).abs().max())
        assert action_audit_error<.002,action_audit_error # stored visual PCA values quantized to float16
        result=json.loads((p/'RESULT.json').read_text())['diagnostics']
        diagnostics.append({'seed':s,**result,'same_observation_action_change_rms':float((a3-a0).square().mean().sqrt()),'action_recompute_max_error_fp16':action_audit_error,'fresh_query_prediction_mse_u0':float((p0-v[:,1:]).square().mean()),'fresh_query_prediction_mse_u3':float((p3-v[:,1:]).square().mean()),'identity_prediction_mse':float((v[:,:-1]-v[:,1:]).square().mean())})
    return {'metrics':metrics,'pairs':pairs,'diagnostics':diagnostics}

def audit():
    count=0;domains=0
    for name in ['DATA_FREEZE.json','TRAIN_FREEZE.json','FINAL_FREEZE.json']:
        record=json.loads((R/name).read_text())
        files=record.get('files',[])
        if isinstance(files,dict):files=[{'path':p,'sha256':h} for p,h in files.items()]
        for f in files:assert digest(f.get('path',f.get('ref')))==f['sha256'],f
    for stage,st,n in [('dev','train',32),('final','train',64),('final','ood',64)]:
        reference=raw(stage,st,0,'base')['positions'][:,0]
        for seed in [0]+SEEDS:
            p=R/stage/st/f'seed{seed}';receipt=json.loads((p/'RESULT.json').read_text())
            for arm,m in receipt['metrics'].items():
                f=p/f'{arm}_RAW.pt';assert digest(f)==m['raw_sha256']
                r=torch.load(f,weights_only=False,map_location='cpu');pos=r['positions']
                assert pos.shape==(n,46,3) and r['actions'].shape==(n,45,3)
                assert np.isfinite(pos).all() and np.isfinite(r['actions']).all()
                np.testing.assert_array_equal(pos[:,0],reference)
                assert np.max(np.abs(r['actions']))<=1.000001
                inside=(np.abs(pos[:,:,:2])+[.014,.0125]<=[.0177,.0162]).all(-1)
                bottom=pos[:,:,2]-.030
                penetration=np.array([[r['facts'][t][i]['max_penetration'] for t in range(46)] for i in range(n)])
                inserted=inside&(bottom>=.0065)&(bottom<=.013)&(penetration<=.0015)
                from_facts=np.array([[r['facts'][t][i]['inserted'] for t in range(46)] for i in range(n)])
                np.testing.assert_array_equal(inserted,from_facts)
                success=inserted[:,-3:].all(-1)
                np.testing.assert_array_equal(success,r['success']);assert int(success.sum())==m['successes']
                count+=1;domains+=n
    return {'passed':True,'complete_arm_files':count,'correlated_episode_results':domains,'unique_final_domains':128,'unique_dev_domains':32,'frozen_hashes_verified':True,'success_independently_recomputed':True,'paired_initial_states_identical':True,'all_actions_bounded':True,'note':'3 training seeds on the same domains are correlated; no inflated independent sample count.'}

def main():
    torch.set_num_threads(2);out=R/'summary';out.mkdir(exist_ok=True)
    summary={'task':'3D translation insertion, self-moving block, real contact, no robot','dev':aggregate('dev','train'),'final':{st:aggregate('final',st) for st in ['train','ood']},'audit':audit()}
    save_json(out/'SUMMARY.json',summary);save_json(out/'AUDIT.json',summary['audit'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    short=['Base','Fixed gain','Oracle','History','Hybrid U0','Hybrid U3','Swapped z'];fig,ax=plt.subplots(figsize=(11,4.7));x=np.arange(7)
    for j,st in enumerate(['train','ood']):
        ax.bar(x+(j-.5)*.36,[summary['final'][st]['metrics'][a]['rate']*100 for a in LABELS],.36,label='ID' if st=='train' else 'OOD')
    ax.set_xticks(x,short);ax.set_ylabel('Insertion success (%)');ax.set_ylim(0,106);ax.grid(axis='y',alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'success.png',dpi=160);plt.close(fig)
    lines=['# P20：三维插入的视觉交互适配','',
        '场景为两个物体、一个任务：受控物块插入木块槽。使用与现有LIBERO栈同版本的MuJoCo，力受限伺服执行动作并保留槽壁接触；这是自定义任务，不是官方LIBERO任务。物块只平移、姿态固定，不接机械臂。','',
        '单全局相机的真实RGB → 冻结DINOv3的A/B局部特征（各1024维） → 训练图像无监督PCA（合计64维） → 5次交互的GRU历史latent z（32维）。预测器拟合下一视觉状态，测试时做3次梯度更新，只改z；网络权重不变。动作网络输出叠加到固定基座上的有界修正。z没有人为指定的物理语义，普通adapter没有解析动力学求逆。','',
        '**权限与监督：**GT物体框用于ROI池化但坐标不作为网络特征；基座用物块本体位置与已知目标；训练使用Oracle动作标签，各方法相同。没有额外几何读出监督；不是完全无GT端到端视觉控制。','',
        '**预算：**1024训练域、14336张图像；3个训练种子。每对模型共享4000步warmup，再各训2000步；末checkpoint固定，未根据dev或final调参。梯度分支具有更多计算开销，不能说相同计算量。Dev32域；最终ID/OOD各64新域、每域1起点，所有方法配对。3个模型种子重复同一域，不算192个独立域。','',
        '|方法|ID成功率|OOD成功率|','|---|---:|---:|']
    for a,label in LABELS.items():lines.append('|'+label+'|'+'|'.join(f"{100*summary['final'][st]['metrics'][a]['rate']:.2f}%" for st in ['train','ood'])+'|')
    lines += ['', '各模型种子的成功数（每个分母64）：','']
    for a in ['history','hybrid_u0','hybrid_u3','hybrid_swapped']:
        lines.append(f"- {LABELS[a]}：ID {summary['final']['train']['metrics'][a]['per_seed_successes']}；OOD {summary['final']['ood']['metrics'][a]['per_seed_successes']}。")
    lines += ['', '配对差异：梯度U3相对各对照；95%逐点bootstrap按域和模型种子重采样。只有3个训练种子，区间仅反映本设定的不确定性。','']
    for st in ['train','ood']:
        for other in ['hybrid_u0','history','hybrid_swapped','base','strong_fixed']:
            p=summary['final'][st]['pairs']['u3_vs_'+other];lo,hi=p['CI95'];lines.append(f"- {st} 对 {LABELS[other]}：{100*p['delta']:+.2f}pp [{100*lo:+.2f}, {100*hi:+.2f}]。")
    lines += ['', '### 适配链条诊断','', '|分布|模型种子|支持损失 U0→U3|同观测动作变化RMS|新动作预测MSE U0→U3|','|---|---:|---:|---:|---:|']
    for st in ['train','ood']:
        for d in summary['final'][st]['diagnostics']:
            lines.append(f"|{st}|{d['seed']}|{d['support_initial_loss']:.4f}→{d['support_updated_loss']:.4f}|{d['same_observation_action_change_rms']:.6f}|{d['fresh_query_prediction_mse_u0']:.4f}→{d['fresh_query_prediction_mse_u3']:.4f}|")
    lines += ['', '支持损失下降只代表拟合已有交互；动作改变也不自动等于插入更好。新动作预测误差在相同U0执行轨迹上比较，属于诊断，不用它选择模型。','', '全部失败、固定增强反馈对照和Oracle都计入；原始轨迹、模型、输入与冻结哈希均保留。最终数据已打开，后续只能当开发材料，不可再次宣称新测试。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    page='<!doctype html><meta charset="utf-8"><title>P20 三维插入</title><style>body{font:17px/1.65 sans-serif;max-width:1160px;margin:36px auto;padding:0 25px;color:#18352f}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #cdd8d4}video{width:100%;max-width:1100px}pre{white-space:pre-wrap}</style><h1>P20：真实三维插入的视觉交互适配</h1><p>两个物体，一个插入任务；物块直接移动，真实接触，不接机械臂。</p><img width="100%" src="success.png"><p><a href="REPORT_ZH.md">完整中文报告</a> · <a href="SUMMARY.json">全部统计与诊断</a> · <a href="AUDIT.json">原始证据验收</a></p><table><tr><th>方法</th><th>ID</th><th>OOD</th></tr>'
    for a,label in LABELS.items():page+='<tr><td>'+label+'</td>'+''.join(f"<td>{100*summary['final'][st]['metrics'][a]['rate']:.2f}%</td>" for st in ['train','ood'])+'</tr>'
    page+='</table><p>每分布64个独立域；学习方法报告3个训练种子的平均。固定基座使用本体位置及已知目标，ROI来自真值，训练用Oracle动作监督。</p><div id="videos"></div>'
    (out/'index.html').write_text(page)
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
