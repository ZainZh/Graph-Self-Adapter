"""All-384 P43 fixed-vs-continuous inference at the condition level."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='evaluations/fast_adapter_p43_neural_geometry_v01');args=parser.parse_args();root=Path(args.root)
    gate=json.loads((root/'ROOT_GATES.json').read_text());plan=gate['independent_plan_if_qualified'];statistics=plan['statistics'];results={};audits=[];bindings={}
    for task in ('cup','insert'):
        for model_seed in (11,22):
            freeze_path=root/'independent_freeze'/f'{task}_seed{model_seed}.json';frozen=json.loads(freeze_path.read_text());out=Path(frozen['execution']['out'])
            expected={(split,int(seed),arm) for seed,split in zip(frozen['seeds'],frozen['splits']) for arm in frozen['arms']}
            if len(expected)!=96:raise ValueError('Wrong frozen independent size')
            files=list(out.glob('*/RESULT.json'))
            if len(files)!=96:raise ValueError('All 384 cells must complete before any statistics')
            audit=json.loads((out/'AUDIT.json').read_text())
            if not audit['passed'] or audit['cells']!=96 or audit['physics_replays']!=4 or not audit['common_first12_actions_equal']:raise ValueError('Independent audit incomplete or failed')
            audits.append(audit)
            for path in files:
                audited=next(a for a in audit['audits'] if Path(a['cell']).name==path.parent.name)
                if sha(path)!=audited['result_sha256']:raise ValueError('Result changed after audit')
                row=json.loads(path.read_text());identity=(row['split'],row['seed'],row['arm'])
                if identity not in expected or row['task']!=task or row['scope']!='independent_validation':raise ValueError('Frozen condition identity mismatch')
                key=(task,model_seed,*identity)
                if key in results:raise ValueError('Duplicate result')
                results[key]=row
            bindings[f'{task}_seed{model_seed}']=dict(freeze_sha256=sha(freeze_path),checkpoint_sha256=frozen['checkpoint_sha256'],audit_sha256=sha(out/'AUDIT.json'))
    if len(results)!=384:raise ValueError('Incomplete independent results')
    rng=np.random.default_rng(statistics['bootstrap_seed']);rows=[]
    for task in ('cup','insert'):
        for split in ('id','geometry'):
            base,last=plan['fresh_seeds'][task][split];seeds=list(range(base,last+1));n=len(seeds)
            fixed=np.array([[float(results[(task,m,split,s,'fixed')]['success']) for m in (11,22)] for s in seeds])
            continuous=np.array([[float(results[(task,m,split,s,'continuous')]['success']) for m in (11,22)] for s in seeds])
            delta=(continuous-fixed).mean(1)
            item=dict(task=task,split=split,conditions=n,model_seeds=[11,22],fixed_success=float(fixed.mean()),continuous_success=float(continuous.mean()),paired_difference=float(delta.mean()),
                by_model={str(m):dict(fixed_successes=int(fixed[:,j].sum()),continuous_successes=int(continuous[:,j].sum()),denominator=n,
                    rescues=int(((continuous[:,j]==1)&(fixed[:,j]==0)).sum()),harms=int(((continuous[:,j]==0)&(fixed[:,j]==1)).sum())) for j,m in enumerate((11,22))},
                condition_records=[dict(seed=s,fixed=fixed[j].tolist(),continuous=continuous[j].tolist(),mean_difference=float(delta[j])) for j,s in enumerate(seeds)])
            if split=='geometry':
                sample=rng.integers(0,n,size=(statistics['condition_bootstrap'],n));means=delta[sample].mean(1)
                null=(rng.choice(np.array([-1.,1.]),size=(100000,n))*delta).mean(1)
                item.update(ci95=np.quantile(means,[.025,.975]).tolist(),ci97_5_bonferroni=np.quantile(means,[.0125,.9875]).tolist(),
                    one_sided_sign_flip_p=float((1+np.count_nonzero(null>=delta.mean()-1e-12))/(len(null)+1)),
                    inference_unit='condition average across two frozen checkpoints; no arbitrary training-seed inference')
            rows.append(item)
    primary=[r for r in rows if r['split']=='geometry'];ordered=sorted(primary,key=lambda r:r['one_sided_sign_flip_p']);previous=0.
    for rank,row in enumerate(ordered):
        previous=max(previous,min(1.,row['one_sided_sign_flip_p']*(2-rank)));row['holm_adjusted_p']=previous
        row['acceptance']=bool(row['paired_difference']>0 and previous<.05)
    summary=dict(scope='Independent frozen same-checkpoint z-update comparison',cells=384,conditions_per_task={'id':16,'geometry':32},
        source_bindings=bindings,postprocessing_sources={p:sha(p) for p in ('tools/fast_adapter/p43_core_independent_audit.py','tools/fast_adapter/p43_core_independent_report.py')},statistics=statistics,summaries=rows,audits_passed=True,full_physics_replays=sum(a['physics_replays'] for a in audits),
        maximum_replay_errors={k:max(r['maximum_error'][k] for a in audits for r in a['audits']) for k in audits[0]['audits'][0]['maximum_error']},
        limits=gate['limits'])
    destination=root/'training/INDEPENDENT_SUMMARY.json';destination.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='summaries'},indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='condition_records'} for r in rows],indent=2))

if __name__=='__main__':main()
