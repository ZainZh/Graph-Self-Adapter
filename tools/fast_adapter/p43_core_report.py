"""Aggregate exactly the 96 predeclared P43 development cells and audits."""
from pathlib import Path
import argparse,json
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='evaluations/fast_adapter_p43_neural_geometry_v01');args=parser.parse_args();root=Path(args.root)
    rows={};expected=[]
    for task,base in [('cup',430100),('insert',431100)]:
        for model in (11,22):
            for index in range(8):
                split='id' if index<4 else 'geometry';seed=base+index
                for arm in ('fixed','continuous','prefix'):
                    key=(task,model,split,seed,arm);expected.append(key)
                    path=root/'development'/f'{task}_seed{model}'/f'{split}_{seed}_{arm}'/'RESULT.json'
                    if not path.exists():raise ValueError(f'Expected development result missing: {path}')
                    row=json.loads(path.read_text());row['model_seed']=model;row['result_path']=str(path)
                    if (row['task'],row['split'],row['seed'],row['arm'])!=(task,split,seed,arm):raise ValueError('Result identity mismatch')
                    rows[key]=row
    if len(list((root/'development').glob('*/*/RESULT.json')))!=96:raise ValueError('Unexpected extra development cells')
    audits=[]
    for task in ('cup','insert'):
        for model in (11,22):
            path=root/'development'/f'{task}_seed{model}'/'AUDIT.json'
            if not path.exists():raise ValueError(f'Missing audit: {path}')
            record=json.loads(path.read_text())
            if record['cells']!=24:raise ValueError(f'Incomplete audit: {path}')
            audits.append(record)
    summaries=[];gates=[];prefix_checks=[]
    for task in ('cup','insert'):
        for split in ('id','geometry'):
            for model in (11,22):
                arms={arm:[r for k,r in rows.items() if k[0]==task and k[1]==model and k[2]==split and k[4]==arm] for arm in ('fixed','continuous','prefix')}
                counts={arm:sum(r['success'] for r in rs) for arm,rs in arms.items()}
                pairs=[]
                for f in arms['fixed']:
                    c=rows[(task,model,split,f['seed'],'continuous')];p=rows[(task,model,split,f['seed'],'prefix')]
                    pairs.append(dict(seed=f['seed'],fixed=f['success'],continuous=c['success'],prefix=p['success']))
                summaries.append(dict(task=task,split=split,model_seed=model,count=4,successes=counts,pairs=pairs,
                    wins=sum(p['continuous'] and not p['fixed'] for p in pairs),harms=sum(p['fixed'] and not p['continuous'] for p in pairs)))
                for pair in pairs:
                    traces=[]
                    for arm in ('fixed','continuous','prefix'):
                        path=Path(rows[(task,model,split,pair['seed'],arm)]['result_path']).parent/'TRACE.npy'
                        trace=np.load(path,allow_pickle=True)
                        traces.append(np.array([np.r_[r['command_p'],r['command_R'].ravel(),r['closed']] for r in trace[:12]]))
                    equal=bool(all(np.array_equal(traces[0],t) for t in traces[1:]));prefix_checks.append(equal)
        ood=[s for s in summaries if s['task']==task and s['split']=='geometry']
        differences=[s['successes']['continuous']-s['successes']['fixed'] for s in ood]
        behavior=bool(min(differences)>=0 and sum(differences)>0 and sum(s['wins'] for s in ood)>0)
        technical=bool(all(a['passed'] for a in audits if f'/{task}_seed' in a['audits'][0]['cell']) and all(prefix_checks) and all(r['neural_weights_unchanged'] for r in rows.values()))
        gates.append(dict(task=task,technical=technical,behavior=behavior,qualified=technical and behavior,
            ood_success_difference_by_model=dict(zip(['11','22'],differences)),independent_started=False))
    result=dict(scope='Complete bounded development; no independent outcome is included',cells=96,summaries=summaries,gates=gates,
                common_first12_actions_equal=all(prefix_checks),audits_passed=all(a['passed'] for a in audits),
                independent_rule='Root must freeze hashes and new exact scope before independent rollout')
    out=root/'training/DEVELOPMENT_SUMMARY.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
