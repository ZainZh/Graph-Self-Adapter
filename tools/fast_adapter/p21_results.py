"""Paired P21 control statistics and stage gate, without adaptive re-selection."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from tools.fast_adapter.p18_run import digest,save_json
R=Path('evaluations/fast_adapter_p21_frontends_v01');SEEDS=[11,22,33]
ARMS=['history','hybrid_u0','hybrid_u3','hybrid_swapped']

def raw(stage,st,rep,seed,arm):
    return torch.load(R/'frontends'/rep/stage/st/f'seed{seed}'/f'{arm}_RAW.pt',weights_only=False)

def paired(a,b):
    delta=a.astype(float)-b.astype(float);rng=np.random.default_rng(21023)
    # Seed and domain are crossed; preserve common domains across all seeds/arms.
    draws=[]
    for _ in range(5000):
        si=rng.integers(0,len(a),len(a));di=rng.integers(0,a.shape[1],a.shape[1])
        draws.append(delta[np.ix_(si,di)].mean()*100)
    return {'delta_pp':float(delta.mean()*100),'ci95_pp':np.quantile(draws,[.025,.975]).tolist(),
            'wins':int((a&~b).sum()),'harms':int((~a&b).sum()),'ties':int((a==b).sum()),
            'seed_deltas_pp':(delta.mean(1)*100).tolist(),'n_domains':a.shape[1],'n_model_seeds':len(a),
            'count_note':'win/harm counts are correlated seed-by-domain trials; CI resamples seed and domain axes'}

def main(stage):
    selection=json.loads((R/'diagnostic/RESULT.json').read_text());reps=selection['control_frontends']
    assert len(reps)==2
    strata=['train'] if stage=='dev' else ['train','ood'];table={};comparisons={};bindings=[]
    for st in strata:
        arrays={};table[st]={};comparisons[st]={}
        for rep in reps:
            arrays[rep]={};table[st][rep]={}
            for arm in ARMS:
                runs=[raw(stage,st,rep,s,arm) for s in SEEDS]
                arr=np.stack([r['success'] for r in runs]);arrays[rep][arm]=arr
                # Verify stored success against the unchanged final insertion/contact definition.
                for run in runs:
                    pos=run['positions'][:,-3:];inside=(np.abs(pos[:,:,:2])+np.array([.014,.0125])<=np.array([.0175,.016])+.0002).all(-1)
                    bottom=pos[:,:,2]-.030
                    pen=np.array([[f['max_penetration'] for f in row] for row in run['facts'][-3:]]).T
                    check=(inside&(bottom>=.0065)&(bottom<=.013)&(pen<=.0015)).all(1)
                    assert np.array_equal(check,run['success']), 'success audit mismatch'
                table[st][rep][arm]={'rate_percent':float(arr.mean()*100),'successes':int(arr.sum()),'trials':int(arr.size),
                   'rates_by_seed_percent':(arr.mean(1)*100).tolist(),'unique_domains':arr.shape[1]}
            comparisons[st][rep]={
                'u3_vs_u0':paired(arrays[rep]['hybrid_u3'],arrays[rep]['hybrid_u0']),
                'u3_vs_history':paired(arrays[rep]['hybrid_u3'],arrays[rep]['history']),
                'correct_vs_swapped':paired(arrays[rep]['hybrid_u3'],arrays[rep]['hybrid_swapped'])}
        selected=selection['selected'];comparisons[st]['selected_vs_last']={a:paired(arrays[selected][a],arrays['last'][a]) for a in ARMS}
        for arm in ['base','strong_fixed','oracle']:
            r=raw(stage,st,'last',0,arm)
            table[st][arm]={'rate_percent':float(r['success'].mean()*100),'successes':int(r['success'].sum()),'unique_domains':len(r['success'])}
        for rep in reps:
            for path in sorted((R/'frontends'/rep/stage/st).rglob('*_RAW.pt')):
                bindings.append({'path':str(path),'sha256':digest(path)})
    decision=None
    if stage=='dev':
        c=comparisons['train'];sel=selection['selected']
        gradient=c[sel]['u3_vs_u0']['delta_pp']>=3-1e-9
        representation=any(c['selected_vs_last'][a]['delta_pp']>=5-1e-9 for a in ['history','hybrid_u3'])
        decision={'enter_final':gradient or representation,'gradient_gate':gradient,'representation_gate':representation,
          'thresholds_pp':{'gradient':3,'representation':5},'no_retuning':True}
    result={'stage':stage,'strata':strata,'selected':selection['selected'],'metrics':table,'comparisons':comparisons,
          'dev_gate':decision,'raw_bindings':bindings,'success_geometry_audit':True,
          'protocol_sha256':digest(R/'PROTOCOL.json')}
    out=R/'summary';out.mkdir(exist_ok=True);save_json(out/f'{stage.upper()}_RESULTS.json',result)
    print(json.dumps({'stage':stage,'metrics':table,'dev_gate':decision},ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['dev','final']);a=p.parse_args();main(a.stage)
