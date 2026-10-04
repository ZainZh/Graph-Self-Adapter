"""Independent record-level acceptance of P18, separate from report aggregation."""
import hashlib,json,math
from pathlib import Path
from collections import Counter,defaultdict


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def main():
    root=Path('evaluations/fast_adapter_p18_visual_v01')
    freeze=json.loads((root/'TRAIN_FREEZE.json').read_text())
    for p,h in freeze['files'].items():assert sha(p)==h,p
    final=json.loads((root/'FINAL_FREEZE.json').read_text())
    for p,h in final['checkpoints'].items():assert sha(p)==h,p
    totals=Counter();lookup=defaultdict(list);keys=set();base={}
    arms={'gradient_z':{'gradient_z':{0,1,2,3,5},'gradient_z_swapped':{3},'gradient_z_zero_support_visual':{3},'base':{0},'oracle':{0}},'history':{'history':{0,1,2,3,5},'history_swapped':{3},'history_zero_support_visual':{3}},'hybrid':{'hybrid':{0,1,2,3,5},'hybrid_swapped':{3},'hybrid_zero_support_visual':{3},'hybrid_no_refinement':{3}}}
    for family in arms:
        for seed in (11,22,33,44,55):
            p=root/'final'/family/f'seed{seed}';rec=json.loads((p/'EVAL.json').read_text())
            assert rec['all_model_parameters_unchanged'];assert rec['data_seed']==2026092231
            assert rec['raw_sha256']==sha(p/'raw.jsonl')
            assert rec['checkpoint_sha256']==final['checkpoints'][str(root/'train'/family/f'seed{seed}/model.pt')]
            expected={(st,arm,k,d) for st in ('id','mild','strong') for arm,ks in arms[family].items() for k in ks for d in range(128)}
            actual=set()
            for line in (p/'raw.jsonl').read_text().splitlines():
                r=json.loads(line);cell=(r['stratum'],r['arm'],r['k'],r['domain']);assert cell in expected and cell not in actual;actual.add(cell)
                assert r['seed']==seed and r['family']==family
                assert len(r['query_success'])==len(r['query_error'])==len(r['out_of_frame'])==4
                for s,e,off in zip(r['query_success'],r['query_error'],r['out_of_frame']):
                    assert math.isfinite(e) and e>=0
                    assert s==((e<=.05) and not off)
                    lookup[(r['stratum'],r['arm'],r['k'])].append((int(s),e))
                    totals['query_measurements']+=1;totals['out_of_frame']+=int(off)
                if r['arm'] in ('base','oracle'):
                    bkey=(r['stratum'],r['arm'],r['domain'])
                    vals=(r['query_success'],r['query_error'])
                    if bkey in base:assert base[bkey]==vals,'common physics base mismatch across seeds/machines'
                    base[bkey]=vals
                totals['domain_rows']+=1
            assert actual==expected
            totals['final_jobs']+=1
    summary=json.loads((root/'summary/SUMMARY.json').read_text())
    for (st,arm,k),values in lookup.items():
        if arm not in summary['means'][st]:continue
        success=sum(v[0] for v in values)/len(values);error=sum(v[1] for v in values)/len(values)
        ref=summary['means'][st][arm][str(k)]
        assert abs(success-ref['success'])<1e-12 and abs(error-ref['error'])<1e-12
    out={'status':'passed','counts':dict(totals),'source_and_all_checkpoint_hashes_intact':True,'exact_expected_cell_sets':True,'raw_success_recomputed_from_error_and_visibility':True,'base_and_oracle_physics_identical_across_all_seeds_and_hosts':True,'summary_recomputed_independently':True,'note':'184320 correlated query measurements are not 184320 independent domains; final unique domains=128 per stratum, query4, repeated across methods/model seeds.'}
    (root/'summary/INDEPENDENT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
