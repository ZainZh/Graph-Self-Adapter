"""Merge completed disjoint workers; reject missing, duplicated or mixed-model data."""
import argparse,json,time
from pathlib import Path

ARMS={'base'}|{f'{name}_seed{seed}' for name in ('fast_correct','fast_mean','fast_donor','history','robot') for seed in (11,22)}

def combine(workers, expected_offsets, task, stratum, round_name):
    offsets=[o for w in workers for o in w['blocks']]
    assert len(offsets)==len(set(offsets)) and set(offsets)==set(expected_offsets), 'Missing or duplicated blocks'
    weights=workers[0]['weights'];rows=[]
    for w in workers:
        assert (w['task'],w['stratum'],w['round'],w['phase'])==(task,stratum,round_name,'final')
        assert w['networks_unchanged'] and w['initial_states_matched'] and w['weights']==weights
        assert w['conditions']==8*len(w['blocks'])
        assert w['query_episodes']==w['conditions']*11 and w['support_episodes']==w['conditions']*2
        for offset in w['blocks']:
            block=[r for r in w['rows'] if r['offset']==offset]
            assert len(block)==11 and {r['arm'] for r in block}==ARMS, 'Missing or duplicated comparison arm'
            assert all(len(r['success'])==len(r['terminal_success'])==8 for r in block)
        assert len(w['rows'])==11*len(w['blocks'])
        rows.extend(w['rows'])
    n=len(expected_offsets)*8
    return {'task':task,'stratum':stratum,'round':round_name,'phase':'final','conditions':n,
            'query_episodes':n*11,'support_episodes':n*2,'networks_unchanged':True,'initial_states_matched':True,
            'weights':weights,'rows':sorted(rows,key=lambda r:(r['offset'],r['arm'])),
            'seconds_sum_workers':sum(w['seconds'] for w in workers),'partitioned_independent_blocks':True}

def main(task,stratum):
    from tools.fast_adapter.p29_common import task_root,save_json,digest
    root=task_root(task);f=json.loads((root/'FINAL_FREEZE.json').read_text());rnd=f['round'];out=root/'final'/rnd/stratum
    grid=json.loads((root/'FINAL_GRID_FREEZE.json').read_text());wanted=[w for w in grid['workers'] if w['stratum']==stratum]
    paths=[out/f'WORKER_{w["worker"]}.json' for w in wanted];workers=[json.loads(p.read_text()) for p in paths]
    for p,w,assignment in zip(paths,workers,wanted):assert w['worker']==assignment['worker'] and w['blocks']==assignment['blocks']
    d=combine(workers,list(range(0,32 if stratum=='canonical' else 64,8)),task,stratum,rnd)
    for row in d['rows']:assert digest(row['path'])==row['sha256'],row['path']
    d.update(time=time.time(),worker_receipts=[{'path':str(p),'sha256':digest(p)} for p in paths]);assert not (out/'RESULT.json').exists();save_json(out/'RESULT.json',d)
    print('MERGED',task,stratum,d['conditions'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--stratum',required=True);a=p.parse_args();main(a.task,a.stratum)
