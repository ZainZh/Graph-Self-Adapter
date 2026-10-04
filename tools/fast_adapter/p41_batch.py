import argparse,json,time,hashlib
from pathlib import Path
from tools.fast_adapter.p41_run import rollout,ROOT,js

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--phase',choices=['engineering_v03','development_v01'],required=True);a=p.parse_args()
    if a.phase=='engineering_v03':
        cells=[(s,410100+i,arm) for s in ('id','geometry') for i in range(2) for arm in ['oracle']]
    else:
        freeze=json.loads((ROOT/'DEV_FREEZE.json').read_text())
        for path,h in freeze['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
        cells=[(s,seed,arm) for s in freeze['splits'] for seed in freeze['seeds'] for arm in freeze['arms']]
    rows=[]
    for split,seed,arm in cells:
        out=ROOT/a.phase/a.task/f'{split}_{seed}_{arm}'
        rows.append(rollout(a.task,split,seed,arm,out,video=(seed==cells[0][1])))
        js(ROOT/a.phase/(a.task+'_PROGRESS.json'),dict(done=len(rows),total=len(cells),time=time.time()))
    js(ROOT/a.phase/(a.task+'_DONE.json'),dict(rows=rows,time=time.time()))

if __name__=='__main__':main()
