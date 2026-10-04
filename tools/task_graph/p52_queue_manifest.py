#!/usr/bin/env python3
"""Prepare a 24-condition P52 queue manifest without executing any job."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True)
    p.add_argument('--timeout-seconds',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2];binding=a.binding.resolve()
    data=json.loads(binding.read_text());conditions=data['conditions']
    if len(conditions)!=24:raise ValueError('Expected 24 frozen conditions')
    files=[binding,Path(__file__).resolve(),root/'tools/task_graph/p52_collect_queue.py',root/'tools/task_graph/collect_libero_p52_variants.py']
    result=dict(schema='prove.p52-collection-queue.v1',cwd=str(root),attempts_per_job=1,
                bound_files={str(f):sha(f) for f in files},jobs=[])
    for i,condition in enumerate(conditions):
        if condition['index']!=i:raise ValueError('Conditions must retain binding order')
        result['jobs'].append(dict(id=f'job_{i:02d}',seed=condition['condition_seed'],timeout_seconds=a.timeout_seconds,
          argv=[str(a.python.absolute()),str(files[-1]),'collect','--binding',str(binding),'--index',str(i)]))
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'manifest':str(a.output),'sha256':sha(a.output),'jobs':24,'execution_started':False}))


if __name__=='__main__':main()
