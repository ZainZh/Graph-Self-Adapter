#!/usr/bin/env python3
"""Create a local P52 variants source/data delta; existing assets are excluded."""
import argparse
import json
from pathlib import Path
import tarfile
from p52_deployment import ROOT, source_closure, record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    entries=['tools/task_graph/collect_libero_p52_variants.py','tools/task_graph/collect_libero_p52_v02.py',
             'tools/task_graph/p52_collect_queue.py','tools/task_graph/p52_queue_manifest.py',
             'tools/task_graph/run_libero_p51_v02.py','tools/task_graph/train_libero_precision_p51.py']
    files=set(source_closure(ROOT,entries))
    files.update(['evaluations/fast_adapter_p52_stage_local_v01/DEPLOY_VARIANTS.sh', 'schemas/task_graph_v1.schema.json',
      'evaluations/fast_adapter_p49_libero_graph_v01/graph_budget_v02/TASK_GRAPH.json',
      'evaluations/fast_adapter_p51_libero_complete_v01/offline_v01/action_head.pt',
      'evaluations/fast_adapter_p51_libero_complete_v01/offline_v01/TRAINING_FREEZE.json',
      'evaluations/fast_adapter_p48_task_graph_v01/gt_replay_v01/frame_0057.npz'])
    files.update(f'evaluations/fast_adapter_p52_stage_local_v01/teacher_smoke_v02/entry_{i:02d}/{name}' for i in range(4) for name in ('RESULT.json','SEAL.json'))
    manifest={f:record(ROOT/f) for f in sorted(files)}
    manifest_path=a.output.with_suffix(a.output.suffix+'.manifest.json')
    if a.output.exists() or manifest_path.exists():raise FileExistsError(a.output)
    with tarfile.open(a.output,'x:gz') as archive:
        for f in sorted(files):archive.add(ROOT/f,arcname=f,recursive=False)
    with manifest_path.open('x') as f:
        json.dump(dict(files=manifest,bundle=record(a.output),requires='existing initial P52 teacher source/data deployment and assets; regenerate remote v02 teacher then collection binding'),f,indent=2);f.write('\n')
    print(json.dumps(dict(bundle=str(a.output),manifest=str(manifest_path),files=len(files))))


if __name__=='__main__':main()
