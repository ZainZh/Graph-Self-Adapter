#!/usr/bin/env python3
"""Additive export and development replay of the exact frozen v02 selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SCHEMA = 'graph-self-adapter.twin-v02-package-export.v1'
CODE = ('prove/task_graph/twin_graph.py', 'prove/task_graph/twin_physics.py',
        'prove/fast_adapter/twin_local_skill.py', 'tools/task_graph/twin_rsi.py',
        'tools/task_graph/twin_train_skills.py')
TASK_CANDIDATES = {'put_knife_board_new': 'v02_a2', 'put_mug_on_box': 'v02_a2',
                   'sorting_cube_bimanual': 'v02_a3'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+'\n')

def checked(binding):
    path = Path(binding['path']).resolve()
    if not path.is_file() or sha(path) != binding['sha256']:
        raise ValueError(f'external binding mismatch: {path}')
    return path

def check_solution(row, solution, snapshot):
    source = load(snapshot/'SOURCE_FREEZE.json')
    if sha(snapshot/'SOURCE_FREEZE.json') != row['source_freeze_sha256']:
        raise ValueError('selected source freeze mismatch')
    for relative in CODE:
        if sha(snapshot/relative) != source[relative]:
            raise ValueError(f'frozen source mismatch: {relative}')
    for key, relative in [('graph_sha256','graph/task_graph.json'), ('model_sha256','models/local_skill.npz'),
                          ('reference_sha256','refs/visible_centroids.json'), ('candidate_freeze_sha256','CANDIDATE_FREEZE.json')]:
        if sha(solution/relative) != row[key]:
            raise ValueError(f'selected artifact mismatch: {relative}')
    freeze = load(solution/'CANDIDATE_FREEZE.json')
    if freeze['task'] != row['task'] or freeze['candidate_id'] != row['candidate_id']:
        raise ValueError('candidate identity mismatch')
    for relative, expected in freeze['code'].items():
        if source.get(relative) != expected:
            raise ValueError('candidate/source freeze disagreement')
    return freeze

def verify(root):
    manifest = load(root/'EXPORT_MANIFEST.json')
    if manifest['schema'] != SCHEMA or manifest['default_execution_mode'] != 'fixed_z':
        raise ValueError('unsupported export/default mode')
    for item in manifest['files']:
        path = root/item['path']
        if not path.resolve().is_relative_to(root.resolve()) or sha(path) != item['sha256']:
            raise ValueError(f'export file mismatch: {path}')
    original = load(root/'provenance/FINAL_SELECTION.json')
    if sha(root/'provenance/FINAL_SELECTION.json') != manifest['original_selection_sha256']:
        raise ValueError('original selection bytes changed')
    derived = load(root/'DERIVED_RUNTIME_MANIFEST.json')
    if len(derived['tasks']) != 3 or {r['task']:r['candidate_id'] for r in derived['tasks']} != TASK_CANDIDATES:
        raise ValueError('export must contain exact A2/A2/A3 selected tasks')
    original_rows = {r['task']:r for r in original['tasks']}
    external_count = 0
    for row in derived['tasks']:
        selected = original_rows[row['task']]
        for key in ('candidate_id','graph_sha256','model_sha256','reference_sha256','candidate_freeze_sha256','source_freeze_sha256'):
            if row[key] != selected[key]:
                raise ValueError('derived manifest changed the scientific selection')
        solution, snapshot = root/row['solution'], root/row['source_snapshot']
        freeze = check_solution(row, solution, snapshot)
        if freeze['runtime_contract_sha256'] != sha(root/'contracts/RUNTIME_CONTRACT.json'):
            raise ValueError('runtime contract mismatch')
        externals = load(solution/'EXTERNAL_BINDINGS.json')
        for binding in externals['sources'].values():
            checked(binding); external_count += 1
    catalog = load(root/'contracts/EXTERNAL_ASSET_BINDINGS.json')
    for row in catalog['task_assets'].values():
        for path, expected in row['external_asset_hashes'].items():
            checked({'path':path,'sha256':expected}); external_count += 1
    return {'schema':SCHEMA,'tasks':3,'files_verified':len(manifest['files']),
            'external_files_verified':external_count,'manifest_sha256':sha(root/'EXPORT_MANIFEST.json')}

def export(selection_path, output):
    selection = load(selection_path)
    if selection['status'] != 'frozen' or selection['default_execution_mode'] != 'fixed_z':
        raise ValueError('selection/default must already be frozen')
    if len(selection['tasks']) != 3 or {r['task']:r['candidate_id'] for r in selection['tasks']} != TASK_CANDIDATES:
        raise ValueError('v02 export requires exact three selected tasks')
    base = selection_path.resolve().parent
    for row in selection['tasks']:
        check_solution(row, Path(row['packages']), Path(row['source_snapshot']))
    if output.exists():
        raise FileExistsError(f'refusing to overwrite export: {output}')
    output.mkdir(parents=True)
    provenance = output/'provenance'; provenance.mkdir()
    # Retain original selection/contracts byte for byte. Relative runnable paths
    # belong only to the explicitly derived manifest below.
    provenance_names = ['FINAL_SELECTION.json','PARENT_TEST_SELECTION.json','COHORT_TEST_SELECTION.json',
                        'DEVELOPMENT_SELECTION.json','FINAL_TEST_SCOPE.json','FINAL_TEST_CONDITIONS.json',
                        'FINAL_REPORT.json','DEVELOPMENT_B0_A1_BATCH_REPORT.json','DEVELOPMENT_A2_BATCH_REPORT.json',
                        'DEVELOPMENT_A3_BATCH_REPORT.json','DEVELOPMENT_SELECTED_ONLINE_BATCH_REPORT.json']
    for name in provenance_names:
        shutil.copy2(base/name, provenance/name)
    for name in ['ROOT_TASK_PLAN.json','RUNTIME_CONTRACT.json','SCOPE.json','SELECTION_RULE.json',
                 'A2_OVERRIDES.json','A3_OVERRIDES.json','EXTERNAL_ASSET_BINDINGS.json']:
        destination=output/'contracts'/name; destination.parent.mkdir(exist_ok=True)
        shutil.copy2(base/name,destination)
    final=load(base/'FINAL_REPORT.json'); metrics={(r['arm_id'],r['task']):r for r in final['tasks']}
    rows=[]; snapshots=set()
    for selected in selection['tasks']:
        task=selected['task']; name=selected['source_freeze_sha256'][:16]
        if name not in snapshots:
            source=Path(selected['source_snapshot']); destination=output/'source_snapshots'/name
            shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            snapshots.add(name)
        source=Path(selected['packages']); destination=output/'tasks'/f'twin_{task}'/'solution'
        for relative in ['MANIFEST.json','CANDIDATE_FREEZE.json','EXTERNAL_BINDINGS.json']:
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/relative,target)
        for directory in ['graph','models','refs','contracts','adapter']:
            if (source/directory).is_dir():shutil.copytree(source/directory,destination/directory)
        qualification=('independent_validation_regressed; not qualified as an improvement; research reuse only'
                       if task=='sorting_cube_bimanual' else 'frozen research candidate; small simulator validation only; no hardware qualification')
        rows.append({**selected,'solution':str(destination.relative_to(output)),
                     'source_snapshot':f'source_snapshots/{name}','qualification':qualification,
                     'independent_validation':{'parent_successes':metrics[('parent_fixed',task)]['strict_successes'],
                                               'selected_fixed_successes':metrics[('selected_fixed',task)]['strict_successes'],
                                               'denominator':10}})
    dump(output/'DERIVED_RUNTIME_MANIFEST.json',{'schema':'graph-self-adapter.twin-v02-derived-runtime.v1',
         'derived_paths_only':True,'scientific_selection_unchanged':True,'default_execution_mode':'fixed_z',
         'original_selection_sha256':sha(selection_path),'replay_condition_set':'development_v02','replay_seed':51005,
         'online_z_role':'explicit research ablation; unsupported extra benefit','tasks':rows})
    shutil.copy2(Path(__file__),output/'launch.py')
    (output/'run.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
package_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "${PYTHON:-python3}" "$package_root/launch.py" launch --export-root "$package_root" "$@"
''');(output/'run.sh').chmod(0o755)
    (output/'README.md').write_text('''# Twin RSI v02 冻结研究复用包

只包含选定 A2 刀、A2 杯和 A3 分类；模型、图、参考及源码字节保持冻结。默认 fixed-z 在在线开发和独立检测前已承诺。

分类独立检测为 3/10，父图为 6/10：出现退化，不能作为已验证改善；仅限研究复用。刀和杯也未获得真机资格。模拟器语义 mask 和 Boolean supervisors 仍是特权输入。

一条命令只校验并预览，不开始回合：

```bash
bash task_packages/twin_rsi_v02/selected/run.sh --task put_knife_board_new --output /tmp/new-twin-v02-replay --dry-run
```

去掉 `--dry-run` 才执行新模拟器回合；每次必须选择不存在的输出目录。固定使用已开放 development seed 51005；不产生 fresh 检测证据。`--online-z` 仅启用研究对照，在线额外收益未获支持。无真机命令入口。

原 selection 和合同在 provenance/contracts 中原样保留；相对路径仅在 DERIVED_RUNTIME_MANIFEST.json 中推导。标准场景、网格、演示及 Python 环境仍在外部，按原路径与哈希校验。此包不重新训练或执行 Agent。
''')
    files=[{'path':str(p.relative_to(output)),'sha256':sha(p),'bytes':p.stat().st_size}
           for p in sorted(output.rglob('*')) if p.is_file()]
    dump(output/'EXPORT_MANIFEST.json',{'schema':SCHEMA,'default_execution_mode':'fixed_z',
         'original_selection_sha256':sha(selection_path),'tasks':list(TASK_CANDIDATES),
         'classification_qualification':'independent validation regressed; research reuse only',
         'files':files,'bytes':sum(r['bytes'] for r in files)})
    return verify(output)

def launch(args):
    root=args.export_root.resolve(); verified=verify(root)
    manifest=load(root/'DERIVED_RUNTIME_MANIFEST.json');matches=[r for r in manifest['tasks'] if r['task']==args.task]
    if len(matches)!=1:raise ValueError('task not uniquely selected')
    if args.trial not in range(10):raise ValueError('development trial must be 0..9')
    if args.output.exists():raise FileExistsError('choose a fresh output directory')
    row=matches[0]; python=args.python or os.environ.get('TWIN_PYTHON') or '/home/zheng/github/Mujoco-RSI/.venv/bin/python'
    command=[python,str(root/row['source_snapshot']/'tools/task_graph/twin_rsi.py'),
             '--plan',str(root/'contracts/ROOT_TASK_PLAN.json'),'--assets',str(root/'contracts/EXTERNAL_ASSET_BINDINGS.json'),
             '--packages',str(root/'tasks'),'run','--task',args.task,'--candidate-id',row['candidate_id'],
             '--condition-set','development_v02','--condition-seed','51005','--runtime-contract',str(root/'contracts/RUNTIME_CONTRACT.json'),
             '--trial',str(args.trial),'--output',str(args.output.resolve())]
    if args.online_z:command.append('--online-z')
    result={**verified,'dry_run':args.dry_run,'controller':'online_z_research_ablation' if args.online_z else 'fixed_z',
            'qualification':row['qualification'],'command':command}
    if not args.dry_run:
        completed=subprocess.run(command,check=False);result['returncode']=completed.returncode
        if completed.returncode:raise RuntimeError(f'replay failed: {completed.returncode}')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('export');q.add_argument('--selection',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('verify');q.add_argument('--export-root',type=Path,required=True)
    q=sub.add_parser('launch');q.add_argument('--export-root',type=Path,required=True);q.add_argument('--task',required=True)
    q.add_argument('--trial',type=int,default=0);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--online-z',action='store_true',help='explicit research ablation');q.add_argument('--python');q.add_argument('--dry-run',action='store_true')
    a=p.parse_args();result=export(a.selection.resolve(),a.output.resolve()) if a.action=='export' else verify(a.export_root.resolve()) if a.action=='verify' else launch(a)
    json.dump(result,sys.stdout,indent=2,ensure_ascii=False);print()

if __name__=='__main__':main()
