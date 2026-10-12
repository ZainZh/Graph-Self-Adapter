#!/usr/bin/env python3
"""V20 task-bound registry; list/check are file-only, execute needs a new freeze.
Assets/old scores are not modern SAM/Gemini or physical deployment qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
INVENTORY=REPO/'evaluations/multitask_runtime_v20/TASK_ASSET_INVENTORY.json'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registry():
    value=json.loads(INVENTORY.read_text())
    if value.get('schema')!='graph-self-adapter.multitask-asset-inventory.v1':raise ValueError('unsupported task registry schema')
    rows={row['task']:row for row in value['tasks']}
    if len(rows)!=10 or len(rows)!=len(value['tasks']):raise ValueError('exact qualified ten-task registry required')
    return value,rows


def check_task(task,mode='sim'):
    value,rows=registry()
    if task not in rows:raise ValueError('unknown/unqualified task; no default task fallback: '+task)
    row=rows[task]
    for binding in [value['qualified_summary'],row['physics_report'],row['historical_runner'],*row['bindings'].values()]:
        if digest(binding['path'])!=binding['sha256']:raise ValueError('task asset binding changed: '+binding['path'])
    graph=json.loads(Path(row['bindings']['graph']['path']).read_text())
    if graph['task'] not in (task,'twin_'+task) or len(graph['stages'])!=row['historical_stage_count']:raise ValueError('selected graph/task binding mismatch')
    manifest=json.loads(Path(row['bindings']['manifest']['path']).read_text())
    if manifest['task']!=task or manifest['model']['training']['input_dim']!=row['model_input_dim']:raise ValueError('task model feature contract mismatch')
    modern=bool(row['modern_executor'])
    if modern:
        evidence=row['modern_pass']['result']
        if digest(evidence['path'])!=evidence['sha256'] or digest(row['modern_entry']['path'])!=row['modern_entry']['sha256']:raise ValueError('modern pass/entry binding changed')
        passed=json.loads(Path(evidence['path']).read_text())
        if passed.get('strict_success') is not True or not passed.get('graph_completed') or passed['graph_stages']!=row['modern_pass']['stage_count']:raise ValueError('modern algorithm pass absent')
    return {'task':task,'mode':mode,'task_assets_verified':True,'physics_nominal_scene_pass':True,'historical_algorithm_evidence':row['historical_algorithm_evidence'],'historical_graph_nodes':len(graph['stages']),'declared_model_input_dim':row['model_input_dim'],'modern_executor_registered':modern,'modern_same_pipeline_pass':row['SAM_Gemini_whole_task_pass'],'missing':row['missing_for_modern_execute'],'hardware_qualified':False,'device_connections':0,'api_calls':0,'episodes_started':0,'execution_requires_new_scope_and_result_free_freeze':True}


def command(task,mode,config,scope,freeze,output):
    report=check_task(task,mode);_,rows=registry();row=rows[task]
    if not report['modern_executor_registered']:raise ValueError('task-specific executor/judge missing for '+task+': '+'; '.join(report['missing']))
    # Only this exact task currently owns the validated direct-white adapter.
    if task!='put_mug_on_box':raise ValueError('no adapter for requested task; cup fallback prohibited')
    if any(path is None for path in (config,scope,freeze,output)):raise ValueError('new explicit config/scope/result-free freeze/fresh output required')
    if output.exists():raise ValueError('fresh task output required')
    authority=json.loads(scope.read_text());seal=json.loads(freeze.read_text());cfg=json.loads(config.read_text());bound={str(Path(r['path']).resolve()):r['sha256'] for r in seal['bindings']}
    if authority.get('task')!=task or seal.get('status')!='result_free_frozen' or seal.get('scope_sha256')!=digest(scope):raise ValueError('exact task/scope freeze required')
    expected={str(Path(__file__).resolve()):digest(__file__),str(INVENTORY.resolve()):digest(INVENTORY),str(config.resolve()):digest(config),str(scope.resolve()):digest(scope)}
    if not expected.items()<=bound.items() or any(digest(path)!=sha for path,sha in bound.items()):raise ValueError('registry/config/asset source binding mismatch')
    job={'task':task,'mode':mode,'output':str(output.resolve())}
    actual_job=seal.get('job');jobs=seal.get('jobs')
    if actual_job!=job and jobs!=[job]:raise ValueError('exact task/backend/output job binding required')
    if mode=='sim':
        graph=json.loads(Path(cfg['direct_route_graph']).read_text())
        if graph.get('task') not in (task,'twin_'+task):raise ValueError('runtime graph belongs to another task')
        return [sys.executable,row['modern_entry']['path'],'sim','--config',str(config),'--scope',str(scope),'--execution-freeze',str(freeze),'--output',str(output)]
    package=Path(row['modern_real_package']);receipt=json.loads((package/'PROMOTION_RECEIPT.json').read_text())
    if receipt['task']!=task:raise ValueError('portable runtime belongs to another task')
    return [sys.executable,str(package/'source/tools/deployment/promoted_gemini_real_entry.py'),'shadow' if mode=='shadow' else 'execute','--package',str(package),'--config',str(config),'--scope',str(scope),'--output',str(output)]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('list','check','run'),nargs='?',default='list');p.add_argument('--task');p.add_argument('--mode',choices=('sim','shadow','real'),default='sim')
    for name in ('config','scope','execution-freeze','output'):p.add_argument('--'+name,type=Path)
    a=p.parse_args(argv)
    if a.action=='list':
        _,rows=registry();result=[{'task':name,'historical_graph_nodes':row['historical_stage_count'],'modern_executor_registered':bool(row['modern_executor']),'SAM_Gemini_whole_task_pass':row['SAM_Gemini_whole_task_pass']} for name,row in rows.items()];print(json.dumps(result,indent=2));return result
    if not a.task:raise ValueError('explicit --task required; no cup default')
    if a.action=='check':result=check_task(a.task,a.mode);print(json.dumps(result,indent=2));return result
    selected=command(a.task,a.mode,a.config,a.scope,a.execution_freeze,a.output)
    return subprocess.run(selected,check=True).returncode

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError) as error:raise SystemExit('Task registry refused: '+str(error))
