#!/usr/bin/env python3
"""Exactly one additive repair campaign: 10 runs + 12 sealed public-packet replays."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from tools.task_graph.twin_sensor_rsi import digest,validate_execution
from tools.task_graph.twin_sensor_rsi_repair1 import REPAIR_SOURCES
from tools.task_graph.twin_sensor_validate import check_bindings
from prove.task_graph.twin_sensor_judge_repair1 import RepairConfig


def post_bindings(bindings):
    """Keep a completed cohort reviewable even if an input vanished/changed."""
    rows=[]
    for binding in bindings:
        try:
            actual=digest(binding['path']);error=None
        except OSError as exc:
            actual=None;error=f'{type(exc).__name__}: {exc}'
        rows.append({'path':binding['path'],'expected':binding['sha256'],'actual':actual,
                     'matches':actual==binding['sha256'],'error':error})
    return rows


def prepare(scope_path,config_path,contract_path,output_root):
    scope=json.loads(scope_path.read_text())
    initial_path=REPO/'evaluations/twin_sensor_v03/EXECUTION_FREEZE.json'
    initial=json.loads(initial_path.read_text())
    check_bindings(initial['bindings'])
    if digest(initial_path)!=scope['initial_execution_freeze']['sha256']:
        raise ValueError('initial execution freeze differs from repair scope')
    paths={Path(row['path']) for row in initial['bindings']}
    paths.update((initial_path,scope_path.resolve(),config_path.resolve(),contract_path.resolve()))
    paths.update(REPO/path for path in REPAIR_SOURCES)
    paths.add(REPO/'evaluations/twin_sensor_v03/INITIAL_REPORT.json')
    jobs=[]
    for expected in scope['jobs']:
        candidates=[job for job in initial['jobs'] if job['kind']=='development' and
                    all(job[key]==expected[key] for key in ('task','trial','judge_mode','condition_sha256'))]
        if len(candidates)!=1 or expected['task']=='sorting_cube_bimanual' or expected['judge_mode']!='sensor_only':
            raise ValueError('repair job does not match unique initial sensor condition')
        job=dict(candidates[0]);job['output']=str(output_root/'development'/f"{job['task']}_{job['trial']:02d}")
        jobs.append(job)
    shadow_path=REPO/'runs/twin_sensor_v03/initial_validation/shadow_BATCH_REPORT.json'
    dev_path=REPO/'runs/twin_sensor_v03/initial_validation/development_BATCH_REPORT.json'
    paths.update((shadow_path,dev_path))
    dev=json.loads(dev_path.read_text());paths.update(Path(row['report']['path']) for row in dev['runs'] if row.get('report'))
    shadow=json.loads(shadow_path.read_text())
    for row in shadow['runs']:
        seal=row['prediction_seal'];sealed=json.loads(Path(seal['path']).read_text())
        paths.update(Path(binding['path']) for binding in (seal,sealed['packet_archive'],sealed['source_report']))
        jobs.append({'kind':'packet_replay','task':row['task'],'trial':row['trial'],'judge_mode':'sensor_only',
            'initial_prediction_seal':seal,'output':str(output_root/'packet_replay'/f"{row['task']}_{row['trial']:02d}")})
    if len(scope['jobs'])!=10 or len(shadow['runs'])!=12:raise ValueError('repair denominator differs from scope')
    return {'schema':'graph-self-adapter.sensor-execution-freeze.v1','execution_variant':'sensor_only_repair1',
        'result_free':True,'scope':str(scope_path.resolve()),'scope_sha256':digest(scope_path),
        'config':str(config_path.resolve()),'contract':str(contract_path.resolve()),'output_root':str(output_root.resolve()),
        'bindings':[{'path':str(path.resolve()),'sha256':digest(path)} for path in sorted(paths)],'jobs':jobs,
        'development_jobs':10,'existing_public_packet_replays':12,'new_GT_controls':0,'new_sorting_runs':0}


def run_one(job,campaign,campaign_path):
    output=Path(job['output']);row={key:job[key] for key in ('kind','task','trial','judge_mode','output')};row['error']=None
    try:
        if output.exists():raise FileExistsError(output)
        if job['kind']=='packet_replay':
            from tools.task_graph.twin_sensor_packet_replay_repair1 import replay_job
            config=RepairConfig(**json.loads(Path(campaign['config']).read_text())['thresholds'])
            row.update(replay_job(job,config=config,output=output,
                provenance={'execution_freeze_sha256':digest(campaign_path),'actual_bindings':campaign['bindings']}))
        else:
            command=[sys.executable,str(REPO/'tools/task_graph/twin_sensor_rsi_repair1.py'),'--task',job['task'],
                '--trial',str(job['trial']),'--judge-mode','sensor_only','--scope',campaign['scope'],
                '--config',campaign['config'],'--contract',campaign['contract'],'--execution-freeze',str(campaign_path),
                '--output',str(output),'--execute']
            completed=subprocess.run(command,capture_output=True,text=True)
            output.mkdir(parents=True,exist_ok=True);(output/'WORKER.log').write_text(completed.stdout+'\n'+completed.stderr)
            if completed.returncode:raise RuntimeError('worker exit '+str(completed.returncode))
            sources=list(output.glob('*_fixed.json'))
            if len(sources)!=1:raise ValueError('expected one full-task report')
            report=json.loads(sources[0].read_text())
            if (report['task']!=job['task'] or report['condition']['trial']!=job['trial']
                or report['controller']!='fixed_z' or report['execution_variant']!='sensor_only_repair1'
                or report['bindings']['condition_sha256']!=job['condition_sha256']):
                raise ValueError('repair worker identity/condition mismatch')
            row.update(strict_success=bool(report['strict_success']),graph_completed=bool(report['graph_completed']),
                graph_failure=report['graph_failure'],report={'path':str(sources[0]),'sha256':digest(sources[0])})
    except Exception as exc:
        row.update(error=f'{type(exc).__name__}: {exc}',strict_success=False)
    return row


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope',type=Path,default=REPO/'evaluations/twin_sensor_v03/REPAIR1_SCOPE.json')
    p.add_argument('--config',type=Path,required=True);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output-root',type=Path)
    p.add_argument('--kind',choices=('development','packet_replay'),default='packet_replay')
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv)
    if args.prepare:
        if args.execute or args.campaign.exists() or args.output_root is None or args.output_root.exists():
            raise ValueError('fresh result-free prepare required')
        campaign=prepare(args.scope,args.config,args.contract,args.output_root.resolve())
        args.campaign.parent.mkdir(parents=True,exist_ok=True);args.campaign.write_text(json.dumps(campaign,indent=2,sort_keys=True)+'\n')
    else:campaign=json.loads(args.campaign.read_text())
    if Path(campaign['config']).resolve()!=args.config.resolve() or Path(campaign['contract']).resolve()!=args.contract.resolve():
        raise ValueError('actual config/contract differs from frozen campaign')
    pre=check_bindings(campaign['bindings']);jobs=[job for job in campaign['jobs'] if job['kind']==args.kind]
    if len(jobs)!=(10 if args.kind=='development' else 12) or len({job['output'] for job in jobs})!=len(jobs):
        raise ValueError('repair denominator/output uniqueness mismatch')
    if not args.execute:
        print(json.dumps({'dry_run':True,'kind':args.kind,'jobs':len(jobs),'bindings':len(pre),'campaign_sha256':digest(args.campaign)},indent=2));return
    validate_execution(Path(campaign['scope']),args.campaign)
    destination=Path(campaign['output_root'])/(args.kind+'_BATCH_REPORT.json')
    if destination.exists():raise FileExistsError(destination)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(lambda job:run_one(job,campaign,args.campaign),jobs))
    post=post_bindings(campaign['bindings'])
    report={'schema':'graph-self-adapter.sensor-repair-batch.v1','execution_variant':'sensor_only_repair1',
        'kind':args.kind,'denominator':len(jobs),'runs':rows,'errors':sum(row['error'] is not None for row in rows),
        'pre_bindings':pre,'post_bindings':post,'all_bindings_match':all(row['matches'] for row in post),
        'execution_freeze_sha256':digest(args.campaign)}
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'report':str(destination),'denominator':len(rows),'errors':report['errors']}))


if __name__=='__main__':main()
