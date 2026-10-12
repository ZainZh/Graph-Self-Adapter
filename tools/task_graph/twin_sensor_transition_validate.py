#!/usr/bin/env python3
"""Result-free freeze and retained-denominator driver for exactly ten v04 jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from tools.task_graph.twin_sensor_rsi import digest,validate_execution
from tools.task_graph.twin_sensor_validate import check_bindings
from tools.task_graph.twin_sensor_validate_repair1 import post_bindings
from tools.task_graph.twin_sensor_transition_rsi import SOURCES


def prepare(scope_path,config_path,contract_path,output_root):
    scope=json.loads(scope_path.read_text())
    preceding=scope['preceding_closed_experiment']
    previous_path=Path(preceding['repair_freeze']['path'])
    if digest(previous_path)!=preceding['repair_freeze']['sha256']:raise ValueError('preceding freeze mismatch')
    previous=json.loads(previous_path.read_text());check_bindings(previous['bindings'])
    paths={Path(row['path']) for row in previous['bindings']}
    paths.update(Path(binding['path']) for binding in preceding.values() if isinstance(binding,dict) and 'path' in binding)
    paths.update((scope_path.resolve(),config_path.resolve(),contract_path.resolve()))
    paths.update(REPO/path for path in SOURCES)
    jobs=[]
    for expected in scope['jobs']:
        candidates=[j for j in previous['jobs'] if j['kind']=='development' and
                    all(j[k]==expected[k] for k in ('task','trial','condition_sha256'))]
        if len(candidates)!=1 or expected['candidate']!='A2':raise ValueError('unique unchanged A2 condition required')
        job=dict(candidates[0]);job['output']=str(output_root/'development'/f"{job['task']}_{job['trial']:02d}")
        jobs.append(job)
    if len(jobs)!=10 or len({(j['task'],j['trial']) for j in jobs})!=10:raise ValueError('v04 scope denominator mismatch')
    return {'schema':'graph-self-adapter.sensor-execution-freeze.v1','execution_variant':'sensor_transition_v04',
        'result_free':True,'scope':str(scope_path.resolve()),'scope_sha256':digest(scope_path),
        'config':str(config_path.resolve()),'contract':str(contract_path.resolve()),'output_root':str(output_root.resolve()),
        'bindings':[{'path':str(path.resolve()),'sha256':digest(path)} for path in sorted(paths)],
        'jobs':jobs,'development_jobs':10,'new_GT_controls':0,'packet_replays':0}


def run_one(job,campaign,campaign_path):
    output=Path(job['output']);row=dict(job);row['error']=None
    try:
        if output.exists():raise FileExistsError(output)
        command=[sys.executable,str(REPO/'tools/task_graph/twin_sensor_transition_rsi.py'),'--task',job['task'],
            '--trial',str(job['trial']),'--judge-mode','sensor_only','--scope',campaign['scope'],
            '--config',campaign['config'],'--contract',campaign['contract'],'--execution-freeze',str(campaign_path),
            '--output',str(output),'--execute']
        done=subprocess.run(command,capture_output=True,text=True)
        output.mkdir(parents=True,exist_ok=True);(output/'WORKER.log').write_text(done.stdout+'\n'+done.stderr)
        if done.returncode:raise RuntimeError('worker exit '+str(done.returncode))
        sources=list(output.glob('*_fixed.json'))
        if len(sources)!=1:raise ValueError('expected one full-task report')
        report=json.loads(sources[0].read_text())
        if (report['task']!=job['task'] or report['condition']['trial']!=job['trial'] or report['controller']!='fixed_z'
            or report['execution_variant']!='sensor_transition_v04'
            or report['bindings']['condition_sha256']!=job['condition_sha256']):raise ValueError('worker identity mismatch')
        row.update(strict_success=bool(report['strict_success']),graph_completed=bool(report['graph_completed']),
            graph_failure=report['graph_failure'],report={'path':str(sources[0]),'sha256':digest(sources[0])})
    except Exception as exc:row.update(error=f'{type(exc).__name__}: {exc}',strict_success=False,graph_completed=False)
    return row


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope',type=Path,default=REPO/'evaluations/twin_sensor_v04/SCOPE.json')
    p.add_argument('--config',type=Path,required=True);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output-root',type=Path)
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv)
    if args.prepare:
        if args.execute or args.campaign.exists() or args.output_root is None or args.output_root.exists():raise ValueError('fresh result-free prepare required')
        campaign=prepare(args.scope,args.config,args.contract,args.output_root.resolve())
        args.campaign.parent.mkdir(parents=True,exist_ok=True);args.campaign.write_text(json.dumps(campaign,indent=2,sort_keys=True)+'\n')
    else:campaign=json.loads(args.campaign.read_text())
    if (Path(campaign['scope']).resolve()!=args.scope.resolve() or Path(campaign['config']).resolve()!=args.config.resolve()
        or Path(campaign['contract']).resolve()!=args.contract.resolve()):raise ValueError('actual paths differ from frozen campaign')
    if campaign['execution_variant']!='sensor_transition_v04':raise ValueError('wrong execution variant')
    pre=check_bindings(campaign['bindings']);jobs=campaign['jobs']
    if len(jobs)!=10 or len({j['output'] for j in jobs})!=10:raise ValueError('denominator/output mismatch')
    if not args.execute:
        print(json.dumps({'dry_run':True,'jobs':len(jobs),'bindings':len(pre),'campaign_sha256':digest(args.campaign)},indent=2));return
    validate_execution(Path(campaign['scope']),args.campaign)
    destination=Path(campaign['output_root'])/'development_BATCH_REPORT.json'
    if destination.exists():raise FileExistsError(destination)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(lambda job:run_one(job,campaign,args.campaign),jobs))
    post=post_bindings(campaign['bindings'])
    report={'schema':'graph-self-adapter.sensor-transition-batch.v1','execution_variant':'sensor_transition_v04',
        'denominator':10,'runs':rows,'errors':sum(r['error'] is not None for r in rows),'pre_bindings':pre,'post_bindings':post,
        'all_bindings_match':all(r['matches'] for r in post),'execution_freeze_sha256':digest(args.campaign)}
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'report':str(destination),'denominator':10,'errors':report['errors']}))


if __name__=='__main__':main()
