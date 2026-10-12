#!/usr/bin/env python3
"""Bounded result-free sensor campaign preparation and batch validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.task_graph.twin_sensor_rsi import TASK_CANDIDATES, digest, load_frozen_runner, validate_execution
from prove.task_graph.twin_sensor_judge import JudgeConfig

NEW_SOURCES = ('prove/task_graph/twin_sensor_judge.py', 'prove/task_graph/twin_sensor_adapter.py',
               'tools/task_graph/twin_sensor_rsi.py', 'tools/task_graph/twin_sensor_shadow.py',
               'tools/task_graph/twin_sensor_validate.py')


def check_bindings(bindings, *, strict=True):
    result = []
    for row in bindings:
        actual = digest(row['path'])
        result.append({'path': row['path'], 'expected': row['sha256'], 'actual': actual,
                       'matches': actual == row['sha256']})
    if strict and not all(row['matches'] for row in result):
        raise ValueError('sensor campaign binding mismatch')
    return result


def prepare(scope_path, config_path, contract_path, output_root):
    scope = json.loads(scope_path.read_text())
    paths = {scope_path.resolve(), config_path.resolve(), contract_path.resolve()}
    paths.update(REPO / source for source in NEW_SOURCES)
    for relative in ('ACCEPTANCE_CRITERIA.json', 'REAL_INPUT_AUDIT.json'):
        paths.add(REPO / 'evaluations/twin_sensor_v03' / relative)
    paths.add(REPO / 'tools/evidence/render_twin_graph_video.py')
    plan_path = REPO / 'evaluations/twin_rsi_v02/ROOT_TASK_PLAN.json'
    asset_path = REPO / 'evaluations/twin_rsi_v02/EXTERNAL_ASSET_BINDINGS.json'
    runtime_path = REPO / 'evaluations/twin_rsi_v02/RUNTIME_CONTRACT.json'
    paths.update((plan_path, asset_path, runtime_path))
    plan, assets, runtime = [json.loads(path.read_text()) for path in (plan_path, asset_path, runtime_path)]
    paths.update(Path(row['path']) for row in assets['external_evaluator_sources'])
    paths.add(Path(assets['parent_asset_catalog']['path']))
    jobs = []
    for task, candidate in TASK_CANDIDATES.items():
        runner, snapshot, packages = load_frozen_runner(task)
        solution = packages / ('twin_' + task) / 'solution'
        runner.verify_freeze(solution)
        freeze = json.loads((solution / 'CANDIDATE_FREEZE.json').read_text())
        for relative, expected in freeze['code'].items():
            path = snapshot / relative
            if digest(path) != expected:
                raise ValueError('frozen base code mismatch')
            paths.add(path)
        for sub in ('graph', 'models', 'refs'):
            paths.update(path for path in (solution / sub).rglob('*') if path.is_file())
        paths.update(solution / name for name in ('CANDIDATE_FREEZE.json','MANIFEST.json','EXTERNAL_BINDINGS.json'))
        task_plan = runner.task_from_plan(plan, task)
        paths.update(Path(row['path']) for row in task_plan['sources'].values())
        paths.update(Path(path) for path in assets['task_assets'][task]['external_asset_hashes'])
        for trial in range(5):
            condition = runner.condition_manifest(task_plan, seed=51005, trial=trial,
                nominal=trial == 0, condition_design=runtime['condition_design'])
            for mode in ('privileged_gt_control', 'sensor_only'):
                jobs.append({'kind':'development', 'task':task, 'trial':trial, 'judge_mode':mode,
                    'candidate':candidate, 'condition':condition,
                    'condition_sha256':runner.canonical_json_sha256(condition),
                    'output':str(output_root / 'development' / mode / f'{task}_{trial:02d}')})
    for row in scope['offline_shadow']['traces']:
        paths.update(Path(row[key]['path']) for key in ('report','visual_trace'))
        jobs.append({**row, 'kind':'shadow', 'judge_mode':'sensor_only',
            'output':str(output_root / 'shadow' / f"{row['task']}_{row['trial']:02d}")})
    bindings = [{'path':str(path.resolve()), 'sha256':digest(path)} for path in sorted(paths)]
    return {'schema':'graph-self-adapter.sensor-execution-freeze.v1', 'result_free':True,
            'scope_sha256':digest(scope_path), 'scope':str(scope_path.resolve()),
            'config':str(config_path.resolve()), 'contract':str(contract_path.resolve()),
            'output_root':str(output_root.resolve()), 'bindings':bindings, 'jobs':jobs,
            'default':'fixed_z', 'hardware_commands':False, 'development_jobs':30, 'shadow_jobs':12}


def run_one(job, campaign, campaign_path):
    output = Path(job['output'])
    row = {'task':job['task'], 'trial':job['trial'], 'judge_mode':job['judge_mode'],
           'kind':job['kind'], 'output':str(output), 'error':None}
    try:
        if output.exists():
            raise FileExistsError(output)
        if job['kind'] == 'shadow':
            from tools.task_graph.twin_sensor_shadow import shadow_job
            config = JudgeConfig(**json.loads(Path(campaign['config']).read_text())['thresholds'])
            result = shadow_job(job, config=config, output=output,
                provenance={'campaign_sha256':digest(campaign_path), 'bindings':campaign['bindings']})
            row.update(result)
        else:
            command = [sys.executable, str(REPO / 'tools/task_graph/twin_sensor_rsi.py'),
                '--task',job['task'],'--trial',str(job['trial']),'--judge-mode',job['judge_mode'],
                '--scope',campaign['scope'],'--config',campaign['config'],'--contract',campaign['contract'],
                '--execution-freeze',str(campaign_path),'--output',str(output),'--execute']
            completed = subprocess.run(command, capture_output=True, text=True)
            output.mkdir(parents=True, exist_ok=True)
            (output / 'WORKER.log').write_text(completed.stdout + '\n' + completed.stderr)
            if completed.returncode:
                raise RuntimeError('worker failed: exit ' + str(completed.returncode))
            reports = list(output.glob('*_fixed.json'))
            if len(reports) != 1:
                raise ValueError('worker must write exactly one full-task report')
            report = json.loads(reports[0].read_text())
            condition_hash = report['bindings']['condition_sha256']
            if (report['task'] != job['task'] or report['condition']['trial'] != job['trial']
                or report['controller'] != 'fixed_z' or condition_hash != job['condition_sha256']
                or report.get('sensor_backend',{}).get('judge_mode') != job['judge_mode']):
                raise ValueError('worker task/condition/controller/sensor-mode mismatch')
            row.update(strict_success=bool(report['strict_success']), graph_failure=report['graph_failure'],
                       report={'path':str(reports[0]),'sha256':digest(reports[0])})
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        row['strict_success'] = False
    return row


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',type=Path,default=REPO/'evaluations/twin_sensor_v03/SCOPE.json')
    parser.add_argument('--config',type=Path,default=REPO/'evaluations/twin_sensor_v03/SENSOR_CONFIG_PROPOSED.json')
    parser.add_argument('--contract',type=Path,default=REPO/'evaluations/twin_sensor_v03/SENSOR_CONTRACT_PROPOSED.json')
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--kind',choices=('shadow','development'),default='shadow')
    parser.add_argument('--workers',type=int,choices=range(1,9),default=4)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if args.prepare:
        if args.execute or args.campaign.exists() or args.output_root is None or args.output_root.exists():
            raise ValueError('prepare requires fresh campaign and output-root, without execution')
        campaign=prepare(args.scope,args.config,args.contract,args.output_root.resolve())
        args.campaign.parent.mkdir(parents=True,exist_ok=True)
        args.campaign.write_text(json.dumps(campaign,indent=2,sort_keys=True)+'\n')
    else:
        campaign=json.loads(args.campaign.read_text())
    pre=check_bindings(campaign['bindings'])
    jobs=[job for job in campaign['jobs'] if job['kind']==args.kind]
    if len(jobs)!=(12 if args.kind=='shadow' else 30) or len({job['output'] for job in jobs})!=len(jobs):
        raise ValueError('invalid bounded campaign denominator or duplicate output')
    if not args.execute:
        print(json.dumps({'dry_run':True,'kind':args.kind,'jobs':len(jobs),
                          'bindings':len(pre),'campaign_sha256':digest(args.campaign)},indent=2));return
    validate_execution(Path(campaign['scope']),args.campaign)
    summary_path=Path(campaign['output_root'])/(args.kind+'_BATCH_REPORT.json')
    if summary_path.exists():
        raise FileExistsError(summary_path)
    # Shadow rendering stays sequential to avoid sharing EGL/OpenGL contexts.
    if args.kind=='shadow':
        rows=[run_one(job,campaign,args.campaign) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows=list(pool.map(lambda job:run_one(job,campaign,args.campaign),jobs))
    post=check_bindings(campaign['bindings'],strict=False)
    report={'schema':'graph-self-adapter.sensor-validation-batch.v1','kind':args.kind,
            'denominator':len(jobs),'runs':rows,'errors':sum(row['error'] is not None for row in rows),
            'campaign_sha256':digest(args.campaign),'pre_bindings':pre,'post_bindings':post,
            'all_bindings_match':all(row['matches'] for row in post)}
    summary_path.parent.mkdir(parents=True,exist_ok=True)
    summary_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'report':str(summary_path),'denominator':len(rows),'errors':report['errors']}))


if __name__=='__main__':
    main()
