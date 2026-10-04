#!/usr/bin/env python3
"""Finite P56 post-collection pipeline; no retry or checkpoint selection.

Starts only after the predeclared expert/policy train and development cells are
sealed. Collection, training, test-entry generation, and local evaluation each
produce immutable receipts. A nonzero command stops subsequent stages.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

from prove.task_graph.p52_stage_policy import Config
from tools.task_graph import p56_collection_v02 as collector
from tools.task_graph import p56_dataset, p56_evaluate
from tools.task_graph.p52_deployment import source_closure
from tools.task_graph.p53_pipeline import kill_process_tree
from tools.task_graph.p53_train_short import sha
from tools.task_graph.p56_train import training_sources


ROOT = Path(__file__).resolve().parents[2]
TRAIN_TIMEOUT = 14400
PACK_TIMEOUT = 1800
CELL_TIMEOUT = 1800
ARMS = p56_evaluate.ARMS


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read(path):
    return json.loads(Path(path).read_text())


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def pipeline_sources():
    entries = ['tools/task_graph/p56_pipeline.py', 'tools/task_graph/p56_dataset.py',
               'tools/task_graph/p56_train.py', 'tools/task_graph/p56_evaluate.py',
               'tools/task_graph/p56_collection_v02.py', 'tools/task_graph/p53_queue.py']
    return {name: sha(ROOT / name) for name in source_closure(ROOT, entries)}


def input_episode(directory, cell):
    root = Path(directory).resolve()
    if read(root / 'CELL.json') != cell:
        raise ValueError('Source cell identity differs from frozen collection binding')
    result = read(root / 'RESULT.json')
    seal = read(root / 'SEAL.json')
    if result['seal_sha256'] != sha(root / 'SEAL.json'):
        raise ValueError('Source result/seal chain mismatch')
    if seal['binding_sha256'] != sha(root / 'BINDING.json'):
        raise ValueError('Source binding/seal chain mismatch')
    if result['cell_index'] != cell['index'] or result['condition_id'] != cell['condition_id']:
        raise ValueError('Source result identity differs')
    return dict(cell_index=cell['index'], directory=str(root), result=reference(root / 'RESULT.json'),
                seal=reference(root / 'SEAL.json'))


def preflight(binding_path, expert_dir, policy_dir):
    binding_path = Path(binding_path).resolve()
    binding = read(binding_path)
    collector.verify(binding)
    cells = binding['cells']
    train_dev = [cell for cell in cells if cell['split'] in ('train', 'validation')]
    if len(train_dev) != 56 or sum(cell['source_policy'] == 'expert' for cell in train_dev) != 28 or sum(cell['source_policy'] == 'policy_fixed' for cell in train_dev) != 28:
        raise ValueError('Expected exactly 28 expert and 28 fixed-policy train/development cells')
    episodes = []
    for cell in train_dev:
        base = expert_dir if cell['source_policy'] == 'expert' else policy_dir
        episodes.append(input_episode(Path(base) / f"cell_{cell['index']:03d}" / 'episode', cell))
    manifest = dict(schema='p56.collection-manifest.v1', collection_binding=reference(binding_path),
                    permitted_geometry_overlap=['1,1,1'],
                    episodes=[dict(cell_index=item['cell_index'], directory=item['directory']) for item in episodes])
    return binding, manifest, episodes


def freeze_pipeline(binding_path, binding, manifest_path, episodes):
    _, _, original = collector.base.verify(binding)
    old = original['checkpoints']
    if not all(name in old for name in ('bc', 'fixed')):
        raise ValueError('Historical P53 BC/joint checkpoints missing')
    old_weights = dict(p53_bc=old['bc'], p53_joint=old['fixed'])
    for item in old_weights.values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Historical checkpoint changed')
    inputs = {str(Path(binding_path).resolve()): sha(binding_path),
              str(Path(manifest_path).resolve()): sha(manifest_path)}
    for item in episodes:
        for key in ('result', 'seal'):
            inputs[item[key]['path']] = item[key]['sha256']
    return dict(schema='p56.finite-pipeline.v1', collection_binding=reference(binding_path),
                manifest=reference(manifest_path), sources=pipeline_sources(), inputs=inputs,
                old_checkpoints=old_weights, runtime_python=str(Path(sys.executable).absolute()),
                training=dict(seed=11, variants=['teacher_only', 'teacher_plus_policy_feedback'],
                              warmup_updates=1000, branch_updates=2000, batch_size=4,
                              learning_rate=3e-4, horizon=24, model=asdict(Config()),
                              gpu_by_variant={'teacher_only': 0, 'teacher_plus_policy_feedback': 1}),
                test_cells=16, eval_cells=144, queue_batch_size=3,
                timeouts=dict(pack=PACK_TIMEOUT, train=TRAIN_TIMEOUT, cell=CELL_TIMEOUT),
                automatic_retries=0, checkpoint_selection='final only')


def verify_pipeline(plan):
    if plan.get('schema') != 'p56.finite-pipeline.v1' or plan.get('automatic_retries') != 0:
        raise ValueError('Wrong P56 pipeline freeze')
    if plan['training'] != dict(seed=11, variants=['teacher_only', 'teacher_plus_policy_feedback'],
            warmup_updates=1000, branch_updates=2000, batch_size=4, learning_rate=3e-4,
            horizon=24, model=asdict(Config()),
            gpu_by_variant={'teacher_only': 0, 'teacher_plus_policy_feedback': 1}):
        raise ValueError('P56 training budget changed')
    if plan['test_cells'] != 16 or plan['eval_cells'] != 144 or plan['queue_batch_size'] != 3:
        raise ValueError('P56 evaluation denominator changed')
    for path, digest in plan['sources'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Pipeline source changed: ' + path)
    for path, digest in plan['inputs'].items():
        if sha(path) != digest:
            raise ValueError('Pipeline input changed: ' + path)


def process(name, argv, output, plan, timeout, gpu=None, stop=None):
    verify_pipeline(plan)
    log_path = output / f'{name}.log'
    receipt_path = output / f'{name}.receipt.json'
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if gpu is not None:
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), MUJOCO_GL='egl')
    started = time.time()
    receipt = dict(name=name, argv=list(map(str, argv)), gpu=gpu, started_unix=started,
                   timeout_seconds=timeout, automatic_retry=False, returncode=None)
    proc = None
    try:
        with log_path.open('x') as stream:
            proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + timeout
            while True:
                if stop is not None and stop.is_set():
                    kill_process_tree(proc.pid)
                    proc.wait()
                    receipt['execution_status'] = 'peer_failed'
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    kill_process_tree(proc.pid)
                    proc.wait()
                    receipt['execution_status'] = 'engineering_timeout'
                    break
                try:
                    receipt['returncode'] = proc.wait(timeout=min(5, remaining))
                    receipt['execution_status'] = 'finished'
                    break
                except subprocess.TimeoutExpired:
                    continue
    except OSError as error:
        receipt.update(execution_status='launch_error', error=str(error))
    receipt.update(finished_unix=time.time(), log_sha256=sha(log_path))
    write(receipt_path, receipt)
    if receipt['returncode'] != 0:
        if stop is not None:
            stop.set()
        raise RuntimeError('P56 stage failed without retry: ' + name)
    verify_pipeline(plan)
    return receipt


def run_queue_batches(stage, jobs, output, plan, extra_bound):
    """Use P53's explicit queue in at most three-job batches; stop after an error."""
    receipts = []
    for offset in range(0, len(jobs), 3):
        batch = jobs[offset:offset + 3]
        batch_no = offset // 3
        folder = output / stage / f'batch_{batch_no:03d}'
        folder.mkdir(parents=True, exist_ok=False)
        queue = dict(schema='p53.explicit-queue.v1', workers=len(batch), attempts_per_job=1,
                     cwd=str(ROOT), bound_files={**extra_bound,
                     str(ROOT / 'tools/task_graph/p53_queue.py'): sha(ROOT / 'tools/task_graph/p53_queue.py')},
                     jobs=batch)
        queue_path = folder / 'QUEUE.json'
        write(queue_path, queue)
        queue_output = folder / 'runs'
        argv = [plan['runtime_python'], '-m', 'tools.task_graph.p53_queue',
                '--manifest', str(queue_path), '--manifest-sha256', sha(queue_path),
                '--output', str(queue_output)]
        process(f'{stage}_{batch_no:03d}', argv, folder, plan, 3 * CELL_TIMEOUT + 600)
        closed = read(queue_output / 'QUEUE_RESULT.json')
        attempts = closed['attempts']
        if len(attempts) != len(batch):
            raise ValueError('Incomplete finite ' + stage + ' queue batch')
        for attempt in attempts:
            if attempt['returncode'] != 0 or attempt['execution_status'] != 'finished':
                raise RuntimeError('P56 ' + stage + ' engineering error; no further batches')
            episode = queue_output / attempt['job']['id'] / 'episode'
            result_path, seal_path = episode / 'RESULT.json', episode / 'SEAL.json'
            if not result_path.is_file() or not seal_path.is_file():
                raise RuntimeError('P56 ' + stage + ' cell lacks sealed result')
            result = read(result_path)
            if result['seal_sha256'] != sha(seal_path):
                raise ValueError('P56 ' + stage + ' result/seal chain changed')
            receipts.append(dict(job_id=attempt['job']['id'], directory=str(episode),
                                 result=reference(result_path), seal=reference(seal_path)))
    return receipts


def summarize(eval_receipts, binding):
    if len(eval_receipts) != 144:
        raise ValueError('Expected all 144 sealed evaluation cells')
    by_index = {}
    for item in eval_receipts:
        result = read(item['result']['path'])
        index = result['cell']['index']
        if index in by_index:
            raise ValueError('Duplicate evaluation cell')
        by_index[index] = result
    if set(by_index) != set(range(144)):
        raise ValueError('Evaluation denominator missing a cell')
    test = [cell for cell in binding['cells'] if cell['split'] == 'test']
    if len(test) != 16:
        raise ValueError('Wrong test cohort')
    comparisons = [('p53_fixed', 'p53_online'),
                   ('teacher_only_fixed', 'teacher_only_online'),
                   ('mixed_fixed', 'mixed_online'),
                   ('teacher_only_bc', 'mixed_bc'),
                   ('teacher_only_fixed', 'mixed_fixed'),
                   ('teacher_only_online', 'mixed_online')]
    def group(indices):
        arms = {}
        for arm_index, arm in enumerate(ARMS):
            values = [by_index[i * 9 + arm_index] for i in indices]
            entered = [v for v in values if v['micro_executed']]
            arms[arm] = dict(success=sum(bool(v['system_success']) for v in values), n=len(values),
                             entered=len(entered), entered_success=sum(bool(v['micro_success']) for v in entered),
                             nonentry=len(values) - len(entered),
                             official_instant_ever=sum(bool(v.get('official_instant_ever')) for v in values),
                             executed_physical_steps=sum(v['executed_physical_steps'] for v in values),
                             z_updates=sum(v['z_updates'] for v in values))
        paired = {}
        for left, right in comparisons:
            li, ri = ARMS.index(left), ARMS.index(right)
            jointly_entered = [i for i in indices if by_index[i * 9 + li]['micro_executed']
                               and by_index[i * 9 + ri]['micro_executed']]
            rescue = [test[i]['condition_id'] for i in indices
                      if not by_index[i * 9 + li]['system_success'] and by_index[i * 9 + ri]['system_success']]
            harm = [test[i]['condition_id'] for i in indices
                    if by_index[i * 9 + li]['system_success'] and not by_index[i * 9 + ri]['system_success']]
            paired[left + '_to_' + right] = dict(rescue=rescue, harm=harm,
                                                rescue_count=len(rescue), harm_count=len(harm),
                                                unchanged=len(indices) - len(rescue) - len(harm),
                                                jointly_entered=len(jointly_entered),
                                                entered_rescue=[test[i]['condition_id'] for i in jointly_entered
                                                    if not by_index[i * 9 + li]['micro_success'] and by_index[i * 9 + ri]['micro_success']],
                                                entered_harm=[test[i]['condition_id'] for i in jointly_entered
                                                    if by_index[i * 9 + li]['micro_success'] and not by_index[i * 9 + ri]['micro_success']])
        return dict(arms=arms, paired=paired)
    geometries = {key: group([i for i, cell in enumerate(test) if cell['geometry_id'] == key])
                  for key in sorted({cell['geometry_id'] for cell in test})}
    return dict(schema='p56.local-summary.v1', scope='16 paired near-entry conditions × 9 arms; local s9 only',
                all=group(list(range(16))), by_geometry=geometries,
                test_conditions=[cell['condition_id'] for cell in test],
                evaluation_cells=144, automatic_retries=0,
                claim='No success or adaptation benefit inferred by this accounting artifact')


def training_command(python, plan_path, data_path, output_path, variant, common_path, branch):
    if branch not in ('warmup', 'bc', 'joint'):
        raise ValueError('Unknown P56 training phase')
    command = 'warmup' if branch == 'warmup' else 'branch'
    argv = [python, '-m', 'tools.task_graph.p56_train', command,
            '--plan', str(plan_path), '--data', str(data_path), '--output', str(output_path),
            '--variant', variant, '--seed', '11', '--device', 'cuda']
    if command == 'branch':
        argv += ['--common', str(common_path), '--branch', branch]
    return argv


def data_accounting(catalog, episodes):
    by_mode = {}
    for item in episodes:
        result = read(item['result']['path'])
        mode = result['source_policy']
        row = by_mode.setdefault(mode, dict(cells=0, entered=0, short_actions=0,
                                            physical_collection_actions=0))
        row['cells'] += 1
        row['entered'] += bool(result['near_entry_qualified'])
        row['short_actions'] += result['actions']
        # A fixed-policy cell restores its paired expert entry; its copied
        # predecessor counts are provenance, not second physical execution.
        row['physical_collection_actions'] += (result['actions'] if mode == 'policy_fixed'
            else result['prefix_actions'] + result['code_actions'] + result['actions'])
    eligible = [row for row in catalog['cells'] if row['eligible'] and row['split'] == 'train']
    def view(modes):
        selected = [row for row in eligible if row['source_policy'] in modes]
        return dict(training_trajectories=len(selected), training_short_transitions=sum(row['actions'] for row in selected),
                    source_modes=list(modes))
    return dict(all_collection_cells_by_mode=by_mode,
                teacher_only=view(('expert',)),
                teacher_plus_policy_feedback=view(('expert', 'policy_fixed')),
                comparison='Same optimizer updates; mixed view uses additional physical policy trajectories. Qualification physics excluded.')


def run(binding_path, expert_dir, policy_dir, output):
    binding_path = Path(binding_path).resolve()
    expert_dir, policy_dir = Path(expert_dir).resolve(), Path(policy_dir).resolve()
    output = Path(output).resolve()
    binding, manifest, episodes = preflight(binding_path, expert_dir, policy_dir)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'TRAIN_MANIFEST.json', manifest)
    plan = freeze_pipeline(binding_path, binding, output / 'TRAIN_MANIFEST.json', episodes)
    write(output / 'PIPELINE_FREEZE.json', plan)
    verify_pipeline(plan)
    report = p56_dataset.catalog(output / 'TRAIN_MANIFEST.json')
    write(output / 'TRAIN_CATALOG.json', report)
    if not report['eligible_counts']['train']:
        raise ValueError('No valid training trajectories; stop without replacement')
    data = {}
    for variant in ('teacher_only', 'teacher_plus_policy_feedback'):
        target = output / f'{variant}.pt'
        argv = [plan['runtime_python'], '-m', 'tools.task_graph.p56_dataset', '--manifest',
                str(output / 'TRAIN_MANIFEST.json'), '--variant', variant, '--output', str(target)]
        process('pack_' + variant, argv, output, plan, PACK_TIMEOUT)
        data[variant] = reference(target)
    training_plan = dict(schema='p56.training-plan.v1', seeds=[11], horizon=24,
                         warmup_updates=1000, branch_updates=2000, batch_size=4,
                         learning_rate=3e-4, model=asdict(Config()), sources=training_sources(),
                         datasets=data)
    training_plan_path = output / 'TRAINING_PLAN.json'
    write(training_plan_path, training_plan)
    weights = {}
    stop = threading.Event()
    def train_variant(variant, gpu):
        common = output / f'{variant}_common.pt'
        bc = output / f'{variant}_bc.pt'
        joint = output / f'{variant}_joint.pt'
        for branch, target in (('warmup', common), ('bc', bc), ('joint', joint)):
            if stop.is_set():
                raise RuntimeError('Peer training branch failed')
            argv = training_command(plan['runtime_python'], training_plan_path,
                                    data[variant]['path'], target, variant, common, branch)
            process(variant + '_' + branch, argv, output, plan, TRAIN_TIMEOUT, gpu=gpu, stop=stop)
        return dict(bc=reference(bc), joint=reference(joint), common=reference(common))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {variant: pool.submit(train_variant, variant, gpu) for variant, gpu in
                   (('teacher_only', 0), ('teacher_plus_policy_feedback', 1))}
        try:
            for variant, future in futures.items():
                weights[variant] = future.result()
        except Exception:
            stop.set()
            for future in futures.values():
                try: future.result()
                except Exception: pass
            raise
    checkpoints = dict(p53_bc=plan['old_checkpoints']['p53_bc'],
                       p53_joint=plan['old_checkpoints']['p53_joint'],
                       teacher_only_bc=weights['teacher_only']['bc'],
                       teacher_only_joint=weights['teacher_only']['joint'],
                       mixed_bc=weights['teacher_plus_policy_feedback']['bc'],
                       mixed_joint=weights['teacher_plus_policy_feedback']['joint'])
    freeze = dict(schema='prove.p56-model-freeze.v1', created_unix=time.time(),
                  collection_binding_sha256=sha(binding_path), checkpoints=checkpoints,
                  test_entry_generation_authorized_after_freeze=True,
                  training_plan=reference(training_plan_path), automatic_retries=0)
    freeze_path = output / 'MODEL_FREEZE.json'
    write(freeze_path, freeze)
    write(output / 'FINAL_WEIGHTS.json', dict(checkpoints=checkpoints,
          warmups={name: weights[name]['common'] for name in weights},
          training_plan=reference(training_plan_path), model_freeze=reference(freeze_path)))
    test_cells = [cell for cell in binding['cells'] if cell['split'] == 'test']
    jobs = [dict(id=f"cell_{cell['index']:03d}", seed=cell['seed'], timeout_seconds=CELL_TIMEOUT,
                 argv=[plan['runtime_python'], '-m', 'tools.task_graph.p56_collection_v02', 'collect',
                       '--binding', str(binding_path), '--index', str(cell['index']),
                       '--model-freeze', str(freeze_path)]) for cell in test_cells]
    test_receipts = run_queue_batches('test_entries', jobs, output, plan,
                                      {str(binding_path): sha(binding_path), str(freeze_path): sha(freeze_path)})
    test_manifest_path = output / 'TEST_MANIFEST.json'
    write(test_manifest_path, dict(schema='p56.test-entry-manifest.v1',
          collection_binding=reference(binding_path),
          episodes=[dict(cell_index=read(Path(row['directory']) / 'CELL.json')['index'],
                         directory=row['directory']) for row in test_receipts]))
    eval_binding_path = output / 'EVAL_BINDING.json'
    eval_argv = [plan['runtime_python'], '-m', 'tools.task_graph.p56_evaluate', 'prepare',
                 '--manifest', str(test_manifest_path), '--model-freeze', str(freeze_path),
                 '--output', str(eval_binding_path)]
    for name, item in checkpoints.items():
        eval_argv += ['--' + name.replace('_', '-'), item['path']]
    process('eval_prepare', eval_argv, output, plan, PACK_TIMEOUT)
    write(output / 'EVAL_FREEZE_RECEIPT.json', dict(binding=reference(eval_binding_path),
          model_freeze=reference(freeze_path), test_manifest=reference(test_manifest_path)))
    eval_jobs = [dict(id=f'cell_{index:03d}', seed=test_cells[index // 9]['seed'],
                      timeout_seconds=CELL_TIMEOUT,
                      argv=[plan['runtime_python'], '-m', 'tools.task_graph.p56_evaluate', 'run',
                            '--binding', str(eval_binding_path), '--index', str(index)])
                 for index in range(144)]
    eval_receipts = run_queue_batches('evaluation', eval_jobs, output, plan,
                                      {str(eval_binding_path): sha(eval_binding_path),
                                       str(freeze_path): sha(freeze_path)})
    summary = summarize(eval_receipts, binding)
    summary['training_data_accounting'] = data_accounting(report, episodes)
    write(output / 'SUMMARY.json', summary)
    write(output / 'COMPLETE.json', dict(schema='p56.finite-pipeline-complete.v1',
          pipeline_freeze=reference(output / 'PIPELINE_FREEZE.json'),
          training_plan=reference(training_plan_path), model_freeze=reference(freeze_path),
          eval_binding=reference(eval_binding_path), summary=reference(output / 'SUMMARY.json'),
          input_collection_cells=56,
          actual_eligible_trajectories=sum(bool(row['eligible']) for row in report['cells']),
          test_entries=16, evaluation_cells=144,
          automatic_retries=0, full_task_success_evidence=False))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--expert-dir', type=Path, required=True)
    parser.add_argument('--policy-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists():
        raise FileExistsError(output)
    try:
        result = run(args.binding, args.expert_dir, args.policy_dir, output)
        print(json.dumps(dict(summary=result['all'], by_geometry=result['by_geometry'])))
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        receipts = sorted(output.rglob('*.receipt.json'), key=lambda path: path.stat().st_mtime)
        write(output / 'PIPELINE_ABORT.json', dict(schema='p56.finite-pipeline-abort.v1',
              failed_unix=time.time(), error_type=type(error).__name__, reason=str(error),
              traceback=traceback.format_exc(), automatic_retry=False,
              last_completed_receipt=(reference(receipts[-1]) if receipts else None)))
        raise


if __name__ == '__main__':
    main()
