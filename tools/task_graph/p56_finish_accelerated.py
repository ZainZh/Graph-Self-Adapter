#!/usr/bin/env python3
"""Independent P56 finish from final weights; no training or condition changes.

This runner never edits the already running finite pipeline or its freeze. It
binds six final checkpoints, creates the model freeze before any test entry,
then uses the frozen P56 collector/evaluator and original summary accounting.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from tools.task_graph import p56_collection_v02 as collector
from tools.task_graph import p56_evaluate, p56_pipeline
from tools.task_graph.p52_deployment import source_closure
from tools.task_graph.p53_train_short import sha

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'p56.accelerated-finish.v1'
AMENDMENT_SCHEMA = 'prove.p56-accelerated-finish-amendment.v1'
GPUS = (0, 1, 2, 3)
CELL_TIMEOUT = 1800
PREPARE_TIMEOUT = 1800


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def runtime_python():
    """Keep the venv launcher path; resolving its symlink can escape the venv."""
    return str(Path(sys.executable).absolute())


def _check_ref(value):
    if sha(value['path']) != value['sha256']:
        raise ValueError('Accelerated finish input changed: '+value['path'])


def validate_amendment(amendment, binding_sha):
    if (amendment.get('schema') != AMENDMENT_SCHEMA or amendment.get('status') != 'approved'
            or amendment.get('collection_binding_sha256') != binding_sha
            or amendment.get('server_gpus') != list(GPUS)
            or amendment.get('authorization', {}).get('source') != 'current user request'
            or amendment.get('authorization', {}).get('quote') !=
                '训练还可以加快一些吗？ 怎么这么慢 服务器四张卡以及本机的显卡都可以使用。'):
        raise ValueError('Missing approved four-GPU P56 execution amendment')
    training = amendment.get('training_amendments')
    if not isinstance(training, dict) or not training:
        raise ValueError('Training amendments must be explicitly referenced')
    for item in training.values():
        if not isinstance(item, dict) or set(item) != {'path', 'sha256'}:
            raise ValueError('Training amendment needs path/SHA-256')
        _check_ref(item)


def validate_checkpoints(checkpoints, original, amendment):
    if set(checkpoints) != set(p56_evaluate.CHECKPOINTS):
        raise ValueError('Exactly six final P53/P56 checkpoints required')
    weights = {}
    packages = {}
    for name in p56_evaluate.CHECKPOINTS:
        item = checkpoints[name]
        if not isinstance(item, dict) or set(item) != {'path', 'sha256'}:
            raise ValueError('Each final checkpoint needs path and SHA-256')
        frozen = ref(item['path'])
        if frozen['sha256'] != item['sha256']:
            raise ValueError('Final checkpoint bytes differ: '+name)
        weights[name] = frozen
        packages[name] = p56_evaluate.load_skill(
            frozen['path'], p56_evaluate.CHECKPOINT_ARM[name])[1]
    if (weights['p53_bc'] != original['checkpoints']['bc'] or
            weights['p53_joint'] != original['checkpoints']['fixed']):
        raise ValueError('Historical P53 baseline weights changed')
    for stem in ('teacher_only', 'mixed'):
        bc, joint = packages[stem+'_bc'], packages[stem+'_joint']
        for key in ('common_warmup_sha256', 'data_sha256', 'plan_sha256'):
            if bc[key] != joint[key]:
                raise ValueError('BC/joint P56 branches lack shared '+key)
        used = joint.get('execution_amendment')
        allowed = {item['sha256'] for item in amendment['training_amendments'].values()}
        if (not isinstance(used, dict) or
                not isinstance(used.get('sha256'), str) or
                used['sha256'] not in allowed):
            raise ValueError('P56 joint checkpoint lacks the bound training execution amendment')
    if packages['teacher_only_bc']['plan_sha256'] != packages['mixed_bc']['plan_sha256']:
        raise ValueError('P56 views use different frozen training plans')
    for key in ('common_warmup_sha256', 'data_sha256'):
        if packages['p53_bc'][key] != packages['p53_joint'][key]:
            raise ValueError('Historical P53 checkpoint pair differs')
    return weights


def source_hashes():
    roots = ['tools/task_graph/p56_finish_accelerated.py',
             'tools/task_graph/p56_collection_v02.py',
             'tools/task_graph/p56_evaluate.py',
             'tools/task_graph/p56_pipeline.py',
             'tools/task_graph/p52_vision_worker.py']
    return {name: sha(ROOT/name) for name in source_closure(ROOT, roots)}


def freeze_inputs(binding_path, checkpoints_json, amendment_path):
    binding_path = Path(binding_path).resolve()
    checkpoints_json = Path(checkpoints_json).resolve()
    amendment_path = Path(amendment_path).resolve()
    binding = read(binding_path)
    collector.verify(binding)
    _, _, original = collector.base.verify(binding)
    amendment = read(amendment_path)
    validate_amendment(amendment, sha(binding_path))
    weights = validate_checkpoints(read(checkpoints_json), original, amendment)
    if len([c for c in binding['cells'] if c['split'] == 'test']) != 16:
        raise ValueError('P56 test condition denominator changed')
    return binding, dict(schema=SCHEMA, collection_binding=ref(binding_path),
        checkpoints_json=ref(checkpoints_json), execution_amendment=ref(amendment_path),
        checkpoints=weights, sources=source_hashes(), runtime_python=runtime_python(),
        server_gpus=list(GPUS), cell_timeout_seconds=CELL_TIMEOUT,
        test_entries=16, evaluation_cells=144, maximum_parallel_jobs=4,
        evaluation_arms=list(p56_evaluate.ARMS), micro_max_actions=24,
        automatic_retries=0, checkpoint_selection='final only',
        prior_pipeline_freeze_reused=False)


def verify_freeze(freeze):
    if (freeze.get('schema') != SCHEMA or freeze.get('server_gpus') != list(GPUS)
            or freeze.get('cell_timeout_seconds') != CELL_TIMEOUT
            or freeze.get('test_entries') != 16 or freeze.get('evaluation_cells') != 144
            or freeze.get('evaluation_arms') != list(p56_evaluate.ARMS)
            or freeze.get('micro_max_actions') != 24
            or freeze.get('maximum_parallel_jobs') != 4 or freeze.get('automatic_retries') != 0
            or freeze.get('prior_pipeline_freeze_reused') is not False):
        raise ValueError('Accelerated finish scope changed')
    for key in ('collection_binding', 'checkpoints_json', 'execution_amendment'):
        _check_ref(freeze[key])
    validate_amendment(read(freeze['execution_amendment']['path']),
                       freeze['collection_binding']['sha256'])
    for item in freeze['checkpoints'].values():
        _check_ref(item)
    for name, digest in freeze['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Accelerated finish source changed: '+name)


def _process(job, folder, gpu, freeze):
    """One attempt in its own session; timeout kills only this process group."""
    folder.mkdir(parents=True, exist_ok=False)
    episode = folder/'episode'
    log = folder/'process.log'
    command = job['argv'] + ['--output', str(episode)]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
        MUJOCO_GL='egl', LIBERO_CONFIG_PATH=str(episode/'libero_config'),
        PYTHONHASHSEED=str(job['seed']), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1')
    row = dict(job_id=job['id'], command=command,
        command_sha256=hashlib.sha256(json.dumps(command, ensure_ascii=False,
            separators=(',', ':')).encode()).hexdigest(),
        gpu=gpu, cell_index=job['cell_index'],
        timeout_seconds=CELL_TIMEOUT, started_unix=time.time(), returncode=None,
        execution_status='not_started', automatic_retry=False,
        finish_freeze=ref(freeze))
    with log.open('x') as stream:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                row['returncode'] = process.wait(timeout=CELL_TIMEOUT)
                row['execution_status'] = 'finished'
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                row['execution_status'] = 'engineering_timeout'
        except OSError as error:
            row.update(execution_status='launch_error', error=str(error))
    row['finished_unix'] = time.time()
    row['log_sha256'] = sha(log)
    if row['returncode'] == 0:
        result, seal = episode/'RESULT.json', episode/'SEAL.json'
        if not result.is_file() or not seal.is_file() or read(result).get('seal_sha256') != sha(seal):
            row['execution_status'] = 'unsealed_result'
        else:
            row['result'] = ref(result)
            row['seal'] = ref(seal)
            row['directory'] = str(episode)
    write(folder/'RECEIPT.json', row)
    return row


def queue_batches(stage, jobs, output, freeze_path, freeze):
    """Four distinct GPUs per batch; an error prevents later batches."""
    rows = []
    for offset in range(0, len(jobs), 4):
        verify_freeze(freeze)
        batch = jobs[offset:offset+4]
        batch_dir = output/stage/f'batch_{offset//4:03d}'
        batch_dir.mkdir(parents=True, exist_ok=False)
        write(batch_dir/'BATCH.json', dict(stage=stage, jobs=batch,
            finish_freeze=ref(freeze_path), gpus=list(GPUS[:len(batch)]),
            maximum_parallel_jobs=4, automatic_retries=0))
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = [pool.submit(_process, job, batch_dir/job['id'], gpu, freeze_path)
                for job, gpu in zip(batch, GPUS)]
            current = [future.result() for future in futures]
        write(batch_dir/'BATCH_RESULT.json', dict(stage=stage, attempts=current,
            jobs_expected=len(batch), batch_sha256=sha(batch_dir/'BATCH.json'),
            all_finished=all(row['execution_status'] == 'finished' and row['returncode'] == 0
                for row in current)))
        rows.extend(current)
        if any(row['execution_status'] != 'finished' or row['returncode'] != 0 for row in current):
            raise RuntimeError(stage+' engineering error; later batches stopped without retry')
    return rows


def prepare_evaluation(command, output, freeze_path, freeze):
    verify_freeze(freeze)
    log = output/'eval_prepare.log'
    receipt = dict(command=command, timeout_seconds=PREPARE_TIMEOUT,
        started_unix=time.time(), finish_freeze=ref(freeze_path), automatic_retry=False)
    with log.open('x') as stream:
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            try:
                receipt['returncode'] = process.wait(timeout=PREPARE_TIMEOUT)
                receipt['execution_status'] = 'finished'
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                receipt.update(returncode=None, execution_status='engineering_timeout')
        except OSError as error:
            receipt.update(returncode=None, execution_status='launch_error', error=str(error))
    receipt.update(finished_unix=time.time(), log_sha256=sha(log))
    write(output/'EVAL_PREPARE_RECEIPT.json', receipt)
    if receipt['returncode'] != 0:
        raise RuntimeError('Evaluation prepare failed without retry')


def run(collection_binding, checkpoints_json, execution_amendment, output):
    output = Path(output).resolve()
    binding, freeze = freeze_inputs(collection_binding, checkpoints_json, execution_amendment)
    output.mkdir(parents=True, exist_ok=False)
    freeze_path = output/'ACCELERATED_FINISH_FREEZE.json'
    write(freeze_path, freeze)
    verify_freeze(freeze)
    weights = freeze['checkpoints']
    model_freeze = output/'MODEL_FREEZE.json'
    write(model_freeze, dict(schema='prove.p56-model-freeze.v1',
        created_unix=time.time(), collection_binding_sha256=freeze['collection_binding']['sha256'],
        checkpoints=weights, test_entry_generation_authorized_after_freeze=True,
        execution_amendment=freeze['execution_amendment'],
        accelerated_finish_freeze=ref(freeze_path), automatic_retries=0))
    test_cells = [cell for cell in binding['cells'] if cell['split'] == 'test']
    jobs = [dict(id=f"cell_{cell['index']:03d}", cell_index=cell['index'], seed=cell['seed'],
        argv=[freeze['runtime_python'], '-m', 'tools.task_graph.p56_collection_v02', 'collect',
            '--binding', freeze['collection_binding']['path'], '--index', str(cell['index']),
            '--model-freeze', str(model_freeze)]) for cell in test_cells]
    test = queue_batches('test_entries', jobs, output, freeze_path, freeze)
    manifest_path = output/'TEST_MANIFEST.json'
    write(manifest_path, dict(schema='p56.test-entry-manifest.v1',
        collection_binding=freeze['collection_binding'], episodes=[dict(
            cell_index=row['cell_index'], directory=row['directory']) for row in test]))
    eval_binding = output/'EVAL_BINDING.json'
    command = [freeze['runtime_python'], '-m', 'tools.task_graph.p56_evaluate', 'prepare',
        '--manifest', str(manifest_path), '--model-freeze', str(model_freeze),
        '--output', str(eval_binding)]
    for name, item in weights.items():
        command.extend(['--'+name.replace('_', '-'), item['path']])
    prepare_evaluation(command, output, freeze_path, freeze)
    write(output/'EVAL_FREEZE_RECEIPT.json', dict(binding=ref(eval_binding),
        model_freeze=ref(model_freeze), test_manifest=ref(manifest_path)))
    eval_jobs = [dict(id=f'cell_{index:03d}', cell_index=index,
        seed=test_cells[index//9]['seed'], argv=[freeze['runtime_python'], '-m',
            'tools.task_graph.p56_evaluate', 'run', '--binding', str(eval_binding),
            '--index', str(index)]) for index in range(144)]
    evaluation = queue_batches('evaluation', eval_jobs, output, freeze_path, freeze)
    summary = p56_pipeline.summarize(evaluation, binding)
    write(output/'SUMMARY.json', summary)
    write(output/'COMPLETE.json', dict(schema='p56.accelerated-finish-complete.v1',
        finish_freeze=ref(freeze_path), model_freeze=ref(model_freeze),
        eval_binding=ref(eval_binding), summary=ref(output/'SUMMARY.json'),
        test_entries=16, evaluation_cells=144, automatic_retries=0))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection-binding', type=Path, required=True)
    parser.add_argument('--checkpoints-json', type=Path, required=True)
    parser.add_argument('--execution-amendment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.collection_binding, args.checkpoints_json,
            args.execution_amendment, args.output)
        print(json.dumps(dict(all=result['all'], by_geometry=result['by_geometry'])))
    except Exception as error:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        write(output/'FINISH_ABORT.json', dict(schema='p56.accelerated-finish-abort.v1',
            failed_unix=time.time(), error_type=type(error).__name__, reason=str(error),
            traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
