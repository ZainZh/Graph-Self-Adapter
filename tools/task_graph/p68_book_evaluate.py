"""P68 opened book-entry development comparison; no training."""
import os
os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=os.environ.get('MUJOCO_EGL_DEVICE_ID','0'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from tools.task_graph import p64_book_evaluate as old
from tools.task_graph.p64_book_evaluate import (read,dump,sha,ref,verify_entry,verify_review,relocated,shift_caddy,_append,_seal,PUBLIC_KEYS)
ROOT=Path(__file__).resolve().parents[2]
ARMS=('frame_fixed','reference_fixed','entry_once','continuous')
SCHEMA='prove.p68-book-reference-evaluation.v1'


def cells():return [dict(index=i*4+j,condition=i,arm=arm) for i in range(12) for j,arm in enumerate(ARMS)]


def verify(binding):
    if binding.get('schema')!=SCHEMA or binding.get('cells')!=cells() or binding.get('arms')!=list(ARMS):
        raise ValueError('Exact P68 scope required')
    validate_assignments(binding)
    legacy=copy.deepcopy(binding);legacy.update(schema=old.SCHEMA,cells=old.cells(),arms=list(old.ARMS))
    return old.verify(legacy)


def prepare(*,assignment_files,**inputs):
    binding=old.prepare(**inputs)
    binding["assignment_files"]=[ref(path) for path in assignment_files]
    binding.update(schema=SCHEMA,cells=cells(),arms=list(ARMS),
        evidence_scope='48 opened P64 local s9 development cells; not fresh or full task')
    from tools.task_graph.p52_deployment import source_closure
    for name in source_closure(ROOT,['tools/task_graph/p68_book_evaluate.py','prove/fast_adapter/p68_book_runtime.py','tests/test_p68_book.py']):
        binding['sources'][str((ROOT/name).resolve())]=sha(ROOT/name)
    verify(binding)
    return binding


def run(binding_path, index, output):
    binding = read(binding_path)
    collection_binding = verify(binding)
    verify_review(binding_path)
    if type(index) is not int or not 0 <= index < 48:
        raise ValueError('Outside exact P68 48-cell queue')
    cell = binding['cells'][index]
    condition = binding['conditions'][cell['condition']]
    entry = next(x for x in binding['entries'] if x['cell_index'] == condition['source_index'])
    source_cell, near = verify_entry(entry, collection_binding, read(binding['legacy_binding']['path']))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('public', 'private'):
        (output/name).mkdir()
    dump(output/'BINDING.json', binding)
    dump(output/'CELL.json', cell)
    dump(output/'EXECUTION.json', dict(started_unix=time.time(), automatic_retry=False))
    if near is None:
        _seal(output, binding_path, cell, 0)
        result = dict(cell=cell, condition=condition, near_entry_qualified=False,
                      system_success=False, native_hold3=False, executed_physical_steps=0,
                      reason='original_entry_failed', full_task_success_evidence=False,
                      seal_sha256=sha(output/'SEAL.json'))
        dump(output/'RESULT.json', result)
        return result
    import torch
    from tools.task_graph import p56_collection as collection
    from tools.task_graph.p52_full_task import InsertJudge, structural_measurement
    from tools.task_graph.collect_libero_p52_v02 import private_measurement
    from tools.task_graph.p54_full_task import near_gate
    from tools.task_graph.collect_libero_demo_gt import snapshot
    from prove.fast_adapter.p64_common import load_encoder, _state_hash
    from prove.fast_adapter.p68_book_runtime import ReferenceBookRuntime
    from prove.fast_adapter.p64_book_bridge import BookPriors
    from prove.task_graph.p57_geometry_policy import load_geometry_policy
    env = renderer = runtime = None
    steps = 0
    try:
        scene = read(binding['source_scene']['path'])
        mapping = read(binding['relocation']['path'])
        # Generate original scaled XML first, verifying its old scene identity,
        # then replace only declared asset paths for local execution.
        scene['geometry_properties'] = read(binding['geometry_binding']['path'])['geometry']['properties']
        scene['prefix']['calibration'] = relocated(scene['prefix']['calibration'], mapping)
        child, geometry, props = collection.build_child(collection_binding, source_cell, output, scene)
        if hashlib.sha256(child['environment']['relocated_xml'].encode()).hexdigest() != near['scene_xml_sha256']:
            raise ValueError('Scaled source scene differs from sealed entry')
        if sha(output/'SUPERVISOR_CALIBRATION.json') != sha(Path(entry['directory'])/'SUPERVISOR_CALIBRATION.json'):
            raise ValueError('Restored evaluator calibration differs from sealed source')
        child = relocated(child, mapping)
        env, _ = collection.initialize_child(child, output, geometry['axes'], props['halfsize'],
                                             near_entry=Path(entry['directory'])/near['frame'])
        shift_caddy(env, condition['caddy_shift_world_m'])
        names = collection._names(env)
        obs = env._get_observations(force_update=True)
        gate = near_gate(child['relation'], private_measurement(env, obs, names, child['relation']),
                         dict(near_position_tolerance=.003585, near_gap_max=.00717))
        dump(output/'private/ENTRY_GATE.json', gate)
        arrays, _ = snapshot(env, obs)
        np.savez_compressed(output/'private/INITIAL_STATE.npz',
                            **{k:v for k,v in arrays.items() if k.startswith('supervisor_')})
        qualified = bool(gate['near_ready'])
        reason = 'shifted_or_restored_entry_failed'
        strict = native_ever = native_instant = False
        if qualified:
            device = read(binding['dino_binding']['path']).get('device', 'cuda')
            encoder = load_encoder(binding['encoder_checkpoint']['path'], device=device,
                                   expected_sha256=binding['encoder_checkpoint']['sha256'])
            policy, _, _ = load_geometry_policy(binding['action_checkpoint']['path'], binding['a2_head']['path'], device=device)
            priors = BookPriors(**read(binding['priors']['path'])['values'])
            from prove.fast_adapter.visual_alignment import FrozenDinoROI
            dino = read(binding['dino_binding']['path'])
            vision = FrozenDinoROI(dino['model_dir'], device=device, microbatch=2)
            runtime = ReferenceBookRuntime(encoder, policy, priors,
                                  cell['arm'],
                                  vision=vision, device=device)
            before = (_state_hash(encoder), _state_hash(policy))
            renderer = collection.PublicRenderer(env, child['public'])
            judge, hold = InsertJudge(), 0
            while True:
                public = renderer.observe(obs)
                public = {key: public[key] for key in PUBLIC_KEYS}
                frame = output/'public'/f'frame_{steps:04d}.npz'
                np.savez_compressed(frame, **public)
                action, telemetry = runtime.step(public)
                action = np.asarray(action, float)
                if action.shape != (7,) or not np.isfinite(action).all() or action[6] != 1:
                    raise ValueError('Invalid frozen neural OSC command')
                # Prediction committed before privileged structural evaluation.
                _append(output/'public/PREDICTIONS.jsonl', dict(t=steps, observation_sha256=sha(frame),
                                                               action=action, telemetry=telemetry))
                official, fingers, wrong = structural_measurement(env, names)
                judgment = judge.observe(official, fingers, wrong)
                hold = hold+1 if official else 0
                native_ever = native_ever or hold >= 3
                native_instant = native_instant or official
                reason = judgment['stop'] or ('micro_budget24' if steps >= 24 else None)
                _append(output/'private/JUDGMENTS.jsonl', dict(t=steps, official=official, fingers=fingers,
                                                              wrong_target=wrong, **judgment))
                arrays, _ = snapshot(env, obs)
                np.savez_compressed(output/'private'/f'frame_{steps:04d}.npz',
                                    **{k:v for k,v in arrays.items() if k.startswith('supervisor_')})
                _append(output/'public/EXECUTED.jsonl', dict(t=steps, executed=reason is None,
                                                            action=action if reason is None else None))
                if reason:
                    strict = bool(judgment['complete'])
                    break
                obs, _, _, _ = env.step(action.copy())
                steps += 1
            after = (_state_hash(encoder), _state_hash(policy))
            dump(output/'public/WEIGHT_AUDIT.json', dict(
                schema='prove.p64-book-weight-audit.v1',
                encoder_before=before[0], encoder_after=after[0],
                policy_before=before[1], policy_after=after[1], unchanged=before == after))
            if before != after:
                raise RuntimeError('Frozen evaluation weights changed')
        if runtime is not None:
            runtime.exit()
            dump(output/'public/LIFECYCLE.json',runtime.lifecycle)
        _seal(output, binding_path, cell, steps)
        result = dict(cell=cell, condition=condition, near_entry_qualified=qualified,
                      system_success=strict, native_hold3=native_ever, native_instant_ever=native_instant,
                      executed_physical_steps=steps, reason=reason, full_task_success_evidence=False,
                      seal_sha256=sha(output/'SEAL.json'))
        dump(output/'RESULT.json', result)
        return result
    except Exception as exc:
        dump(output/'ABORT.json', dict(cell=cell, reason=str(exc), executed_physical_steps=steps,
                                      automatic_retry=False))
        raise
    finally:
        if runtime is not None and runtime.active: runtime.exit()
        if renderer is not None: renderer.close()
        if env is not None: env.close()


def validate_assignments(binding):
    refs=binding.get('assignment_files',[])
    if len(refs)!=2:raise ValueError('Exactly two bound shards required')
    for gpu,item in enumerate(refs):
        old.check_ref(item)
        if read(item['path'])!=dict(shard=f'shard{gpu}',gpu=gpu,indices=list(range(gpu*24,(gpu+1)*24))):
            raise ValueError('Only complete predeclared shards permitted')


def signal_program_error(root,index,returncode):
    """Shared task stop, first writer owns the receipt; no active child is killed."""
    path=Path(root)/'STOP_PROGRAM_ERROR.json'
    try:
        with path.open('x') as stream:json.dump(dict(index=index,returncode=returncode,automatic_retry=False),stream)
    except FileExistsError:pass


def queue(binding_path,output,indices_file,queue_output):
    binding=read(binding_path);verify(binding);verify_review(binding_path)
    selected=[r for r in binding['assignment_files'] if Path(r['path']).resolve()==Path(indices_file).resolve()]
    if len(selected)!=1:raise ValueError('Unbound indices file')
    old.check_ref(selected[0]);assignment=read(indices_file);indices=assignment['indices']
    output=Path(output);queue_output=Path(queue_output)
    if any((output/f'cell_{index:03d}').exists() for index in indices):raise FileExistsError('Preexisting assigned cell')
    if assignment['gpu']==1:
        first=output/'cell_000'
        if not (first/'RESULT.json').is_file():raise ValueError('First cell must complete before shard1')
        seal=read(first/'SEAL.json')
        if seal['binding_sha256']!=sha(binding_path) or seal['cell']!=cells()[0]:raise ValueError('First cell identity differs')
        for name,digest in seal['files'].items():
            if sha(first/name)!=digest:raise ValueError('First cell seal changed')
    output.mkdir(parents=True,exist_ok=True);queue_output.mkdir(parents=True,exist_ok=False)
    environment=dict(os.environ,MUJOCO_EGL_DEVICE_ID=str(assignment['gpu']),CUDA_VISIBLE_DEVICES=str(assignment['gpu']))
    dump(queue_output/'ASSIGNMENT.json',dict(binding_sha256=sha(binding_path),assignment=assignment,indices_file=ref(indices_file)))
    for ordinal,index in enumerate(indices):
        if (output/'STOP_PROGRAM_ERROR.json').exists():raise RuntimeError('Other shard program error; no new submissions')
        if (output/f'cell_{index:03d}').exists():raise FileExistsError('Concurrent assigned cell')
        try:
            with (queue_output/f'{index:03d}.log').open('x') as log:
                result=subprocess.run([sys.executable,'-m','tools.task_graph.p68_book_evaluate','run',
                    '--binding',str(Path(binding_path).resolve()),'--index',str(index),
                    '--output',str((output/f'cell_{index:03d}').resolve())],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        except Exception:
            signal_program_error(output,index,-1)
            raise
        _append(queue_output/'QUEUE.jsonl',dict(index=index,returncode=result.returncode))
        print(json.dumps(dict(completed=ordinal+1,program_errors=int(result.returncode!=0))),flush=True)
        if result.returncode:
            signal_program_error(output,index,result.returncode)
            raise RuntimeError('P68 queue stopped; no automatic retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--inputs', type=Path, required=True, help='JSON kwargs for prepare')
    prep.add_argument('--output', type=Path, required=True)
    for name in ('run', 'queue'):
        command = commands.add_parser(name)
        command.add_argument('--binding', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        if name == 'run': command.add_argument('--index', type=int, required=True)
        else:
            command.add_argument('--indices-file',type=Path,required=True)
            command.add_argument('--queue-output',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.output.exists(): raise FileExistsError(args.output)
        dump(args.output, prepare(**read(args.inputs)))
    elif args.command == 'run':
        print(json.dumps(run(args.binding, args.index, args.output)))
    else:
        queue(args.binding, args.output,args.indices_file,args.queue_output)


if __name__ == '__main__':
    main()
