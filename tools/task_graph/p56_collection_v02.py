#!/usr/bin/env python3
"""P56 v02 collection wrapper for a Code-completed, non-s9 predecessor.

The v01 collector, P55 graph, teacher, model, geometry and frozen conditions
remain unchanged. A policy cell whose paired expert never entered s9 is a
sealed, zero-action cohort result; it does not replay the predecessor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from tools.task_graph import p56_collection as base
from tools.task_graph.collect_libero_demo_gt import dump, sha

REVISION = 'v02'
SKIP = 'skip_due_to_paired_predecessor'


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def _read_reference(ref):
    if sha(ref['path']) != ref['sha256']:
        raise ValueError('V02 qualification decision changed')
    return json.loads(Path(ref['path']).read_text())


def prepare(design, p55_binding, p53_joint_checkpoint, qualification_decision):
    binding = base.prepare(design, p55_binding, p53_joint_checkpoint)
    binding['collection_revision'] = REVISION
    binding['qualification_decision'] = _reference(qualification_decision)
    binding['sources'][str(Path(__file__).resolve())] = sha(__file__)
    verify(binding)
    return binding


def verify(binding):
    if binding.get('collection_revision') != REVISION:
        raise ValueError('P56 v02 revision missing')
    if binding.get('sources', {}).get(str(Path(__file__).resolve())) != sha(__file__):
        raise ValueError('P56 v02 wrapper source changed')
    decision = _read_reference(binding['qualification_decision'])
    if decision.get('schema') != 'prove.p56-qualification-decision.v2':
        raise ValueError('Expected P56 v02 qualification decision')
    if decision.get('status') != 'approved' or decision.get('proceed_collection') is not True:
        raise ValueError('V02 collection lacks an explicitly approved qualification revision')
    base.verify(binding)
    return decision


def _source_expert(directory, binding_path, policy_cell, cells):
    if directory is None:
        raise ValueError('Paired expert directory required for fixed policy cell')
    root = Path(directory).resolve()
    expert_cell = next((cell for cell in cells if cell['condition_id'] == policy_cell['condition_id']
                        and cell['source_policy'] == 'expert'), None)
    if expert_cell is None:
        raise ValueError('Frozen paired expert cell is missing')
    result = json.loads((root/'RESULT.json').read_text())
    seal = json.loads((root/'SEAL.json').read_text())
    if (json.loads((root/'CELL.json').read_text()) != expert_cell or
            result['source_policy'] != 'expert' or result['condition_id'] != policy_cell['condition_id'] or
            result['seal_sha256'] != sha(root/'SEAL.json') or
            seal['binding_sha256'] != sha(binding_path) or
            sha(root/'BINDING.json') != sha(binding_path)):
        raise ValueError('Paired expert result/binding/seal mismatch')
    for relative, digest in seal['files'].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError('Paired expert sealed file changed: '+relative)
    return root, result, expert_cell


def _blocked(binding_path, binding, cell, output, expert_root, expert_result):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dump(output/'BINDING.json', binding)
    dump(output/'CELL.json', cell)
    source = dict(directory=str(expert_root), result_sha256=sha(expert_root/'RESULT.json'),
        seal_sha256=sha(expert_root/'SEAL.json'), source_cell_index=expert_result['cell_index'],
        predecessor_stop_reason=expert_result['stop_reason'],
        predecessor_near_entry_qualified=False)
    dump(output/'SOURCE_EXPERT.json', source)
    started = time.time()
    dump(output/'EXECUTION.json', dict(schema='prove.p56-cell-execution.v1',
        run_started_unix=started, cell_index=cell['index'], model_freeze=None,
        paired_expert=source, physics_executed=False))
    files = {name: sha(output/name) for name in ('BINDING.json', 'CELL.json',
        'SOURCE_EXPERT.json', 'EXECUTION.json')}
    dump(output/'SEAL.json', dict(files=files, binding_sha256=sha(binding_path),
        cell_index=cell['index'], run_started_unix=started, model_freeze=None,
        original_initial_restores=0, paired_near_entry_restores=0,
        later_state_restores=0, physics_executed=False))
    result = dict(schema='prove.p56-collection-result.v1', cell_index=cell['index'],
        condition_id=cell['condition_id'], geometry_id=cell['geometry_id'],
        scale_xyz=cell['scale_xyz'], split=cell['split'], seed=cell['seed'],
        source_policy=cell['source_policy'], near_entry_qualified=False,
        near_entry=None, prefix_actions=expert_result['prefix_actions'],
        code_actions=expert_result['code_actions'], actions=0, frames=0,
        official_success=expert_result['official_success'], graph_complete=False,
        stop_reason=SKIP, failure_stage=expert_result['failure_stage'], updates=0,
        label_valid=False, all_executed_expert_labels_valid=False,
        label_exclusion_reason=SKIP, source_expert=source,
        seal_sha256=sha(output/'SEAL.json'), automatic_retry=False,
        full_task_success_evidence=False, physics_executed=False)
    dump(output/'RESULT.json', result)
    return result


def collect(binding_path, index, output, paired_entry=None, model_freeze=None):
    binding_path = Path(binding_path)
    binding = json.loads(binding_path.read_text())
    verify(binding)
    if not 0 <= index < len(binding['cells']):
        raise ValueError('Cell outside frozen P56 v02 scope')
    cell = binding['cells'][index]
    if cell['source_policy'] != 'policy_fixed':
        return base.run(binding_path, index, output, paired_entry=paired_entry,
            model_freeze=model_freeze)
    expert_root, expert_result, _ = _source_expert(
        paired_entry, binding_path, cell, binding['cells'])
    if expert_result['near_entry_qualified']:
        return base.run(binding_path, index, output, paired_entry=expert_root,
            model_freeze=model_freeze)
    return _blocked(binding_path, binding, cell, output, expert_root, expert_result)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--design', type=Path, required=True)
    prepare_parser.add_argument('--p55-binding', type=Path, required=True)
    prepare_parser.add_argument('--p53-joint-checkpoint', type=Path, required=True)
    prepare_parser.add_argument('--qualification-decision', type=Path, required=True)
    prepare_parser.add_argument('--output', type=Path, required=True)
    collect_parser = sub.add_parser('collect')
    collect_parser.add_argument('--binding', type=Path, required=True)
    collect_parser.add_argument('--index', type=int, required=True)
    collect_parser.add_argument('--output', type=Path, required=True)
    collect_parser.add_argument('--paired-entry', type=Path)
    collect_parser.add_argument('--model-freeze', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.output.exists():
            raise FileExistsError(args.output)
        dump(args.output, prepare(args.design, args.p55_binding,
            args.p53_joint_checkpoint, args.qualification_decision))
    else:
        print(json.dumps(collect(args.binding, args.index, args.output,
            paired_entry=args.paired_entry, model_freeze=args.model_freeze), indent=2))


if __name__ == '__main__':
    main()
