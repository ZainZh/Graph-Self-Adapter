#!/usr/bin/env python3
"""Build one local P52 teacher source/data bundle; no remote actions/binding reuse."""
import argparse
import json
from pathlib import Path
import tarfile
from p52_deployment import ROOT, source_closure, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = set(source_closure(ROOT, ['tools/task_graph/collect_libero_p52.py', 'tools/task_graph/p52_deployment.py']))
    base = Path('data/sources/LIBERO-PRO')
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT / base / 'libero').rglob('*.py'))
    task = 'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl'
    files.add(str(base / 'libero/libero/bddl_files/libero_10' / task))
    files.add('data/sources/libero_one_shot_p48/book_demo_0.hdf5')
    files.add('evaluations/fast_adapter_p49_libero_graph_v01/SUPERVISOR_CALIBRATION.json')
    for capture, frames in [('evaluations/fast_adapter_p48_task_graph_v01/gt_replay_v01', (191, 200, 234)),
                            ('evaluations/fast_adapter_p51_libero_complete_v01/baseline_v02', (333, 342))]:
        files.update(f'{capture}/{name}' for name in ('BINDING.json', 'FRAMES.jsonl', 'SUPERVISOR_NAMES.json'))
        files.update(f'{capture}/frame_{frame:04d}.npz' for frame in frames)
    manifest = {p: record(ROOT / p) for p in sorted(files)}
    # Source capture bindings are provenance only: new remote prepare is required.
    if args.output.exists():
        raise FileExistsError(args.output)
    with tarfile.open(args.output, 'x:gz') as archive:
        for p in sorted(files):
            archive.add(ROOT / p, arcname=p, recursive=False)
    manifest_path = args.output.with_suffix(args.output.suffix + '.manifest.json')
    with manifest_path.open('x') as handle:
        json.dump({'files': manifest, 'bundle': record(args.output), 'execution_binding': 'remote prepare required'}, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'bundle': str(args.output), 'manifest': str(manifest_path), 'files': len(files)}))


if __name__ == '__main__':
    main()
