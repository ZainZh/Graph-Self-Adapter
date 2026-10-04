"""Read-only local acceptance of returned P44 evidence; no simulator calls."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / 'evaluations/fast_adapter_p44_robot_geometry_v01'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    decision = read(COHORT / 'ALL_COMPLETED_DECISION_AUDIT.json')
    repair = read(COHORT / 'REPAIR_V05_REPLAY_AUDIT.json')
    physics = read(COHORT / 'FIRST_PAIR_REPLAY_AUDIT.json')
    audits = {r['cell']: r for r in decision['rows'] + repair['records']}
    counts = defaultdict(lambda: dict(episodes=0, successes=0, conditions=set(), models=set()))
    inventory, limits = [], []
    for phase in ('development', 'development_v05'):
        for path in sorted((COHORT / phase).glob('**/RESULT.json')):
            cell = path.parent
            key = str(cell.relative_to(ROOT))
            pre, seal, result = [read(cell / f) for f in ('PRE_RUN.json', 'SEAL.json', 'RESULT.json')]
            row = audits[key]
            assert row['passed'] and row['evaluator_recomputed']
            assert sha(path) == row['result_sha256']
            for field, filename in [('trace', 'TRACE.npy'), ('observations', 'OBS.npy'),
                                    ('terminal', 'TERMINAL.npy'), ('pre_run', 'PRE_RUN.json'),
                                    ('frames', 'FRAMES.npz'), ('detail_frames', 'DETAIL_FRAMES.npz')]:
                if field in seal:
                    assert sha(cell / filename) == seal[field], (key, filename)
            freeze = pre['bindings']['freeze']
            assert sha(ROOT / freeze['path']) == freeze['sha256']
            for name, digest in pre['sources'].items():
                assert sha(ROOT / name) == digest, name
                assert sha(cell / 'source' / name) == digest, name
            assert seal['neural_parameters_before'] == seal['neural_parameters_after']
            trace = np.load(cell / 'TRACE.npy', allow_pickle=True)
            assert len(trace) == pre['steps'] == 190
            maximum = {k: max(float(r['robot_tracking'][k]) for r in trace)
                       for k in ('arm_joint_limit_violation_rad', 'finger_joint_limit_violation_m')}
            retained = not (phase == 'development' and result['task'] == 'cup')
            mechanical_flag = maximum['finger_joint_limit_violation_m'] >= 1e-4
            if retained:
                # Same predeclared mechanical tolerance used in the v05 review.
                assert maximum['arm_joint_limit_violation_rad'] < 1e-6
                if mechanical_flag:
                    assert (phase == 'development' and result['task'] == 'insert'
                            and result['split'] == 'id' and not result['success'])
            limits.append(dict(cell=key, retained_in_full_report=retained,
                               mechanical_warning=mechanical_flag, **maximum))
            k = '|'.join((phase, result['task'], result['split'], result['arm']))
            counts[k]['episodes'] += 1
            counts[k]['successes'] += int(result['success'])
            counts[k]['conditions'].add(result['seed'])
            counts[k]['models'].add(result['model_seed'])
            inventory.append(dict(cell=key, phase=phase, retained_in_full_report=retained,
                                  accepted_for_clean_task_comparison=retained and not mechanical_flag,
                                  result_sha256=sha(path), seal_sha256=sha(cell / 'SEAL.json')))
    assert len(inventory) == 24
    assert sum(x['retained_in_full_report'] for x in inventory) == 20
    assert sum(x['accepted_for_clean_task_comparison'] for x in inventory) == 18
    for c in counts.values():
        c['conditions'] = sorted(c['conditions'])
        c['models'] = sorted(c['models'])
    media = []
    for manifest in ['summary/original_pilot/MEDIA_MANIFEST.json', 'summary/repair_v05/MEDIA_MANIFEST.json']:
        for v in read(COHORT / manifest)['videos']:
            assert sha(ROOT / v['video']) == v['video_sha256']
            assert sha(ROOT / v['cover']) == v['cover_sha256']
            media.append(dict(path=v['video'], sha256=v['video_sha256'], origin='sealed original RGB'))
    reconstruction = read(COHORT / 'summary/insert_rescue_reconstruction/MANIFEST.json')
    assert sha(ROOT / reconstruction['video']) == reconstruction['video_sha256']
    assert reconstruction['physics_steps_executed'] == reconstruction['network_calls'] == 0
    for source in reconstruction['sources']:
        for filename, digest in source['sha256'].items():
            assert sha(ROOT / source['cell'] / filename) == digest
    media.append(dict(path=reconstruction['video'], sha256=reconstruction['video_sha256'],
                      origin='posthoc rescue illustration reconstructed from sealed states'))
    engineering = COHORT / 'engineering/render_v05_cup_id440000'
    for filename, digest in read(engineering / 'SEAL.json').items():
        assert sha(engineering / filename) == digest
    assert read(engineering / 'RESULT.json')['success']
    media.append(dict(path=str((engineering / 'annotated.mp4').relative_to(ROOT)),
                      sha256=sha(engineering / 'annotated.mp4'), origin='Oracle engineering demonstration'))
    exceptions = [r for r in physics if not r['passed']]
    assert len(exceptions) == 1 and '/cup_seed11/' in exceptions[0]['cell']
    assert read(COHORT / 'summary/RESOURCE_CLOSEOUT.json')['p44_jobs_ended']
    bindings = ['PILOT_FREEZE.json', 'REPAIR_FREEZE_V05.json', 'REPAIR_AMENDMENT.json',
                'ALL_COMPLETED_DECISION_AUDIT.json', 'FIRST_PAIR_REPLAY_AUDIT.json',
                'REPAIR_V05_REPLAY_AUDIT.json', 'ROOT_ENGINEERING_GATE_V02.json',
                'engineering/V05_MECHANICAL_REPAIR_REVIEW.json', 'summary/RESOURCE_CLOSEOUT.json']
    acceptance = dict(
        schema='prove.p44-root-acceptance.v1', status='complete_development_mixed',
        utc=datetime.now(timezone.utc).isoformat(),
        implementation_request_completed=True, real_hardware=False,
        full_numerical_audit_passed=False,
        audit_note='All 24 neural decision/latent/evaluator replays pass. One exact-physics exception remains in an excluded original cup cell. All four v05 cup full-physics replays and the original insertion first-ID pair pass.',
        original_completed=20, original_cup_excluded=4, original_cup_cancelled=12,
        new_cup_recheck=4, total_neural_episodes=24, full_report_episodes=20,
        mechanically_unflagged_episodes=18,
        insertion_id_warning='Two already-failed online ID441201 trajectories have transient finger limit overtravel (0.847 and 0.127 mm). Retain both failures in full 0/4 ID statistics; ID is not an entirely mechanically clean method comparison. All OOD trajectories and rescues pass the same trajectory check. No post-hoc denominator filtering.',
        frozen_weights_unchanged=True, fresh_final_test=False,
        cup_v05_full_physics_passed=repair['all_passed'],
        cup_mechanical_backend_qualified=True, cup_neural_improvement_demonstrated=False,
        insert_ood_development_rescues=3, insert_id_improvement_demonstrated=False,
        sample_unit='Insertion: two distinct conditions per split crossed with two model checkpoints; repaired cup: one opened condition per split, one checkpoint.',
        limits='GT masks, known camera calibration, PnP/corner geometry and fixed skill/controller priors remain. No claim of DINO necessity, meta-training necessity, statistical significance, real hardware or general RSI.',
        original_physics_exceptions=exceptions, counts=dict(counts), mechanical_trajectory_limits=limits,
        inventory=inventory, media=media,
        bindings=[dict(path=str((COHORT / x).relative_to(ROOT)), sha256=sha(COHORT / x)) for x in bindings],
        acceptance_generator_sha256=sha(__file__), next_experiment_started=False)
    target = COHORT / 'summary/ROOT_ACCEPTANCE.json'
    target.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=acceptance['status'], counts=dict(counts), media=len(media)), ensure_ascii=False))


if __name__ == '__main__':
    main()
