"""Assemble P29 root acceptance from completed immutable audits and media.

No simulator, training, candidate choice or statistical recomputation is started.
Raw record hashes were verified by the full task audits; index them with those
receipts rather than claiming another independent replay.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

R = Path('evaluations/fast_adapter_p29_object_transfer_v01')
S = R / 'summary'
TASKS = ('mug-handle-lift', 'hammer-block')
STRATA = ('canonical', 'train', 'geometry_ood')


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, d):
    assert not p.exists(), p
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


def binding(p):
    return {'path': str(p), 'sha256': digest(p)}


def main():
    for name in ('ROOT_ACCEPTANCE.json', 'FINAL_RECORD_INVENTORY.json', 'EVIDENCE_INDEX.json'):
        assert not (S / name).exists(), name
    result = read(S / 'RESULTS.json')
    assert result['counts'] == {'queries': 3520, 'support': 640, 'conditions': 320}
    freezes = [R / n for n in ('FINAL_REPORT_SOURCE_FREEZE.json', 'ANALYSIS_FREEZE.json',
               'FINAL_TOOL_FREEZE.json', 'LEARNING_CLOSE_FREEZE.json', 'BLOCK_EXECUTION_PREPARATION.json')]
    freezes += [R / 'tasks' / t / 'FINAL_FREEZE.json' for t in TASKS]
    freezes += [R / 'tasks/hammer-block/FINAL_GRID_FREEZE.json']
    verified = {}

    def check(d):
        if isinstance(d, dict):
            if isinstance(d.get('path'), str) and isinstance(d.get('sha256'), str):
                p = d['path']
                if p not in verified:
                    verified[p] = digest(p)
                assert verified[p] == d['sha256'], p
            for v in d.values():
                check(v)
        elif isinstance(d, list):
            for v in d:
                check(v)

    for p in freezes:
        check(read(p))
    check(result['sources'])
    audits = []
    records = []
    task_acceptance = {}
    total_states = 0
    record_paths = set()
    evidence = list(freezes) + [S / 'RESULTS.json', S / 'ROOT_REPORT_ZH.md',
        R / 'TRAIN_LABEL_DIAGNOSTICS.json', R / 'PRESENTATION_CORRECTION.json',
        R / 'RECURRENT_META_RECOMPUTATION.json', R / 'RECURRENT_ABORT_PRESERVATION.json',
        R / 'VISUAL_SOURCE_LOCAL.json', R / 'VISUAL_SOURCE_SERVER.json',
        R / 'logs/FINAL_CAPTURE_TRANSFER_COMPLETE.json', R / 'logs/FINISH_CAPTURE_DONE.json',
        R / 'logs/RESOURCE_RELEASE_RECEIPT.json',
        R / 'tasks/hammer-block/data/meta_relabel_r3/INTEGRITY.json']

    def add_record(p, role, raw_sha=None):
        p = Path(p)
        assert p.exists() and str(p) not in record_paths, p
        record_paths.add(str(p))
        records.append({'path': str(p), 'role': role, 'bytes': p.stat().st_size,
                        'sha256': raw_sha or digest(p),
                        'verification': 'full_task_audit' if raw_sha else 'root_closeout_hash'})

    for task in TASKS:
        rnd = result['tasks'][task]['round']
        ap = S / 'final' / rnd / task / 'AUDIT.json'
        pp = S / f'INITIAL_PAIRING_AUDIT_{task}.json'
        sp = S / f'SUPPORT_INPUT_AUDIT_{task}.json'
        audit, pairing, support = read(ap), read(pp), read(sp)
        assert audit['passed'] and audit['counts']['queries'] == 1760
        assert audit['counts']['support_episodes'] == 320
        assert audit['every_query_success_state_replayed']
        assert max(audit['max_errors'].values()) < audit['tolerance']
        assert pairing['physical_proprio_initial_equal'] and pairing['stored_initial_rgb_equal']
        assert pairing['initial_visual_features_exactly_equal']
        assert support['passed'] and support['training_labels_absent']
        if task == TASKS[0]:
            assert support['absent_second_slot_zero']
        total_states += audit['counts']['physical_states']
        audits.append(binding(ap)); evidence.extend([ap, pp, sp])
        task_acceptance[task] = {
            'selected_round': rnd, 'implementation_and_record_integrity_passed': True,
            'ood_three_primary_gates_pass': result['tasks'][task]['ood_three_primary_gates_pass'],
            'promoted': False,
            'conclusion': 'No established current-object-specific adaptation advantage; in-range regression present.'}
        for st in STRATA:
            folder = R / 'tasks' / task / 'final' / rnd / st
            rp = folder / 'RESULT.json'; d = read(rp); evidence.append(rp)
            for row in d['rows']:
                p = Path(row['path']); add_record(p, 'query', row['sha256'])
                add_record(p.with_suffix('.json'), 'query_evaluator')
            for off in range(0, d['conditions'], 8):
                qp = folder / f'QUERY_START_SEAL_{off:03d}.json'; q = read(qp)
                add_record(qp, 'query_start_seal')
                for b in q['support_bindings']:
                    p = Path(b['path']); add_record(p, 'support', b['sha256'])
                    add_record(p.with_suffix('.json'), 'support_evaluator')
                b = q['latents']; add_record(b['path'], 'frozen_query_latents', b['sha256'])
    assert total_states == 704000
    by_role = {k: sum(x['role'] == k for x in records) for k in {x['role'] for x in records}}
    assert by_role['query'] == 440 and by_role['support'] == 80 and by_role['frozen_query_latents'] == 40

    media = []
    for rnd in ('r1', 'r3'):
        oldp = S / 'final' / rnd / 'media/VIDEO_INDEX.json'
        newp = S / 'final' / rnd / 'media_v2/VIDEO_INDEX.json'
        old, new = read(oldp), read(newp)
        assert new['supersedes'] == str(oldp)
        assert len(old['videos']) == len(new['videos']) == 2
        for a, b in zip(old['videos'], new['videos']):
            for k in ('task', 'stratum', 'seed', 'condition', 'kind', 'sources', 'outcomes', 'frames'):
                assert a[k] == b[k], k
            assert digest(b['file']) == b['sha256']
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
                '-show_entries', 'stream=codec_name,width,height,nb_frames',
                '-show_entries', 'format=duration', '-of', 'json', b['file']], text=True))
            v = probe['streams'][0]
            assert v['codec_name'] == 'h264' and (v['width'], v['height']) == (1344, 600)
            assert int(v['nb_frames']) == b['frames'] == 362
            assert abs(float(probe['format']['duration']) - 362 / 12) < .01
            media.append({**binding(b['file']), 'probe': probe,
                          'same_source_episode_and_outcome_as_v1': True})
        evidence.extend([oldp, newp])

    assert len(media) == 4
    assert not any(v['ood_three_primary_gates_pass'] for v in task_acceptance.values())
    inventory = {'time': time.time(), 'scope': 'All selected final query/support records and evaluation sidecars returned locally',
                 'counts': result['counts'], 'files_by_role': by_role, 'records': records,
                 'raw_hash_validation_receipts': audits,
                 'training_archive_note': 'Early canonical demonstration raw visual caches retained on ubuntu@100.110.12.50:/home/ubuntu/github/prove-p22-geometry; not included in this final-only inventory.'}
    write(S / 'FINAL_RECORD_INVENTORY.json', inventory)
    evidence.append(S / 'FINAL_RECORD_INVENTORY.json')
    code = [Path(n) for n in ('tools/fast_adapter/p29_closeout.py', 'tools/fast_adapter/p29_pairing_audit.py',
            'tools/fast_adapter/p29_support_audit.py', 'tools/fast_adapter/p29_media_v2.py', 'tools/fast_adapter/p29_index.py')]
    write(S / 'EVIDENCE_INDEX.json', {'time': time.time(), 'sources': [binding(p) for p in evidence],
          'presentation_and_posthoc_audit_code': [binding(p) for p in code], 'videos': media})
    decision = '两项任务完成独立测试；物体反馈梯度适配尚未证明超过固定经验，并损害范围内能力。本轮不晋升、不扩类，保留救回与误伤证据。'
    write(S / 'ROOT_ACCEPTANCE.json', {
        'time': time.time(), 'reviewer': 'root GPT-6 Astra',
        'status': 'completed_valid_mixed_no_object_specific_adaptation_evidence',
        'decision_summary_zh': decision, 'scientific_promotion': False,
        'counts': result['counts'], 'physical_query_states_recomputed': total_states,
        'tasks': task_acceptance, 'freeze_receipts_verified': len(freezes),
        'unique_frozen_or_result_source_bindings_verified': len(verified),
        'paired_initial_inputs_equal': True, 'absent_cup_target_mask_verified': True,
        'query_has_no_adaptation_updates': True, 'full_dino_reencoding_claimed': False,
        'fixed_model_seeds': [11, 22], 'primary_comparisons': 6, 'primary_gates_passed': 0,
        'training_and_final_rollouts_complete': True, 'no_final_retuning': True,
        'annotation_revision_changes_outcomes': False, 'annotated_videos': len(media),
        'root_visual_review': 'Success-rate plot and rescue posters for both tasks reviewed; hammer support/query/contact frames reviewed; v2 fixes task-specific support text without changing trajectories.',
        'limitations': ['Procedural fixed-color object families; not arbitrary appearance or cross-category generalization.',
                        'Hammer is horizontal tool-mediated push/strike, not ballistic impact.',
                        'Privileged training teacher; RGB/proprio/actions only at runtime.',
                        '160 support actions, not three physical interactions.',
                        'Two fixed model seeds; bootstrap is conditional on these checkpoints.',
                        'Proprio-feedback variant lacks its own mean/donor controls.'],
        'evidence_index': binding(S / 'EVIDENCE_INDEX.json'),
        'result': binding(S / 'RESULTS.json'), 'report': binding(S / 'ROOT_REPORT_ZH.md'),
        'resources': binding(R / 'logs/RESOURCE_RELEASE_RECEIPT.json'),
        'dashboard_updated': False})
    print('ROOT_ACCEPTANCE', decision)
    print('FINAL_RECORDS', len(records), by_role)


if __name__ == '__main__':
    main()
