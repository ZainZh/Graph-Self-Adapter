"""Synthetic schema/seal checks only; no visual inference, training or physics."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from tools.task_graph import p53_dataset as data


def put(path,value):path.write_text(json.dumps(value))
def rows(path,values):path.write_text(''.join(json.dumps(v)+'\n' for v in values))


def episode(root,binding,entry,success=True,historical=False):
    root.mkdir();(root/'public').mkdir();(root/'private').mkdir()
    put(root/'BINDING.json',binding);put(root/'ENTRY.json',entry)
    put(root/'CODE_RESULT.json',dict(qualified=not historical,next_stage='s9' if not historical else None))
    labels=[];transitions=[]
    if not historical:
        for t in range(4):
            public=root/'public'/f'frame_{t:04d}.npz';private=root/'private'/f'frame_{t:04d}.npz'
            np.savez(public,marker=t);np.savez(private,supervisor_sim_state=np.zeros(2),supervisor_ctrl=np.zeros(1))
            labels.append(dict(t=t,stage_id='s9',teacher_phase='local_align' if t==0 else 'insert',authority='private_training_micro_teacher_gt_v03',expert_action=[0]*6+[1],
                public_observation=public.name,public_observation_sha256=data.sha(public),private_state=private.name,private_state_sha256=data.sha(private),
                official_success=t>0 and success,finger_contacts=[True,True],footprint_qualified=t>0,footprint_slack=[-.001,.001] if t==0 else [.001,.001],bottom_gap=.001,
                stop_reason='qualified_goal_contact_hold' if t==3 and success else 'micro_budget' if t==3 else None))
        for t in range(3):transitions.append(dict(t=t,stage_id='s9',episode_id=entry['job_id'],executed_action=[0]*6+[1],
            observation=labels[t]['public_observation'],next_observation=labels[t+1]['public_observation'],
            observation_sha256=labels[t]['public_observation_sha256'],next_observation_sha256=labels[t+1]['public_observation_sha256']))
        rows(root/'public/TRANSITIONS.jsonl',transitions);rows(root/'private/LABELS.jsonl',labels)
    seal=dict(binding_sha256=data.sha(root/'BINDING.json'),files={str(p.relative_to(root)):data.sha(p) for p in root.rglob('*') if p.is_file()})
    put(root/'SEAL.json',seal)
    result=dict(job_id=entry['job_id'],entry=entry,actions=len(transitions),frames=len(labels),near_entry_qualified=not historical,
                teacher_qualified=success and not historical,stop_reason='qualified_goal_contact_hold' if success and not historical else 'code_budget',seal_sha256=data.sha(root/'SEAL.json'))
    put(root/'RESULT.json',result)


def fixture(tmp_path,val_success=True):
    conditions=[dict(index=0,job_id='new_train',split='train'),dict(index=1,job_id='new_val',split='validation')]
    binding=dict(conditions=conditions,relation=dict(book_collision_halfsize=[.01,.03,.04]),
        entry_contract=dict(bottom_gap_min=0,bottom_gap_max=.005,footprint_slack_min=[-.0025,-.0025],require_bilateral_contact=True,require_official_false=True),
        allowed_phases=['local_align','insert'],teacher_authority='private_training_micro_teacher_gt_v03',
                 task_graph=dict(nodes=[dict(id='s9',skill='Insert',backend='private_training_micro_teacher',max_actions=24)]))
    bind=tmp_path/'collection.json';put(bind,binding)
    manifest=dict(stage_scope='near_contact_insert',collection_binding=dict(path=str(bind),sha256=data.sha(bind)),episodes=[],historical_qualifications=[])
    for i,c in enumerate(conditions):
        root=tmp_path/c['job_id'];episode(root,binding,c,success=True if i==0 else val_success)
        manifest['episodes'].append(dict(episode_id=c['job_id'],split=c['split'],directory=str(root)))
    for i in range(8):
        root=tmp_path/f'qualification{i}';entry=dict(job_id=f'qualification{i}',index=i)
        episode(root,dict(conditions=[entry]),entry,success=i>=4,historical=i<4)
        manifest['historical_qualifications'].append(dict(episode_id=entry['job_id'],directory=str(root)))
    path=tmp_path/'manifest.json';put(path,manifest);return path


def test_catalog_retains_failures_and_does_not_fix_cohort_size(tmp_path):
    report=data.catalog(fixture(tmp_path,val_success=False))
    assert report['cohort_size']==2 and len(report['historical_qualifications'])==8
    assert report['eligible_counts']==dict(train=1,validation=0)
    assert len(report['original_collection'])==2
    assert not any(r['eligible_bc'] for r in report['historical_qualifications'])


def test_pack_has_only_public_features_and_explicit_offline_labels(tmp_path):
    path=fixture(tmp_path);report=data.catalog(path);cache=tmp_path/'features';cache.mkdir()
    for row in report['original_collection']:
        features=dict(raw=torch.zeros(4,2,3,3072),boxes=torch.zeros(4,2,3,4),visible=torch.ones(4,2,3),proprio=torch.zeros(4,18))
        torch.save(dict(schema='p52_public_features_v01',features=features,sources=row['frames'],source_seal_sha256=row['source_seal_sha256'],
            frontend={k:'software-fixture' for k in ('dino_sha256','lila_sha256','lila_source_sha256','frontend_source_sha256')}),cache/f'{row["episode_id"]}.pt')
    bundle=data.pack(path,cache)
    assert bundle['stage_scope']=='near_contact_insert' and len(bundle['episodes'])==2
    assert all(len(e['expert_action'])==3 and set(e['features'])=={'raw','boxes','visible','proprio'} for e in bundle['episodes'])


def test_actual_source_corruption_rejected(tmp_path):
    path=fixture(tmp_path)
    np.savez(tmp_path/'new_train/public/frame_0000.npz',marker=999)
    with pytest.raises(ValueError,match='Sealed file changed'):data.catalog(path)


def test_qualification_cohort_without_split_cannot_be_training(tmp_path):
    path=fixture(tmp_path);manifest=data.read(path);binding=data.read(manifest['collection_binding']['path'])
    for c in binding['conditions']:c.pop('split')
    put(Path(manifest['collection_binding']['path']),binding);manifest['collection_binding']['sha256']=data.sha(manifest['collection_binding']['path']);put(path,manifest)
    with pytest.raises(ValueError,match='explicit unique trajectory-level'):data.catalog(path)


def test_entry_contract_controls_gap_and_slack_without_full_fit():
    first=dict(bottom_gap=.006,footprint_slack=[-.002,.001],finger_contacts=[True,True],official_success=False,footprint_qualified=False)
    contract=dict(bottom_gap_min=0,bottom_gap_max=.00717,footprint_slack_min=[-.003585,-.003585],require_bilateral_contact=True,require_official_false=True)
    data.validate_entry_measurement(first,contract)
    with pytest.raises(ValueError,match='frozen near-entry contract'):
        data.validate_entry_measurement(dict(first,bottom_gap=.008),contract)
    with pytest.raises(ValueError,match='frozen near-entry contract'):
        data.validate_entry_measurement(dict(first,footprint_slack=[-.004,.001]),contract)


def test_micro_evaluator_accepts_bound_partial_alignment_entry(tmp_path):
    from tools.task_graph.p53_micro_evaluate import verify_entry
    report=data.catalog(fixture(tmp_path));row=report['original_collection'][1]
    verify_entry(dict(name='validation',split='validation',near_entry_qualified=True,
        source_result=row['source_result'],source_result_sha256=row['source_result_sha256'],
        source_seal=row['source_seal'],source_seal_sha256=row['source_seal_sha256'],
        frame=row['entry_frame'],frame_sha256=row['entry_frame_sha256']))
