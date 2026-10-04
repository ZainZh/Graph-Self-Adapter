#!/usr/bin/env python3
"""P53 seal-verified data catalog, frozen features, and offline bundle only.

No collector, optimizer, simulator, retry, replacement, or implicit split. The
root-approved manifest owns cohort size and split; historical qualification
failures stay in catalog and can never become successful BC episodes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tools.task_graph.p52_pack_dataset import sha
from tools.task_graph.p53_train_short import validate_dataset


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def verify_seal(root):
    root=Path(root).resolve();result=read(root/'RESULT.json');seal=read(root/'SEAL.json')
    if sha(root/'SEAL.json')!=result['seal_sha256'] or sha(root/'BINDING.json')!=seal['binding_sha256']:
        raise ValueError('Result/seal/binding chain mismatch')
    if not all(k in seal['files'] for k in ('BINDING.json','ENTRY.json','CODE_RESULT.json')):
        raise ValueError('Required provenance/routing files are not sealed')
    for relative,digest in seal['files'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or sha(path)!=digest:raise ValueError(f'Sealed file changed: {relative}')
    return result,seal


def validate_entry_measurement(first,contract):
    contacts=first['finger_contacts']
    gap=float(first['bottom_gap']);slack=np.asarray(first['footprint_slack'],float)
    minimum=np.asarray(contract['footprint_slack_min'],float)
    if (not np.isfinite(gap) or slack.shape!=(2,) or minimum.shape!=(2,) or not np.isfinite(slack).all()
        or not contract['bottom_gap_min']-1e-9<=gap<=contract['bottom_gap_max']+1e-9
        or np.any(slack<minimum-1e-9)
        or (contract['require_official_false'] and first['official_success'])
        or (contract['require_bilateral_contact'] and (contacts is None or not all(contacts)))):
        raise ValueError('Initial actual teacher measurement does not meet frozen near-entry contract')


def inspect_episode(item,condition=None,binding_sha=None):
    root=Path(item['directory']).resolve();historical=condition is None
    row=dict(episode_id=item['episode_id'],directory=str(root),cohort='historical_qualification' if historical else 'new_native_collection',
             split=None if historical else condition['split'],eligible_bc=False,frames=[],transitions=0)
    if not (root/'RESULT.json').is_file() or not (root/'SEAL.json').is_file():
        row.update(status='unsealed_or_incomplete',reason='Never accepted for feature extraction or BC')
        if (root/'ABORT.json').is_file():row['abort']=read(root/'ABORT.json')
        return row
    result,seal=verify_seal(root);binding=read(root/'BINDING.json');entry=read(root/'ENTRY.json')
    if result['job_id']!=item['episode_id'] or entry['job_id']!=item['episode_id']:raise ValueError('Episode identity mismatch')
    row.update(status='sealed',near_entry_qualified=bool(result.get('near_entry_qualified')),teacher_qualified=bool(result.get('teacher_qualified')),
               stop_reason=result.get('stop_reason'),source_result=str(root/'RESULT.json'),source_result_sha256=sha(root/'RESULT.json'),
               source_seal=str(root/'SEAL.json'),source_seal_sha256=sha(root/'SEAL.json'),binding_sha256=sha(root/'BINDING.json'))
    if not historical:
        if row['binding_sha256']!=binding_sha or entry!=condition:raise ValueError('Actual entry/split differs from frozen collection condition')
        if item['split']!=condition['split']:raise ValueError('Manifest must preserve condition split')
        if result.get('split',entry.get('split'))!=condition['split']:raise ValueError('Result split mismatch')
    if historical:
        row['reason']='Historical qualification outcome retained, successful or failed; never formal training data'
        return row
    transitions=lines(root/'public/TRANSITIONS.jsonl') if (root/'public/TRANSITIONS.jsonl').exists() else []
    labels=lines(root/'private/LABELS.jsonl') if (root/'private/LABELS.jsonl').exists() else []
    count=len(transitions);row['transitions']=count
    if count!=result['actions'] or len(labels)!=result['frames']:raise ValueError('Actual T/count differs from RESULT')
    if not labels:
        if row['near_entry_qualified'] or count:raise ValueError('Qualified entry missing micro trajectory')
        row['reason']='Code/prefix did not produce qualified near entry; retained as system failure'
        return row
    if not row['near_entry_qualified']:raise ValueError('Micro trajectory despite rejected near entry')
    required=['private/LABELS.jsonl']+(['public/TRANSITIONS.jsonl'] if count else [])
    if not all(k in seal['files'] for k in required):raise ValueError('Unsealed label/transition stream')
    if count>24 or len(labels)!=count+1:raise ValueError('Expected <=24 real micro actions and T+1 observations')
    graph=binding['task_graph'];nodes=[n for n in graph['nodes'] if n['skill']=='Insert' and n['backend']=='private_training_micro_teacher']
    if len(nodes)!=1 or nodes[0]['max_actions']!=24:raise ValueError('Need one explicitly bound short training Insert')
    stage=nodes[0]['id'];code=read(root/'CODE_RESULT.json')
    contract=binding['entry_contract'];allowed_phases=binding['allowed_phases'];authority=binding['teacher_authority']
    if not allowed_phases or not authority:raise ValueError('Frozen teacher phase/authority declaration required')
    if not code['qualified'] or code.get('next_stage')!=stage:raise ValueError('Code did not hand off to declared micro Insert')
    for t,label in enumerate(labels):
        if label['t']!=t or label['stage_id']!=stage or label['teacher_phase'] not in allowed_phases or label.get('authority')!=authority:raise ValueError('Nonmicro or nonconsecutive labels prohibited')
        action=np.asarray(label['expert_action'],float)
        if action.shape!=(7,) or not np.isfinite(action).all() or action[-1]!=1:raise ValueError('Expected finite closed-gripper OSC7 teacher')
        frame=root/'public'/label['public_observation'];private=root/'private'/label['private_state']
        for path,digest in [(frame,label['public_observation_sha256']),(private,label['private_state_sha256'])]:
            if path.parent not in (root/'public',root/'private') or seal['files'].get(str(path.relative_to(root)))!=digest or sha(path)!=digest:
                raise ValueError('Label does not bind actual sealed source frame')
        row['frames'].append(dict(path=str(frame),sha256=sha(frame)))
        if t<count:
            tr=transitions[t]
            if (tr['t'],tr['stage_id'],tr['episode_id'])!=(t,stage,item['episode_id']):raise ValueError('Prefix/Code/nonconsecutive transition prohibited')
            if (tr['observation']!=label['public_observation'] or tr['next_observation']!=labels[t+1]['public_observation']
                or tr['observation_sha256']!=label['public_observation_sha256'] or tr['next_observation_sha256']!=labels[t+1]['public_observation_sha256']):
                raise ValueError('Transition observation alignment mismatch')
            if not np.array_equal(np.asarray(tr['executed_action']),action):raise ValueError('Executed action differs from bound teacher command')
    first=labels[0];validate_entry_measurement(first,contract)
    qualified=labels[-1]['stop_reason']=='qualified_goal_contact_hold'
    if qualified!=row['teacher_qualified']:raise ValueError('Teacher qualification differs from final label')
    row.update(stage_id=stage,eligible_bc=bool(qualified and count>=1),reason='Qualified native micro teacher' if qualified else 'Failed micro trajectory retained; no successful BC label inclusion',
               entry_frame=str(root/'private'/first['private_state']),entry_frame_sha256=first['private_state_sha256'])
    return row


def catalog(manifest_path):
    manifest=read(manifest_path)
    if manifest.get('stage_scope')!='near_contact_insert':raise ValueError('Explicit near-contact scope required')
    parent=manifest['collection_binding'];binding_path=Path(parent['path'])
    if sha(binding_path)!=parent['sha256']:raise ValueError('Frozen collection binding changed')
    binding=read(binding_path);conditions=binding['conditions']
    by_id={c['job_id']:c for c in conditions}
    if len(by_id)!=len(conditions) or any(c.get('split') not in ('train','validation') for c in conditions):
        raise ValueError('New collection must have explicit unique trajectory-level train/validation identities')
    items=manifest['episodes']
    if len(items)!=len(conditions) or {e['episode_id'] for e in items}!=set(by_id):raise ValueError('Retain all frozen collection conditions without replacements')
    historical=manifest.get('historical_qualifications',[])
    if len(historical)<8:raise ValueError('Retain all four v01 and four v02 qualification outcomes, including failures and successes')
    if len({e['directory'] for e in historical})!=len(historical):raise ValueError('Do not duplicate historical qualification records')
    rows=[inspect_episode(e,by_id[e['episode_id']],parent['sha256']) for e in items]
    old=[inspect_episode(e) for e in historical]
    if any(r['status']!='sealed' for r in old):raise ValueError('Historical qualification records must retain their seals')
    return dict(schema='p53.native-data-catalog.v1',stage_scope='near_contact_insert',manifest_sha256=sha(manifest_path),collection_binding=parent,
        cohort_size=len(conditions),split_counts={s:sum(c['split']==s for c in conditions) for s in ('train','validation')},
        original_collection=rows,historical_qualifications=old,eligible_counts={s:sum(r['eligible_bc'] and r['split']==s for r in rows) for s in ('train','validation')},
        failure_rule='Never replace a condition or turn qualification/prefix/Code actions into successful BC labels')


def extract(manifest_path,feature_dir,dino,lila,device,shard_index=0,shards=1):
    from prove.task_graph.p52_visual_features import FrozenStageVision,public_features
    if not 0<=shard_index<shards:raise ValueError('Invalid explicit shard')
    report=catalog(manifest_path);torch.set_num_threads(1);feature_dir.mkdir(parents=True,exist_ok=True)
    vision=FrozenStageVision(str(dino),str(lila),device)
    for i,row in enumerate(report['original_collection']):
        if i%shards!=shard_index or not row['frames']:continue
        ident=row['episode_id']
        if Path(ident).name!=ident or ident in ('.','..'):raise ValueError('Unsafe episode identifier')
        target=feature_dir/f'{ident}.pt'
        if target.exists():raise FileExistsError(target)
        features=[]
        for source in row['frames']:
            if sha(source['path'])!=source['sha256']:raise ValueError('Source changed after catalog verification')
            with np.load(source['path'],allow_pickle=False) as frame:features.append(public_features(frame,vision))
        torch.save(dict(schema='p52_public_features_v01',features={k:torch.as_tensor(np.stack([f[k] for f in features])) for k in features[0]},
                        sources=row['frames'],frontend=vision.binding,source_seal_sha256=row['source_seal_sha256']),target)
        print(json.dumps(dict(episode=ident,frames=len(features),qualified=row['eligible_bc'],output=str(target))),flush=True)


def pack(manifest_path,feature_dir):
    report=catalog(manifest_path);episodes=[];bindings=[];frontend=None
    if any(r['status']!='sealed' for r in report['original_collection']):raise ValueError('Original cohort incomplete/unsealed; catalog retained, packing blocked')
    for row in report['original_collection']:
        if not row['eligible_bc']:continue
        cache_path=feature_dir/f'{row["episode_id"]}.pt';cache=torch.load(cache_path,map_location='cpu',weights_only=True)
        if cache['schema']!='p52_public_features_v01' or cache.get('source_seal_sha256')!=row['source_seal_sha256']:raise ValueError('Feature cache schema/seal mismatch')
        if len(cache['sources'])!=len(row['frames']) or any(a['sha256']!=b['sha256'] for a,b in zip(cache['sources'],row['frames'])):raise ValueError('Feature source is not actual sealed public trajectory')
        expected=dict(raw=(2,3,3072),boxes=(2,3,4),visible=(2,3),proprio=(18,))
        if set(cache['features'])!=set(expected):raise ValueError('Unexpected public feature keys')
        for key,shape in expected.items():
            value=cache['features'][key]
            if value.dtype!=torch.float32 or tuple(value.shape)!=(len(row['frames']),*shape) or not torch.isfinite(value).all():
                raise ValueError(f'Invalid public feature: {key}')
        identity={k:cache['frontend'][k] for k in ('dino_sha256','lila_sha256','lila_source_sha256','frontend_source_sha256')}
        if frontend is None:frontend=identity
        if identity!=frontend:raise ValueError('Mixed pretrained visual frontends in dataset')
        root=Path(row['directory']);transitions=lines(root/'public/TRANSITIONS.jsonl');labels=lines(root/'private/LABELS.jsonl')
        episodes.append(dict(episode_id=row['episode_id'],split=row['split'],features=cache['features'],
            executed_action=torch.tensor([r['executed_action'][:6] for r in transitions],dtype=torch.float32),
            expert_action=torch.tensor([r['expert_action'][:6] for r in labels[:-1]],dtype=torch.float32)))
        bindings.append(dict(episode_id=row['episode_id'],feature_sha256=sha(cache_path),source_seal_sha256=row['source_seal_sha256'],source_result_sha256=row['source_result_sha256']))
    result=dict(schema='p53_native_short_insert_v01',stage_scope='near_contact_insert',episodes=episodes,catalog=report,bindings=bindings,frontend=frontend)
    validate_dataset(result,24);return result


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for command in ('catalog','extract','pack'):
        q=sub.add_parser(command);q.add_argument('--manifest',type=Path,required=True)
        if command in ('catalog','pack'):q.add_argument('--output',type=Path,required=True)
        if command in ('extract','pack'):q.add_argument('--feature-dir',type=Path,required=True)
        if command=='extract':
            q.add_argument('--dino',type=Path,required=True);q.add_argument('--lila',type=Path,required=True);q.add_argument('--device',default='cuda')
            q.add_argument('--shard-index',type=int,default=0);q.add_argument('--shards',type=int,default=1)
    a=p.parse_args()
    if a.command=='catalog':write(a.output,catalog(a.manifest))
    elif a.command=='extract':extract(a.manifest,a.feature_dir,a.dino,a.lila,a.device,a.shard_index,a.shards)
    else:
        if a.output.exists():raise FileExistsError(a.output)
        data=pack(a.manifest,a.feature_dir);a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(data,a.output)
        print(json.dumps(dict(episodes=len(data['episodes']),eligible_counts=data['catalog']['eligible_counts'],output=str(a.output))))


if __name__=='__main__':main()
