#!/usr/bin/env python3
"""Join sealed public feature caches with offline expert labels by trajectory.

Manifest episodes: episode_id, split, directory, feature_cache. Never random frame
splitting. Teacher qualification smoke is forbidden in a formal training bundle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tools.task_graph.p52_train import validate


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()
def lines(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def verify_trajectory(root, *, require_local=True):
    root=Path(root).resolve()
    seal_path=root/'SEAL.json';result_path=root/'RESULT.json'
    seal=json.loads(seal_path.read_text());result=json.loads(result_path.read_text())
    if result.get('seal_sha256')!=sha(seal_path):raise ValueError('Result does not bind actual trajectory seal')
    if seal.get('binding_sha256')!=sha(root/'BINDING.json'):raise ValueError('Trajectory binding changed')
    for relative,digest in seal['files'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root):raise ValueError('Seal path escapes trajectory')
        if sha(path)!=digest:raise ValueError(f'Sealed trajectory file changed: {relative}')
    required=['public/TRANSITIONS.jsonl','private/LABELS.jsonl']
    if require_local and not all(p in seal['files'] for p in required):raise ValueError('Missing sealed public transitions/offline labels')
    return seal,result


def pack(manifest):
    episodes=[];bindings=[]
    for item in manifest['episodes']:
        root=Path(item['directory']);cache_path=Path(item['feature_cache'])
        if 'teacher_smoke' in root.parts:raise ValueError('Teacher smoke is excluded from pilot training')
        seal,result=verify_trajectory(root)
        transitions=lines(root/'public/TRANSITIONS.jsonl');labels=lines(root/'private/LABELS.jsonl')
        if item['split']=='train' and (not result.get('teacher_qualified') or len(transitions)<16):
            raise ValueError('Training requires qualified teacher and >=16 real transitions')
        cache=torch.load(cache_path,map_location='cpu',weights_only=True)
        if cache['schema']!='p52_public_features_v01':raise ValueError('Wrong feature cache')
        if len(cache['sources'])!=len(transitions)+1:raise ValueError('Require T+1 feature frames')
        by_t={r['t']:r for r in labels}
        source=cache['sources']
        for t,cached in enumerate(source):
            frame=root/'public'/f'frame_{t:04d}.npz'
            if seal['files'].get(str(frame.relative_to(root)))!=cached['sha256'] or sha(frame)!=cached['sha256']:
                raise ValueError('Cached feature source differs from actual sealed public frame')
        if seal['actions']!=len(transitions) or seal['observations']!=len(source):
            raise ValueError('Seal action/observation counts differ from trajectory')
        for t,row in enumerate(transitions):
            if row['t']!=t or row['stage_id']!='Insert':raise ValueError('Nonconsecutive stage sequence')
            if row['episode_id']!=item['episode_id']:raise ValueError('Episode binding mismatch')
            if source[t]['sha256']!=row['observation_sha256'] or source[t+1]['sha256']!=row['next_observation_sha256']:
                raise ValueError('Feature/public-transition hash mismatch')
            label=by_t[t]
            if label['public_observation_sha256']!=row['observation_sha256']:
                raise ValueError('Offline action label/public observation mismatch')
            if row['executed_action'][6]!=1 or label['expert_action'][6]!=1:
                raise ValueError('Insert must retain closed gripper')
        episodes.append(dict(episode_id=item['episode_id'],split=item['split'],features=cache['features'],
            executed_action=torch.tensor([r['executed_action'][:6] for r in transitions],dtype=torch.float32),
            expert_action=torch.tensor([by_t[t]['expert_action'][:6] for t in range(len(transitions))],dtype=torch.float32)))
        bindings.append(dict(episode_id=item['episode_id'],feature_sha256=sha(cache_path),
                            seal_sha256=sha(root/'SEAL.json'),result_sha256=sha(root/'RESULT.json'),
                            transition_sha256=sha(root/'public/TRANSITIONS.jsonl'),
                            offline_label_sha256=sha(root/'private/LABELS.jsonl'),frontend=cache['frontend']))
    validate(episodes)
    return dict(schema='p52_native_training_v01',episodes=episodes,bindings=bindings)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=pack(json.loads(a.manifest.read_text()));result['manifest_sha256']=sha(a.manifest)
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(result,a.output)
    print(json.dumps(dict(episodes=len(result['episodes']),output=str(a.output))))


if __name__=='__main__':main()
