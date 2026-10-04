"""Post-seal action, state, success and source audit; no candidate changes."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from prove.fast_adapter.geometry3d_env import GeometrySkill, configuration
from prove.fast_adapter.geometry3d_adapter import GeometryLatent, command
from tools.fast_adapter.p41_run import js, sha
from tools.fast_adapter.p42_batch import ROOT, verify


def audit_task(task):
    frozen, protocol = verify()
    rows, prefixes = [], {}
    maxima = dict(action=0., qpos=0., qvel=0., estimate=0., observed_proprioception=0.)
    started = time.time()
    for split in protocol['splits']:
        for seed in protocol['seeds']:
            for arm in protocol['arms']:
                path = ROOT/'independent_v01'/task/f'{split}_{seed}_{arm}'
                if not (path/'RESULT.json').exists():
                    rows.append(dict(path=str(path), status='execution_exception_not_replayable'))
                    continue
                pre = json.loads((path/'PRE_RUN.json').read_text())
                seal = json.loads((path/'SEAL.json').read_text())
                result = json.loads((path/'RESULT.json').read_text())
                assert pre['cfg'] == configuration(task, seed, split)
                assert result['force_adaptation'] is False
                for key, file in [('trace','TRACE.npy'),('observations','OBS.npy'),('terminal','TERMINAL.npy')]:
                    assert sha(path/file) == seal[key]
                for source, digest in pre['sources'].items():
                    assert digest == frozen['sources'][source]
                    assert sha(path/'source'/source) == digest
                contract=json.loads((path/'P42_EPISODE_CONTRACT.json').read_text())
                assert contract['freeze_sha256'] == sha(ROOT/'FREEZE.json')
                trace=np.load(path/'TRACE.npy',allow_pickle=True)
                obs=np.load(path/'OBS.npy',allow_pickle=True)
                terminal=np.load(path/'TERMINAL.npy',allow_pickle=True).item()
                assert len(trace)==len(obs)==190
                key=(split,seed)
                prefix=np.array([x['qpos'] for x in trace[:13]])
                if key in prefixes:
                    assert np.max(np.abs(prefix-prefixes[key])) < 1e-12
                else:
                    prefixes[key]=prefix
                env=GeometrySkill(pre['cfg'],render=False)
                latent=GeometryLatent(task,arm)
                anchor=None
                try:
                    for t,x in enumerate(trace):
                        if task!='cup' or t<105:
                            latent.update(obs[t],t)
                        est=latent.estimate()
                        if est is None:
                            assert x['estimate'] is None
                        else:
                            error=max(float(np.max(np.abs(a-b))) for a,b in zip(est,x['estimate']))
                            maxima['estimate']=max(maxima['estimate'],error)
                            assert error<1e-10
                        if task=='cup' and t==105 and est is not None:
                            p,R,size=est
                            anchor=(p+R[:,0]*(size[0]-.003)-R[:,2]*.004,R.copy())
                        p,R,closed,stage=command(task,t,est,env.start_pos,env.start_R,
                                                result['tool_calibration'].get('length',0.),anchor)
                        error=max(float(np.max(np.abs(p-x['command_p']))),float(np.max(np.abs(R-x['command_R']))))
                        maxima['action']=max(maxima['action'],error)
                        assert error<1e-10 and closed==x['closed'] and stage==x['stage'],(path,t,'action')
                        for field in ('qpos','qvel'):
                            error=float(np.max(np.abs(getattr(env.data,field)-x[field])))
                            maxima[field]=max(maxima[field],error)
                            assert error<1e-8,(path,t,field,error)
                        ep,eR=env.proprio()
                        error=max(float(np.max(np.abs(ep-obs[t]['position']))),float(np.max(np.abs(eR-obs[t]['rotation']))))
                        maxima['observed_proprioception']=max(maxima['observed_proprioception'],error)
                        assert error<1e-8,(path,t,'proprioception',error)
                        env.step(x['command_p'],x['command_R'],x['closed'])
                    for field in ('qpos','qvel'):
                        error=float(np.max(np.abs(getattr(env.data,field)-terminal[field])))
                        maxima[field]=max(maxima[field],error)
                        assert error<1e-8,(path,'terminal',field,error)
                    terminal_facts=env.facts()
                    for field,value in terminal_facts.items():
                        if field=='success':
                            continue
                        assert abs(float(value)-float(result[field]))<1e-10,(path,field)
                    good,penetration=[],[]
                    for state in list(trace[-9:])+[terminal]:
                        env.data.qpos[:]=state['qpos'];env.data.qvel[:]=state['qvel']
                        env.mj.mj_forward(env.model,env.data)
                        good.append(env.facts()['success'])
                        penetration.append(max([max(0.,-float(c.dist)) for c in env.data.contact]+[0.]))
                    assert bool(all(good) and max(penetration)<.002)==result['success']
                    assert sum(good)==result['last10_successes']
                    assert abs(max(penetration)-result['max_last10_penetration'])<1e-10
                    valid=[i for i,o in enumerate(obs) if o['corners'] is not None]
                    assert len(valid)==result['valid_frames']
                    rows.append(dict(path=str(path),status='verified',success=result['success'],
                                     first_valid_frame=valid[0] if valid else None,
                                     valid_frames=len(valid),
                                     waiting_actions=sum(x['stage']=='目标不可见，保持' for x in trace)))
                finally:
                    env.close()
                if len(rows)%25==0:
                    js(ROOT/'summary'/f'{task}_AUDIT_PROGRESS.json',dict(done=len(rows),time=time.time()))
    output=dict(task=task,episodes=len(rows),fully_replayed=sum(r['status']=='verified' for r in rows),
                actions=190*sum(r['status']=='verified' for r in rows),max_errors=maxima,
                all_source_and_trace_hashes=True,common_prefix_equal=True,
                outcomes_recomputed=True,started=started,ended=time.time(),rows=rows)
    js(ROOT/'summary'/f'{task}_AUDIT.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='rows'}),flush=True)


def combine():
    parts=[json.loads((ROOT/'summary'/f'{task}_AUDIT.json').read_text()) for task in ('cup','insert')]
    js(ROOT/'summary/AUDIT.json',dict(episodes=sum(p['episodes'] for p in parts),
        fully_replayed=sum(p['fully_replayed'] for p in parts),actions=sum(p['actions'] for p in parts),
        max_errors={k:max(p['max_errors'][k] for p in parts) for k in parts[0]['max_errors']},
        all_source_and_trace_hashes=all(p['all_source_and_trace_hashes'] for p in parts),
        common_prefix_equal=all(p['common_prefix_equal'] for p in parts),
        outcomes_recomputed=all(p['outcomes_recomputed'] for p in parts),
        parts={p['task']:f"{p['task']}_AUDIT.json" for p in parts}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['cup','insert','combine'],required=True)
    task=p.parse_args().task
    combine() if task=='combine' else audit_task(task)
