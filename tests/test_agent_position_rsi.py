"""Position campaign contracts only; no simulator/world or Agent reply fixtures."""
import copy
import json
from pathlib import Path
import pytest
from tools.task_graph import twin_agent_position_rsi as position

IDENTITY='codex-current-root-20261006-v07'


def setup(tmp_path):
    cfg=json.loads((position.REPO/'evaluations/agent_stage_v06/CONFIG.json').read_text())
    cfg.update(execution_variant=position.VARIANT);cfg['agent']['root_identity']=IDENTITY
    contract=json.loads((position.REPO/'evaluations/agent_stage_v06/CONTRACT.json').read_text())
    contract['execution_variant']=position.VARIANT
    scope={'execution_authorized':True,'tasks':[position.TASK],'trials':list(range(10)),
           'condition_seed':92005,'condition_set':'test_v02','maximum_full_task_episodes':10,
           'position_only':True,'maximum_hardware_motion_episodes':0,
           'condition_binding':{'path':str(position.CONDITIONS.resolve()),'sha256':position.digest(position.CONDITIONS)}}
    paths={}
    for name,value in [('config',cfg),('contract',contract),('scope',scope)]:
        paths[name]=tmp_path/(name+'.json');paths[name].write_text(json.dumps(value))
    return paths


def test_all_ten_are_exact_opened_positions_and_same_frozen_assets(tmp_path):
    p=setup(tmp_path)
    runner,packages,solution,required,conditions=position.prepare(p['config'],p['contract'],p['scope'])
    manifest=json.loads(position.CONDITIONS.read_text())
    assert len(conditions)==10
    for trial,value in conditions.items():
        assert value['condition']==manifest['tasks'][position.TASK][str(trial)]
        assert value['sha256']==manifest['condition_sha256_by_task_trial'][position.TASK][str(trial)]
        assert value['condition']['seed']==92005 and value['condition']['force_shift'] is False
        assert any(any(abs(v)>0 for v in delta) for delta in value['condition']['actual_initial_translation_m'].values())
    assert position.CONDITIONS in required and position.HISTORY in required
    assert solution/'graph/task_graph.json' in required and solution/'models/local_skill.npz' in required
    assert Path(manifest['condition_generator']['path']) in required
    args=position.build_args(runner,packages,solution,trial=8,output=position.RUN_ROOT/'trial_08')
    assert args.condition_set=='test_v02' and args.condition_seed==92005 and args.trial==8
    assert args.selection_freeze==position.SELECTION and args.online_z is False


def campaign(tmp_path):
    p=setup(tmp_path);source=tmp_path/'source.py';source.write_text('actual source')
    conditions={i:{'sha256':str(i)*64} for i in range(10)}
    freeze={'schema':position.FREEZE_SCHEMA,'execution_variant':position.VARIANT,'status':'result_free_frozen',
            'scope_sha256':position.digest(p['scope']),'bindings':[{'path':str(source),'sha256':position.digest(source)}],
            'jobs':[{'task':position.TASK,'trial':i,'seed':92005,'condition_set':'test_v02','fixed_z':True,
                     'condition_sha256':conditions[i]['sha256'],'actual_root_reply_identity':IDENTITY,
                     'output':str(position.RUN_ROOT/f'trial_{i:02d}')} for i in range(10)]}
    path=tmp_path/'freeze.json';path.write_text(json.dumps(freeze))
    kwargs={'required_paths':{source},'conditions':conditions,'trial':3,'output':position.RUN_ROOT/'trial_03','identity':IDENTITY}
    return p,path,freeze,kwargs


def test_ten_exact_jobs_and_result_free_guard(tmp_path):
    p,path,freeze,kwargs=campaign(tmp_path)
    assert position.validate_campaign(p['scope'],path,**kwargs)
    bad=copy.deepcopy(freeze);bad['jobs'].pop();path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='ten exact'):position.validate_campaign(p['scope'],path,**kwargs)


@pytest.mark.parametrize('case',['task','seed','hash','identity','duplicate_trial','output','fixed','opened','binding'])
def test_wrong_campaign_bindings_fail_closed(tmp_path,case):
    p,path,freeze,kwargs=campaign(tmp_path);bad=copy.deepcopy(freeze)
    if case=='task':bad['jobs'][2]['task']='put_knife_board_new'
    if case=='seed':bad['jobs'][2]['seed']=51005
    if case=='hash':bad['jobs'][2]['condition_sha256']='bad'
    if case=='identity':bad['jobs'][2]['actual_root_reply_identity']='fake'
    if case=='duplicate_trial':bad['jobs'][2]['trial']=1
    if case=='output':bad['jobs'][2]['output']=str(position.RUN_ROOT/'trial_03')
    if case=='fixed':bad['jobs'][2]['fixed_z']=False
    if case=='opened':bad['status']='results_available'
    if case=='binding':bad['bindings'][0]['sha256']='changed'
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):position.validate_campaign(p['scope'],path,**kwargs)


@pytest.mark.parametrize('case',['trials','position_only','hardware','manifest','scope_task'])
def test_scope_cannot_expand_to_nominal_other_tasks_or_hardware(tmp_path,case):
    p,path,freeze,kwargs=campaign(tmp_path);scope=json.loads(p['scope'].read_text())
    if case=='trials':scope['trials']=[0]
    if case=='position_only':scope['position_only']=False
    if case=='hardware':scope['maximum_hardware_motion_episodes']=1
    if case=='manifest':scope['condition_binding']['sha256']='changed'
    if case=='scope_task':scope['tasks'].append('put_knife_board_new')
    p['scope'].write_text(json.dumps(scope));freeze['scope_sha256']=position.digest(p['scope']);path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError,match='scope'):position.validate_campaign(p['scope'],path,**kwargs)


def test_dry_run_never_installs_physics_or_creates_trial(tmp_path,monkeypatch):
    p=setup(tmp_path);output=tmp_path/'not_created'
    monkeypatch.setattr(position.core,'install_runtime',lambda *a,**kw:pytest.fail('physics install in dry-run'))
    result=position.main(['--trial','0','--root-identity',IDENTITY,'--config',str(p['config']),
                         '--contract',str(p['contract']),'--scope',str(p['scope']),'--output',str(output)])
    assert result['dry_run'] and result['fixed_z'] and result['seed']==92005 and not output.exists()
    with pytest.raises(ValueError,match='freeze required'):
        position.main(['--trial','0','--root-identity',IDENTITY,'--config',str(p['config']),
                       '--contract',str(p['contract']),'--scope',str(p['scope']),'--output',str(output),'--execute'])


def test_original_sensor_and_agent_limits_cannot_change(tmp_path):
    p=setup(tmp_path);position.validate_config(p['config'],p['contract'],IDENTITY)
    cfg=json.loads(p['config'].read_text());cfg['agent']['max_requests_per_episode']=17;p['config'].write_text(json.dumps(cfg))
    with pytest.raises(ValueError):position.validate_config(p['config'],p['contract'],IDENTITY)


def test_changed_historical_source_refused_before_physics(monkeypatch,tmp_path):
    p=setup(tmp_path);actual=position.digest
    monkeypatch.setattr(position,'digest',lambda path:'changed' if Path(path)==position.CONDITIONS else actual(path))
    with pytest.raises(ValueError):position.prepare(p['config'],p['contract'],p['scope'])
