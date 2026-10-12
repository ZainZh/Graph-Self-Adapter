"""Relocated package and explicit authorization tests; no hardware or simulator."""
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import pytest
from prove.xtrainer.portable import load_portable_bundle,verify_package,rebind_config,SafetyError
from tools.deployment.package_xtrainer import package
from tools.task_graph.xtrainer_portable import main,validate_execute,prepare_freeze


@pytest.fixture
def packed(tmp_path):
    return Path(package(tmp_path/'portable_cup')['package'])


def test_relocated_bundle_does_not_need_task_plan_or_mujoco(packed,tmp_path,monkeypatch):
    relocated=tmp_path/'laptop'/'cup';relocated.parent.mkdir();shutil.move(str(packed),str(relocated))
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('implicit device connection'))
    bundle=load_portable_bundle(relocated)
    assert len(bundle['graph']['stages'])==28 and bundle['skill'].model.input_dim==29 and not bundle['skill'].online_z
    assert not list(relocated.rglob('scene.xml')) and not list(relocated.rglob('ROOT_TASK_PLAN.json'))
    assert bundle['initial_q'].shape==(2,6) and all(str(relocated) in p for p in bundle['bindings'])


@pytest.mark.parametrize('relative',['models/local_skill.npz','motor_targets.npz','source/frozen_twin_local_skill.py','prove/xtrainer/agent.py'])
def test_tampered_actual_assets_or_implementation_refused(packed,relative):
    with (packed/relative).open('ab') as stream:stream.write(b'changed')
    with pytest.raises(SafetyError,match='hash mismatch'):verify_package(packed)


def test_default_check_and_missing_external_paths_never_connect(packed,monkeypatch,tmp_path):
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('implicit device connection'))
    report=main(['--package',str(packed),'--robot-project',str(tmp_path/'absent_project')])
    assert report['hardware_connections']==0 and report['hardware_ready'] is False and report['missing_external_files']
    assert report['hardware_safety_reviewed'] is False and report['initial_gripper_state_confirmed'] is False


def test_external_rebind_preserves_safety_confirmations():
    cfg={'hardware_safety_reviewed':False,'initial_gripper_state_confirmed':False,'vision':{}}
    result=rebind_config(cfg,robot_project='/tmp/project',sam_root='/tmp/SAM',sam_checkpoint='/tmp/weights.pt')
    assert result['sdk_file'].startswith('/tmp/project/') and result['vision']['sam_root']=='/tmp/SAM'
    assert result['hardware_safety_reviewed'] is False and result['initial_gripper_state_confirmed'] is False
    assert 'sdk_file' not in cfg


def test_zero_scope_refused_before_connections(packed,tmp_path,monkeypatch):
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('device connection before authorization'))
    scope=tmp_path/'scope.json';scope.write_text(json.dumps({'hardware_motion_commands_authorized':False}))
    with pytest.raises(SafetyError,match='authorize'):
        main(['--package',str(packed),'--execute','--scope',str(scope),'--execution-freeze',str(tmp_path/'missing'),
              '--output',str(tmp_path/'not_started')])


def test_preparing_bindings_does_not_authorize_or_freeze(packed,tmp_path):
    config=packed/'CONFIG_PROPOSED.json';scope=tmp_path/'scope.json';scope.write_text(json.dumps({'read_only_connections_authorized':False}))
    report={'package_bindings':{str(packed/'PACKAGE_MANIFEST.json'):'hash'},'external_bindings':{},'missing_external_files':[]}
    value=prepare_freeze(report,config,scope,tmp_path/'future','shadow','root')
    assert value['authorization_not_created'] and value['status']=='result_free_prepared'
    assert value['scope_declared_permissions']['read_only_connections_authorized'] is False


def test_packaged_entry_runs_in_fresh_process_after_relocation(packed,tmp_path):
    relocated=tmp_path/'relocated';shutil.move(str(packed),str(relocated))
    result=subprocess.run([sys.executable,str(relocated/'tools/task_graph/xtrainer_portable.py'),'--check',
                           '--robot-project',str(tmp_path/'absent')],cwd=tmp_path,capture_output=True,text=True,check=True)
    report=json.loads(result.stdout)
    assert report['graph_stages']==28 and report['hardware_connections']==0
