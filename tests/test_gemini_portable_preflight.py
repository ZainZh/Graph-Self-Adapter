"""Offline portable receipt and no-motion preflight tests, no real key/device/API."""
from pathlib import Path
import sys,json,importlib.util,socket
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/deployment'))
from gemini_readonly_preflight import preflight
from gemini_promotion_check import check,digest
from prove.xtrainer.io import SafetyLayer,SafetyError,RobotSample
from test_stage_trajectory import Clock
PACKAGE=ROOT/'evaluations/gemini_graph_v19/deployment/cup_gemini_v19_r1'


def fake_io():
    clock=Clock();clock.t=10.;q=np.zeros((2,6));calls=[]
    class Robot:
        def samples(self):return [RobotSample(clock(),row,np.zeros(6),row[:3],np.eye(3),None,'fake_readonly_actual') for row in q]
        def fk(self,target,g,arm):clock.sleep(.015);calls.append(arm);return target[0 if arm=='left' else 1,:3],np.eye(3)
        def command(self,*a):pytest.fail('preflight must never command')
        def home(self,*a):pytest.fail('preflight must never home')
        def enable(self,*a):pytest.fail('preflight must never enable')
        def force(self,*a):pytest.fail('preflight must never force')
    class Vision:
        def capture(self):clock.sleep(.02);return SimpleNamespace(stamp=clock(),roles={'red_mug':[0,0,0],'white_mug':[0,0,0],'box':[0,0,0]})
    cfg=json.loads((ROOT/'evaluations/gemini_graph_v19/CONFIG_PROPOSED.json').read_text())['runtime']['safety']
    return clock,Robot(),Vision(),SafetyLayer(cfg,clock=clock),calls


def test_readonly_preflight_reports_fk_rtt_unknown_jaw_no_motion():
    clock,robot,vision,safety,calls=fake_io();report=preflight(robot,vision,safety,[1.,1.],clock=clock)
    assert report['passed'] and calls==['left','right'] and report['hardware_commands']==0
    assert report['actual_aperture_m']==[None,None] and report['gripper_reference_is_command_not_measurement']
    assert report['FK_call_count']==2 and all(abs(row['wall_sec']-.015)<1e-8 for row in report['FK_RTT_samples'])


def test_slow_readonly_fk_cannot_make_cached_rgb_fresh():
    clock,robot,vision,safety,calls=fake_io();original=robot.fk
    def slow(*args):clock.sleep(.15);return original(*args)
    robot.fk=slow
    with pytest.raises(SafetyError,match='freshness') as error:preflight(robot,vision,safety,[1.,1.],clock=clock)
    assert error.value.preflight_report['RGBD_age_at_end_sec']>.25


def test_v19_relative_receipt_and_archive_inputs_exclude_secret_or_scene():
    report=check(PACKAGE,(PACKAGE/'local/ROOT_RECEIPT_SHA256.txt').read_text().strip())
    assert report['shared_runtime_sha256']=='c7b35f7ecfcaf2e2a3cb227f197888956f0ac7e3e95fa02e5a72851af4366e1c'
    assert not any(path.suffix=='.key' or path.name=='scene.xml' for path in PACKAGE.rglob('*'))
    cfg=json.loads((PACKAGE/'local/CONFIG_REAL.json').read_text())
    assert cfg['stage_judge']['key_file']=='/home/clover/.config/graph-self-adapter/openrouter.key'


def test_packaged_default_check_never_reads_key_or_connects(monkeypatch,tmp_path):
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('check connected'))
    cfg=json.loads((ROOT/'evaluations/gemini_graph_v19/CONFIG_PROPOSED.json').read_text())['runtime'];cfg['robot_TCP_offset_m']=.197;cfg['stage_judge']['key_file']='/never/read/openrouter.key'
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    original=Path.read_text
    def read_text(self,*a,**k):
        if self.suffix=='.key':pytest.fail('check read key')
        return original(self,*a,**k)
    monkeypatch.setattr(Path,'read_text',read_text)
    source=PACKAGE/'source';sys.path.insert(0,str(source));sys.path.insert(0,str(source/'tools/deployment'))
    spec=importlib.util.spec_from_file_location('_portable_v19_entry',source/'tools/deployment/promoted_gemini_real_entry.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    report=entry.main(['check','--package',str(PACKAGE),'--config',str(config)])
    assert report['device_connections']==0 and report['hardware_commands']==0
    text=(source/'tools/deployment/promoted_gemini_real_entry.py').read_text()
    assert 'GeminiStageJudge' in text and 'from gemini_readonly_preflight import preflight' in text
