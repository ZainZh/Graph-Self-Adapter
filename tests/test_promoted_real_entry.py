import ast
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]


def test_real_launcher_is_real_motion_and_default_is_check():
    text=(ROOT/'tools/deployment/promoted_run.sh').read_text();assert 'mode=${1:-check}' in text and 'mode=execute' in text
    source=(ROOT/'tools/deployment/promoted_real_entry.py').read_text()
    assert '--safety-reviewed' not in source and '--initial-gripper-confirmed' not in source
    assert 'hardware_safety_reviewed=a.' not in source and 'initial_gripper_state_confirmed=a.' not in source


def test_all_connections_are_inside_main_after_default_check_returns():
    tree=ast.parse((ROOT/'tools/deployment/promoted_real_entry.py').read_text())
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef,ast.Import,ast.ImportFrom,ast.Assign,ast.Expr,ast.If)):continue
        pytest.fail('unexpected top-level hardware preparation')
    source=(ROOT/'tools/deployment/promoted_real_entry.py').read_text();assert source.index("if a.mode=='check':")<source.index('stream.connect()')
    assert 'DobotRobot(' not in source and 'EnableRobot(' not in source and 'Home(' not in source


def test_shadow_thin_io_with_mock_devices_has_zero_motion(tmp_path,monkeypatch):
    import importlib.util,sys,json,numpy as np
    from types import SimpleNamespace
    sys.path.insert(0,str(ROOT/'tools/deployment'))
    spec=importlib.util.spec_from_file_location('_promoted_mock_io',ROOT/'tools/deployment/promoted_real_entry.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    root=tmp_path/'pkg';(root/'local').mkdir(parents=True);(root/'PROMOTION_RECEIPT.json').write_text(json.dumps({'files':[],'shared_runtime_sha256':'mock_fixture'}));cfgfile=root/'local/CONFIG_REAL.json';cfgfile.write_text('{}');scope=root/'local/SCOPE_REAL.json';scope.write_text(json.dumps({'task':'put_mug_on_box','read_only_shadow_authorized':True}))
    cfg={'vision':{},'T_left_from_camera':np.eye(4).tolist(),'robot_ips':{'left':'mock_L','right':'mock_R'},'sdk_file':'mock_only','safety':{},'known_initial_gripper_commands':[1,1]}
    monkeypatch.setattr(entry,'prepare',lambda *a:({},cfg,{}));calls=[]
    import prove.xtrainer.io as io,prove.xtrainer.perception as perception,real_feedback_stream,operator_launcher,direct_white_runtime
    class RPC:
        commands=0
        def connect(self):calls.append('RPC read connection')
        def close(self):pass
        def command(self,*a):pytest.fail('shadow sent motion')
    class Stream:
        def __init__(self,*a):pass
        def connect(self):calls.append('feedback read connection')
        def snapshot(self):return SimpleNamespace(aperture=None)
        def close(self):pass
    class Vision:
        def __init__(self,*a,**kw):pass
        def connect(self):calls.append('mock camera')
        def capture(self):return SimpleNamespace(roles={'white_mug':np.zeros(3)})
        def close(self):pass
    monkeypatch.setattr(io,'NativeRPC',RPC);monkeypatch.setattr(real_feedback_stream,'FramedFeedbackStream',Stream);monkeypatch.setattr(io,'feedback_fk_check',lambda *a:{'passed':True,'mock_fixture_only':True});monkeypatch.setattr(io,'SafetyLayer',lambda *a:SimpleNamespace(samples=lambda *a:None))
    monkeypatch.setattr(perception,'ZEDCamera',lambda *a,**kw:object());monkeypatch.setattr(operator_launcher,'startup_vision_class',lambda *a:Vision)
    result=entry.main(['shadow','--package',str(root),'--config',str(cfgfile),'--scope',str(scope),'--receipt-sha256','mock_approval','--output',str(tmp_path/'output')])
    assert result['hardware_commands']==0 and result['actual_aperture_m']==[None,None] and calls.count('feedback read connection')==2
