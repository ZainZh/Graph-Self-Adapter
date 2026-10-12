import importlib.util
import json
from pathlib import Path
import shutil
import socket
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/deployment'))
from trajectory_promotion_check import check,digest
PACKAGE=ROOT/'evaluations/stage_trajectory_v18/deployment/cup_trajectory_v18_r1'


def test_v18_receipt_relative_files_and_runtime_no_sim_scene():
    report=check(PACKAGE,digest(PACKAGE/'PROMOTION_RECEIPT.json'))
    assert report['device_connections']==0 and report['simulation_reexecuted'] is False
    assert not any(p.name=='scene.xml' for p in PACKAGE.rglob('*'))
    assert report['shared_runtime_sha256']=='d8d508af876f16a50ad55a25447bd057aaa655f9e5a3c1aa713ccc9e0c1adba5'


def test_changed_promoted_runtime_rejected(tmp_path):
    destination=tmp_path/'package';shutil.copytree(PACKAGE,destination)
    receipt_sha=digest(destination/'PROMOTION_RECEIPT.json')
    with (destination/'source/tools/deployment/trajectory_white_runtime_v18.py').open('a') as f:f.write('\n# changed\n')
    with pytest.raises(ValueError,match='source/data'):check(destination,receipt_sha)


def test_packaged_default_check_no_network_and_new_executor(monkeypatch,tmp_path):
    def forbidden(*a,**k):pytest.fail('file-only package check connected a socket')
    monkeypatch.setattr(socket,'create_connection',forbidden)
    cfg=json.loads((ROOT/'evaluations/stage_trajectory_v18/CONFIG_PROPOSED.json').read_text())['runtime'];cfg['robot_TCP_offset_m']=.197
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    # Actual packaged entry and packaged sources, not the workstation controller.
    source=PACKAGE/'source';sys.path.insert(0,str(source));sys.path.insert(0,str(source/'tools/deployment'))
    spec=importlib.util.spec_from_file_location('_packaged_v18_entry',source/'tools/deployment/promoted_trajectory_real_entry.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    report=entry.main(['check','--package',str(PACKAGE),'--config',str(config)])
    assert report['device_connections']==0 and report['physical_qualification_inferred'] is False
    text=(source/'tools/deployment/promoted_trajectory_real_entry.py').read_text()
    assert 'import trajectory_white_runtime_v18 as runtime' in text and 'runtime.SharedCupExecutor(bundle' in text

def test_packaged_v18_segment_mock_uses_new_scheduler_and_keeps_passive_arm(tmp_path):
    from test_trajectory_white_runtime import executor
    from test_stage_trajectory import Clock,Robot,fk
    import numpy as np
    from types import SimpleNamespace
    source=PACKAGE/'source';spec=importlib.util.spec_from_file_location('_packaged_v18_runtime',source/'tools/deployment/trajectory_white_runtime_v18.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    fixture=executor();x=runtime.TrajectoryWhiteExecutor.__new__(runtime.TrajectoryWhiteExecutor);x.__dict__.update(fixture.__dict__)
    x.lower_contact_tolerance=.003;x.maximum_stall_sec=.25;clock=Clock();start=np.zeros((2,6));start[1,:3]=.1
    robot=Robot(start,clock);robot.fk=fk;x.robot=robot;x.command_g=np.array([1.,1.]);x.clock=clock;x.sleep=clock.sleep;x.T=np.eye(4)
    x.b={'initial_g':[1.,1.]};x.macro_grasp=None;x.output=tmp_path;x._hardware_attempts=0;x._hardware_acknowledged=0;x.command_index=0
    x.safety=SimpleNamespace(last_target=None,samples=lambda samples:None);x.stops=[robot];x.guard_limits=lambda:None
    x.boundary_capture=lambda:(robot.samples(),SimpleNamespace(stamp=clock()));x._shared_pose=lambda q,g,arm:fk(q,g,arm)
    from stage_trajectory import Limits
    x.trajectory_limits=Limits([-1]*6,[1]*6,[-1]*3,[1]*3,rate_hz=25,velocity=.3,acceleration=.6,position_tolerance=.001)
    _,progress,record=x.command(x.stage,.1,robot.samples(),None,.1)
    assert progress==1 and record['segment']['completed'] and np.array_equal(robot.q[1],start[1])
    assert record['segment']['hard_realtime_claim'] is False and x.command_index==len(robot.commands)
