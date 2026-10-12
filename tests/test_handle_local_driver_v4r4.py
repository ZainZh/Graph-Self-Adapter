import importlib.util
from pathlib import Path
from collections import deque
from types import SimpleNamespace
import pytest
from prove.xtrainer.io import SafetyError
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_handle_retry_v4r4',ROOT/'tools/deployment/handle_local_driver_v4r4.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def fixture(tmp_path):
    ex=driver.LocalExecutor.__new__(driver.LocalExecutor);ex.rgb_history=deque([('accepted_old_frame',)],maxlen=3);ex.output=tmp_path;ex.cfg={'safety':{'max_age_sec':.25}}
    ex.vision=SimpleNamespace(latest_capture_stamp=1.);ex.clock=lambda:1.31;ex.command_index=0
    return ex


def test_stale_reacquires_actual_frame_restores_history_without_command(tmp_path,monkeypatch):
    ex=fixture(tmp_path);calls=[]
    def capture(self):
        calls.append(len(calls));self.rgb_history.append(('discarded_stale' if len(calls)==1 else 'new_actual_frame',))
        if len(calls)==1:raise SafetyError('stale real RGB-D')
        return 'fresh-feedback','fresh-new-observation'
    monkeypatch.setattr(driver.SharedCupExecutor,'capture',capture)
    assert ex.capture()==('fresh-feedback','fresh-new-observation')
    assert len(calls)==2 and ex.command_index==0 and ('discarded_stale',) not in ex.rgb_history
    assert ex.rgbd_discard_log[0]['observed_age_sec']==pytest.approx(.31)


def test_non_rgbd_failure_is_not_swallowed(tmp_path,monkeypatch):
    ex=fixture(tmp_path);calls=[]
    def capture(self):calls.append(1);raise SafetyError('stale actual robot feedback')
    monkeypatch.setattr(driver.SharedCupExecutor,'capture',capture)
    with pytest.raises(SafetyError,match='robot feedback'):ex.capture()
    assert len(calls)==1 and ex.command_index==0


def test_persistent_rgbd_stale_stops_after_two_new_reacquisitions(tmp_path,monkeypatch):
    ex=fixture(tmp_path);calls=[]
    def capture(self):calls.append(1);self.rgb_history.append(('bad',));raise SafetyError('RGBD/SAM processing too stale')
    monkeypatch.setattr(driver.SharedCupExecutor,'capture',capture)
    with pytest.raises(SafetyError,match='processing too stale'):ex.capture()
    assert len(calls)==3 and len(ex.rgbd_discard_log)==3 and ex.command_index==0
    assert list(ex.rgb_history)==[('accepted_old_frame',)]
