"""Fake camera boundary tests only; no live devices, sockets or startup service."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

path=Path(__file__).resolve().parents[1]/'tools/deployment/operator_launcher.py'
spec=importlib.util.spec_from_file_location('_operator_launcher_test',path)
operator=importlib.util.module_from_spec(spec);spec.loader.exec_module(operator)
class Error(RuntimeError):pass


def camera_base(outcomes):
    class Base:
        def __init__(self):self.outcomes=list(outcomes);self.real_grabs=0
        def capture(self):
            self.real_grabs+=1;outcome=self.outcomes.pop(0)
            if isinstance(outcome,Exception):raise outcome
            return SimpleNamespace(stamp=outcome,frame_id=self.real_grabs)
    return Base


def test_startup_discards_stale_and_acquires_new_frame_then_later_stale_rejected():
    Base=camera_base([Error('RGBD/SAM processing too stale'),Error('RGBD/SAM processing too stale'),3.25,
                      Error('RGBD/SAM processing too stale')])
    vision=operator.startup_vision_class(Base,Error)();frame=vision.capture()
    assert vision.real_grabs==3 and frame.stamp==3.25 and frame.frame_id==3
    with pytest.raises(Error,match='too stale'):vision.capture()
    assert vision.real_grabs==4


def test_startup_retry_budget_is_exactly_two():
    Base=camera_base([Error('RGBD/SAM processing too stale')]*4)
    vision=operator.startup_vision_class(Base,Error)()
    with pytest.raises(Error,match='too stale'):vision.capture()
    assert vision.real_grabs==3


@pytest.mark.parametrize('error',[Error('actual feedback stale'),RuntimeError('CUDA kernel error')])
def test_unrelated_errors_are_never_swallowed(error):
    vision=operator.startup_vision_class(camera_base([error,1.]),Error)()
    with pytest.raises(type(error),match=str(error)):vision.capture()
    assert vision.real_grabs==1


def test_run_requires_both_explicit_operator_declarations_before_import_or_connect(monkeypatch):
    monkeypatch.setattr(operator,'load_cli',lambda *a:pytest.fail('package loaded before run flags'))
    for flags in ([],['--safety-reviewed'],['--initial-gripper-confirmed']):
        with pytest.raises(ValueError,match='requires explicit'):
            operator.main(['run','--package','/unopened/package',*flags])
    operator.require_run_declarations('run',True,True)
