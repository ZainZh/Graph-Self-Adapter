import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from cup_v25_root_grounding_vision_v2 import RootFallbackGroundingVision
from autonomous_sam_grounding import GroundingUnknown
from prove.xtrainer.io import SafetyError

def test_real_auto_grounding_unknown_enters_fresh_root_provider():
    vision=RootFallbackGroundingVision.__new__(RootFallbackGroundingVision)
    vision.config={'prompt_frame_output':'/tmp/current/INITIAL_RGB.png','prompt_reply_path':'/tmp/current/INITIAL_BOXES.json'}
    providers=[]
    def select():
        providers.append(vision._actual_root_provider)
        if not vision._actual_root_provider:raise GroundingUnknown('automatic grounding remained invalid/unknown after bounded review')
        return 'Root_current_frame_with_original_validator'
    vision._initialize_selected_provider=select
    assert vision._initialize_autonomously()=='Root_current_frame_with_original_validator'
    assert providers==[False,True]
    assert vision.config['prompt_frame_output']=='/tmp/current/INITIAL_ROOT_RGB.png'
    assert vision.config['prompt_reply_path']=='/tmp/current/INITIAL_ROOT_BOXES.json'

def test_unrelated_safety_failure_does_not_bypass_safety():
    vision=RootFallbackGroundingVision.__new__(RootFallbackGroundingVision);vision.config={}
    def unsafe():raise SafetyError('actual feedback stale')
    vision._initialize_selected_provider=unsafe
    with pytest.raises(SafetyError,match='actual feedback stale'):vision._initialize_autonomously()
    assert vision._actual_root_provider is False
