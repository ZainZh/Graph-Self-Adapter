import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from cup_v25_root_grounding_vision_v3 import RootFallbackGroundingVision

def test_explicit_codex_root_provider_never_calls_initial_gemini():
    vision=RootFallbackGroundingVision.__new__(RootFallbackGroundingVision)
    vision.config={'initial_grounding_provider':'codex_root','prompt_frame_output':'/tmp/INITIAL_RGB.png','prompt_reply_path':'/tmp/INITIAL_BOXES.json'}
    calls=[]
    def selected():calls.append(vision._actual_root_provider);return SimpleNamespace(source='legacy')
    vision._initialize_selected_provider=selected
    result=vision._initialize_autonomously()
    assert calls==[True]
    assert result.source=='RGBD+actual_SAM2+actual_current_Codex_Root_grounding+functional_part_gate'
