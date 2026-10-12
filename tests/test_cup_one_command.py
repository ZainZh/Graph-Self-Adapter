"""File-only checks for the independent operator wrapper."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_cup_one_command_test',ROOT/'tools/deployment/cup_one_command.py')
wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)


def test_sim_config_cloned_and_grounding_bound_to_fresh_output(tmp_path):
    source=tmp_path/'source.json';original={'runtime':{'vision':{'role_boxes':{'red_mug':[1,2,3,4]},'prompt_frame_sha256':'old','prompt_reply_path':'old'}}}
    source.write_text(json.dumps(original));before=source.read_bytes()
    cfg,aux=wrapper.fresh_config_plan(source,'sim',tmp_path/'control',tmp_path/'output')
    assert cfg['runtime']['vision']['prompt_reply_path']==str(tmp_path/'output/GROUNDING/current_first_frame_boxes.json')
    assert cfg['runtime']['vision']['prompt_frame_output']==str(tmp_path/'output/GROUNDING/current_first_frame.png')
    assert cfg['runtime']['vision']['role_boxes']=={} and cfg['runtime']['vision']['prompt_frame_sha256'] is None
    assert not aux and source.read_bytes()==before and not (tmp_path/'control').exists()


def test_real_runtime_clone_preserves_safety_and_does_not_write_or_connect(tmp_path):
    runtime=tmp_path/'runtime.json';runtime.write_text(json.dumps({'vision':{'role_boxes':{}},'hardware_safety_reviewed':False,'initial_gripper_state_confirmed':False}))
    source=tmp_path/'source.json';source.write_text(json.dumps({'real_operator_config':str(runtime)}));before=runtime.read_bytes()
    cfg,aux=wrapper.fresh_config_plan(source,'real',tmp_path/'control',tmp_path/'output')
    clone=aux[Path(cfg['real_operator_config'])]
    assert clone['hardware_safety_reviewed'] is False and clone['initial_gripper_state_confirmed'] is False
    assert clone['vision']['prompt_reply_path'].startswith(str(tmp_path/'output'))
    assert runtime.read_bytes()==before and not (tmp_path/'control').exists()
