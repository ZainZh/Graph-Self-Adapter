import importlib.util
from pathlib import Path
import hashlib
import json
import pytest
ROOT=Path(__file__).resolve().parents[1]

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools/deployment'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_failure_cannot_create_promotion_package(tmp_path):
    module=load('macro_promotion_receipt');result=tmp_path/'RESULT.json';result.write_text(json.dumps({'strict_success':False,'graph_completed':False}));freeze=tmp_path/'FREEZE.json';freeze.write_text(json.dumps({'status':'result_free_frozen'}))
    with pytest.raises(ValueError,match='strict full-task'):module.build(freeze,result,tmp_path/'package','test-fixture-only')
    assert not (tmp_path/'package').exists()


def receipt_fixture(tmp_path):
    check=load('macro_promotion_check');source=tmp_path/'source.py';source.write_text('test fixture only\n');sha=check.digest(source);runtime=hashlib.sha256(sha.encode()).hexdigest()
    receipt={'schema':'graph-self-adapter.macro-promotion-receipt.v1','status':'Root-attested-engineering-simulation-pass','task':'put_mug_on_box','files':[{'relative_path':'source.py','sha256':sha}],
        'runtime_relative_paths':['source.py'],'shared_runtime_sha256':runtime,'external_resources':{},'fixture_only':True}
    path=tmp_path/'PROMOTION_RECEIPT.json';path.write_text(json.dumps(receipt));return check,path,source


def test_laptop_checks_relative_identity_without_simulator_scene_or_connection(tmp_path):
    check,path,source=receipt_fixture(tmp_path);result=check.check(tmp_path,check.digest(path))
    assert result['device_connections']==0 and result['simulation_reexecuted'] is False and result['scene_assets_required_for_real_check'] is False
    source.write_text('modified\n')
    with pytest.raises(ValueError,match='changed'):check.check(tmp_path,check.digest(path))


def test_independent_receipt_approval_and_path_containment_required(tmp_path):
    check,path,source=receipt_fixture(tmp_path)
    with pytest.raises(ValueError,match='approved receipt'):check.check(tmp_path,'wrong')
    receipt=json.loads(path.read_text());receipt['files'][0]['relative_path']='../outside.py';receipt['runtime_relative_paths']=['../outside.py'];path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='missing or changed'):check.check(tmp_path,check.digest(path))


def test_direct_runtime_identity_list_matches_frozen_entry_hash_without28_assumption():
    import sys
    sys.path.insert(0,str(ROOT/'tools/deployment'))
    import direct_white_entry
    builder=load('macro_promotion_receipt')
    computed=hashlib.sha256(''.join(builder.digest(ROOT/p) for p in builder.DIRECT_RUNTIME_PATHS).encode()).hexdigest()
    assert computed==direct_white_entry.runtime_hash()
    import inspect
    assert '!=28' not in inspect.getsource(builder.verified_success)
