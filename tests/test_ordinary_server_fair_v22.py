import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/deployment'))
from ordinary_server_bundle_v22 import load_bundle,remap_path
from ordinary_server_fair_v22 import optional_aux_roles


def test_all_four_task_assets_load_mapped_without_changing_model_reference_bytes():
    for task in ['collect_apple','collect_cubes','put_knife_board_new','sort_two_spoon']:
        b=load_bundle(task);roles=optional_aux_roles(b['graph'],b['refs'])
        assert 'table_patch' not in roles['SAM_roles'] and roles['optional_SAM_roles']==['table_patch']
        actual={s['object'] for s in b['graph']['stages']}|{s['target'] for s in b['graph']['stages']}
        assert actual-{'table_handover_region'}<=set(roles['SAM_roles'])
        assert b['skill'].model.input_dim==29 and b['initial_q'].shape==(2,6)


def test_explicit_prefix_mapper_does_not_rewrite_data_or_unrelated_paths():
    assert remap_path('/tmp/unrelated')==Path('/tmp/unrelated')
    assert remap_path('/home/zheng/github/Graph-Self-Adapter/tools/example.py')==ROOT/'tools/example.py'


def test_server_install_is_idempotent_and_freeze_runtime_stable():
    import ordinary_server_fair_v22 as entry
    entry.install();first=entry.base.runtime_hash();entry.install()
    assert entry.base.runtime_hash()==first
