"""Separate V25 task-bound install, no hardware connections."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import multitask_ordinary_entry_v21 as base
from multitask_ordinary_entry_v21 import bundle_check,bindings,runtime_hash
backend=base.backend
import multitask_ordinary_runtime as parent_runtime
from multitask_ordinary_route import role_contract as original_roles
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import digest
HERE=Path(__file__).resolve().parent

def roles(graph,refs):
    result=original_roles(graph,refs)
    actual={s['object'] for s in graph['stages']}|{s['target'] for s in graph['stages']}
    if 'table_patch' not in actual:result['SAM_roles']=[r for r in result['SAM_roles'] if r!='table_patch'];result['optional_SAM_roles']=['table_patch']
    return result
base.role_contract=roles;parent_runtime.role_contract=roles
original_load=base.backend.load

def loader(path,name):
    if Path(path).name=='sim_robot_io.py':
        import ordinary_v25_cube_observer_sim_io
        return ordinary_v25_cube_observer_sim_io
    if Path(path).name=='direct_white_runtime.py':
        import ordinary_v25_cube_pad_runtime
        return ordinary_v25_cube_pad_runtime
    return original_load(path,name)
base.backend.load=loader
base.SOURCES=base.SOURCES+tuple(HERE/name for name in ('ordinary_v25_cube_pad_entry.py','ordinary_v25_cube_pad_adapter.py','ordinary_v25_cube_pad_runtime.py','ordinary_v25_cube_pad_judge.py','ordinary_v25_cube_observer_runtime.py','ordinary_v25_apple_pad_runtime.py','ordinary_v25_cube_observer_sim_io.py','ordinary_v25_cube_observer_agent.py','ordinary_v25_cube_observer_route.py','ordinary_v25_cube_pad_route.py','three_task_v26_public_io.py','ordinary_v25_repair_runtime.py','ordinary_v25_runtime.py','ordinary_v25_sim_io.py','kinematic_sim_robot_io.py','fair_sim_robot_io.py','all_code_stage_runtime.py','all_code_stage_trajectory.py','stage_trajectory.py','trajectory_white_runtime_v17.py','stage_trajectory_v17.py'))
# Own module remains a config/freeze adapter: only this fork's process-global bindings are replaced.
_original_main=base.main

def main(argv=None):
    parser=__import__('argparse').ArgumentParser(add_help=False);parser.add_argument('--config',type=Path,required=True);a,_=parser.parse_known_args(argv)
    cfg=json.loads(a.config.read_text())
    from ordinary_v25_cube_pad_judge import MultitaskSemanticJudge
    def judge(*args,**kwargs):return MultitaskSemanticJudge(*args,**kwargs,key_file=cfg['runtime']['stage_judge']['key_file'],maximum_calls=cfg['runtime']['stage_judge']['maximum_calls'])
    base.backend.RealRootJudge=judge;base.backend.bindings=base.bindings;base.backend.runtime_hash=base.runtime_hash;base.backend.TASK=cfg['task'];base.backend.VARIANT='ordinary_task_stage_trajectory_v25'
    base.backend.load=loader
    return base.backend.main(['sim','--config',str(a.config)]+list(argv)[list(argv).index('--scope'):])

_original_bindings=base.bindings
def bindings(config_path,config,mode):
    import copy
    cfg=copy.deepcopy(config);requested=cfg['runtime']['stage_judge']['maximum_calls'];base_calls=bundle_check(cfg['task'])['required_supported_judgements']
    if requested!=3*base_calls//2:raise SafetyError('repair declares exactly three semantic boundaries per object cycle')
    cfg['runtime']['stage_judge']['maximum_calls']=base_calls
    report,runtime,bundle,bound=_original_bindings(config_path,cfg,mode)
    runtime['stage_judge']['maximum_calls']=requested
    prior=Path(config['runtime']['cube_robot_pad_prior']);bound[str(prior.resolve())]=digest(prior)
    authority=Path(config['runtime']['actual_codex_observer_authority']);bound[str(authority.resolve())]=digest(authority)
    return report,runtime,bundle,bound
base.bindings=bindings

from ordinary_v25_cube_pad_route import observer_route
base.ordinary_route=observer_route
