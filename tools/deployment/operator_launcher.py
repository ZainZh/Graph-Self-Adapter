#!/usr/bin/env python3
"""Operator-only local launcher; package/core bytes remain immutable."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid


def load_cli(package):
    package=Path(package).resolve()
    if str(package) not in sys.path:sys.path.insert(0,str(package))
    path=package/'tools/task_graph/xtrainer_portable.py'
    spec=importlib.util.spec_from_file_location('_operator_portable_cli',path)
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    return cli


def startup_vision_class(base,safety_error):
    class StartupFreshVision(base):
        """Discard stale startup output only; base always acquires a real new frame."""
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self._startup_succeeded=False
        def capture(self):
            if self._startup_succeeded:return super().capture()
            for attempt in range(3):
                try:
                    observation=super().capture()
                except safety_error as exc:
                    if str(exc)!='RGBD/SAM processing too stale' or attempt==2:raise
                    # Re-enter original capture: new camera grab + SAM track, unchanged timestamp.
                    continue
                self._startup_succeeded=True;return observation
            raise AssertionError('unreachable')
    return StartupFreshVision


def require_run_declarations(mode,safety_reviewed,initial_confirmed):
    if mode=='run' and not (safety_reviewed and initial_confirmed):
        raise ValueError('run requires explicit --safety-reviewed and --initial-gripper-confirmed operator declarations')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',nargs='?',choices=('check','shadow','run'),default='check')
    p.add_argument('--package',type=Path,required=True);p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--safety-reviewed',action='store_true')
    p.add_argument('--initial-gripper-confirmed',action='store_true')
    a=p.parse_args(argv);require_run_declarations(a.mode,a.safety_reviewed,a.initial_gripper_confirmed)
    package=a.package.resolve();config_path=(a.config or package/'local/CONFIG.json').resolve()
    cli=load_cli(package);cfg=json.loads(config_path.read_text())
    report,cfg,bundle=cli.portable_check(package,cfg)
    # This is the recorded nominal command, never current gripper feedback or an initialization action.
    expected_g=bundle['initial_g'].tolist();expected_q=bundle['initial_q'].tolist()
    report.update(expected_native_initial_gripper_command=expected_g,expected_native_initial_q_rad=expected_q,
        expected_start_note='Recorded expected commands only; actual current jaws unmeasured. Operator must independently establish/confirm start; no homing or initialization is performed.',
        operator_helper_sha256=cli.digest(Path(__file__)))
    if a.mode=='check':print(json.dumps(report,indent=2));return report
    if report['missing_external_files'] or not all(report['dependencies'].values()):
        raise cli.SafetyError('external dependency gap; no device connection made')
    cfg['known_initial_gripper_commands']=expected_g
    cfg['hardware_safety_reviewed']=bool(a.safety_reviewed) if a.mode=='run' else cfg['hardware_safety_reviewed']
    cfg['initial_gripper_state_confirmed']=bool(a.initial_gripper_confirmed) if a.mode=='run' else cfg['initial_gripper_state_confirmed']
    identifier=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:10]
    output=(a.output or package/'local/operator_runs'/identifier).resolve()
    if output.exists():raise FileExistsError('fresh task output required')
    control=package/'local/operator_controls'/identifier;control.mkdir(parents=True,exist_ok=False)
    runtime_config=control/'CONFIG.json';scope_path=control/'SCOPE.json';freeze_path=control/'EXECUTION_FREEZE.json'
    cli.write(runtime_config,cfg)
    mode='execute' if a.mode=='run' else 'shadow'
    scope={'schema':'graph-self-adapter.xtrainer-operator-scope.v1','task':cli.TASK,
        'read_only_connections_authorized':True,'hardware_motion_commands_authorized':a.mode=='run',
        'maximum_hardware_motion_episodes':1 if a.mode=='run' else 0,
        'operator_declarations':{'safety_reviewed':a.safety_reviewed,'initial_gripper_confirmed':a.initial_gripper_confirmed,
                                'source':'explicit operator CLI flags; not Root or measured hardware acceptance'},
        'expected_start_q_rad':expected_q,'expected_start_gripper_command':expected_g,
        'no_auto_enable_home_or_initialization':True,'output':str(output)}
    cli.write(scope_path,scope)
    fresh_report,_,_=cli.portable_check(package,cfg)
    freeze=cli.prepare_freeze(fresh_report,runtime_config,scope_path,output,mode,cfg['root_identity'])
    helper=Path(__file__).resolve();helper_sha=cli.digest(helper)
    framer_source=helper.with_name('real_feedback_stream.py')
    if not framer_source.is_file():raise cli.SafetyError('bound real feedback framer source missing')
    framer_sha=cli.digest(framer_source)
    freeze['bindings'].extend([{'path':str(helper),'sha256':helper_sha},{'path':str(framer_source),'sha256':framer_sha}])
    freeze['feedback_framing']={'packet_size':1440,'magic_offset':48,'max_discard_bytes':8192,'max_gap_events':8}
    cli.write(control/'PREPARED_FREEZE.json',freeze)
    # Explicit shadow/run invocation authorizes this exact operator job, not physical qualification.
    freeze={**freeze,'status':'result_free_frozen','freeze_source':'explicit operator invocation',
            'operator_declarations':scope['operator_declarations']}
    cli.write(freeze_path,freeze)
    if cli.digest(helper)!=helper_sha or cli.digest(framer_source)!=framer_sha:raise cli.SafetyError('operator helper changed before execution')
    cli.validate_execute(fresh_report,cfg,runtime_config,scope_path,freeze_path,mode=mode,output=output)
    stream_spec=importlib.util.spec_from_file_location('_operator_real_feedback_stream',framer_source)
    stream_module=importlib.util.module_from_spec(stream_spec);stream_spec.loader.exec_module(stream_module)
    cli.FeedbackStream=stream_module.FramedFeedbackStream
    cli.LiveVision=startup_vision_class(cli.LiveVision,cli.SafetyError)
    print(json.dumps({'operator_mode':a.mode,'output':str(output),'control_files':str(control),
                      'operator_declarations':scope['operator_declarations'],'expected_gripper_is_measurement':False}))
    return cli.main(['--package',str(package),'--config',str(runtime_config),'--scope',str(scope_path),
                     '--execution-freeze',str(freeze_path),'--output',str(output),'--execute' if a.mode=='run' else '--shadow'])

if __name__=='__main__':main()
