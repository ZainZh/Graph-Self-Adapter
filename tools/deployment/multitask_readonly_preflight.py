"""Task-role-generalized read-only preflight; V19 source remains untouched."""
import time
import numpy as np
from prove.xtrainer.io import SafetyError,feedback_fk_check


def preflight(robot,vision,safety,gripper_reference,*,required_roles,clock=time.monotonic):
    calls=[]
    class ReadOnlyFK:
        def fk(self,*args):
            start=clock()
            try:return robot.fk(*args)
            finally:calls.append({'wall_sec':clock()-start})
    started=clock();observation=vision.capture();samples=robot.samples();safety.samples(samples)
    if any(s.running or s.mode!=5 or np.max(abs(s.qdot))>.01 for s in samples):raise SafetyError('read-only preflight actual robot not stationary')
    comparison=feedback_fk_check(ReadOnlyFK(),samples,gripper_reference)
    latest=robot.samples();safety.samples(latest)
    age=clock()-observation.stamp;required_roles=tuple(required_roles)
    report={'mode':'shadow','hardware_commands':0,'camera_robot_read_only':True,'FK_feedback':comparison,'FK_RTT_samples':calls,'FK_call_count':len(calls),'vision_plus_feedback_wall_sec':clock()-started,'RGBD_age_at_end_sec':age,'feedback_age_sec':[clock()-s.stamp for s in latest],'required_roles':list(required_roles),'roles':{k:None if v is None else np.asarray(v).tolist() for k,v in observation.roles.items()},'actual_aperture_m':[s.aperture for s in latest],'gripper_reference_is_command_not_measurement':True,'passed':bool(comparison['passed'] and 0<=age<=safety.config['max_age_sec'] and all(observation.roles.get(k) is not None for k in required_roles))}
    if not report['passed']:
        error=SafetyError('read-only preflight FK/role/freshness failed');error.preflight_report=report;raise error
    return report
