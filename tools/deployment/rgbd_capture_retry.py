"""Bounded reacquisition of rejected RGB-D; no command, restore, or age relaxation."""
from collections import deque
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write


def bounded_rgbd_capture(owner,capture):
    history=deque(owner.rgb_history,maxlen=owner.rgb_history.maxlen)
    for attempt in range(3):
        try:return capture()
        except SafetyError as exc:
            reason=str(exc)
            if not (reason.startswith('stale real RGB-D') or reason=='RGBD/SAM processing too stale'):raise
            owner.rgb_history=deque(history,maxlen=history.maxlen)
            if not hasattr(owner,'rgbd_discard_log'):owner.rgbd_discard_log=[]
            stamp=getattr(owner.vision,'latest_capture_stamp',None)
            owner.rgbd_discard_log.append({'reason':reason,'retry_index':attempt,'observed_age_sec':None if stamp is None else owner.clock()-stamp,
                'discard_count':len(owner.rgbd_discard_log)+1,'no_command_sent_by_retry':True,'old_history_restored':True})
            for row in getattr(owner.vision,'frame_records',[]):
                if row.get('observed_monotonic_sec')==stamp:row['discarded_due_to_stale']=True
            write(owner.output/'RGBD_DISCARDS.json',{'max_reacquisitions_per_capture':2,'max_age_unchanged_sec':owner.cfg['safety']['max_age_sec'],'discards':owner.rgbd_discard_log})
            if attempt==2:raise
