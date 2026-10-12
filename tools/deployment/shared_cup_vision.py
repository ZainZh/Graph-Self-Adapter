"""Same actual-SAM LiveVision for sim and real, with sparse mask/RGB evidence."""
import hashlib
from pathlib import Path
import time
import numpy as np
from PIL import Image
from prove.xtrainer.perception import LiveVision
from prove.xtrainer.agent import write,digest


class SharedSAMVision(LiveVision):
    def __init__(self,*args,evidence_root,**kwargs):
        super().__init__(*args,**kwargs);self.evidence_root=Path(evidence_root)
        self.mask_records=[];self.frame_count=0;self.stage_context=lambda:None
    def _tracker(self):
        tracker=super()._tracker();owner=self
        class RecordedTracker:
            def record(self,rgb,masks,method):
                index=owner.frame_count;owner.frame_count+=1
                record={'frame_index':index,'wall_recorded_sec':time.monotonic(),'method':method,'mask_source':'actual_SAM2',
                    'physical_time_sec':getattr(owner.camera,'last_physical_time',None),'stage_id':(owner.stage_context() or {}).get('stage_id'),
                    'mask_pixels':{role:int(np.asarray(mask,bool).sum()) for role,mask in masks.items()}}
                if index==0 or index%10==0:
                    owner.evidence_root.mkdir(parents=True,exist_ok=True)
                    frame=owner.evidence_root/f'RGB_{index:06d}.jpg';Image.fromarray(rgb).save(frame,quality=90)
                    archive=owner.evidence_root/f'MASKS_{index:06d}.npz';np.savez_compressed(archive,**{r:np.asarray(m,bool) for r,m in masks.items()})
                    record.update(RGB={'path':str(frame),'sha256':digest(frame)},masks={'path':str(archive),'sha256':digest(archive)})
                owner.mask_records.append(record)
            def initialize(self,rgb,boxes):
                masks=tracker.initialize(rgb,boxes);self.record(rgb,masks,'Root_box_initialization');return masks
            def track(self,rgb):
                masks=tracker.track(rgb);self.record(rgb,masks,'SAM_temporal_tracking');return masks
        return RecordedTracker()
    def close(self):
        if self.mask_records:write(self.evidence_root/'MASK_EVIDENCE.json',{'mask_source':'actual_SAM2','GT_fallback':False,'frames':self.mask_records})
        return super().close()
