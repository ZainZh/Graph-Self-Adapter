import io,json
import numpy as np
from pathlib import Path
from PIL import Image
from shared_cup_vision import SharedSAMVision
from api_grounding_v28 import proposal,IDENTITY
class AutoTaskVision(SharedSAMVision):
    def _tracker(self):
        base=super()._tracker();owner=self
        class Tracker:
            def initialize(self,*args):m=base.initialize(*args);owner.public_masks=m;return m
            def track(self,*args):m=base.track(*args);owner.public_masks=m;return m
        return Tracker()
    def capture(self):
        if self.tracker is None and not self.config.get('role_boxes'):
            camera=self.camera;sample=camera.capture();b=io.BytesIO();Image.fromarray(sample[1]).save(b,format='PNG');path=Path(self.config['prompt_frame_output']);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b.getvalue())
            value=proposal(path,self.config['role_descriptions'],path.parent/'API_GROUNDING/attempt_01');self.background_roi=value['role_boxes'].get('support_plane_region');boxes={k:v for k,v in value['role_boxes'].items() if k!='support_plane_region'};self.config.update(role_boxes=boxes,prompt_frame_sha256=value['frame_sha256']);Path(self.config['prompt_reply_path']).write_text(json.dumps(value,indent=2))
            class Once:
                def __init__(self):self.once=True
                def __getattr__(self,k):return getattr(camera,k)
                def capture(self):
                    if self.once:self.once=False;return sample
                    return camera.capture()
            self.camera=Once()
            try:obs=super().capture()
            finally:self.camera=camera
            obs.source+='+'+IDENTITY;return obs
        obs=super().capture();obs.roles.pop('red_knife_grasp_part',None);return obs
