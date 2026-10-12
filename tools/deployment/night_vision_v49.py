import io,json
from pathlib import Path
from PIL import Image
from shared_cup_vision import SharedSAMVision
from api_grounding_v28 import proposal,IDENTITY
class AutoTaskVision(SharedSAMVision):
    def capture(self):
        if self.tracker is None and not self.config.get('role_boxes'):
            camera=self.camera;sample=camera.capture();b=io.BytesIO();Image.fromarray(sample[1]).save(b,format='PNG');path=Path(self.config['prompt_frame_output']);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b.getvalue())
            value=proposal(path,self.config['role_descriptions'],path.parent/'API_GROUNDING/attempt_01');self.config.update(role_boxes=value['role_boxes'],prompt_frame_sha256=value['frame_sha256']);Path(self.config['prompt_reply_path']).write_text(json.dumps(value,indent=2))
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
        return super().capture()
