"""Single red-cup task, same tested red controllers, no white manipulation."""
import io,json,hashlib
from pathlib import Path
from PIL import Image
from api_grounding_v28 import proposal,IDENTITY
from handle_macro_runtime import MacroVision
from shared_cup_runtime import SharedCupExecutor as CommonExecutor
from cup_clear_goalset_v41 import SharedCupExecutor as BaseExecutor

class RedOnlyVision(MacroVision):
    def capture(self):
        if self.tracker is None and not self.config.get('role_boxes'):
            real=self.camera;sample=real.capture();b=io.BytesIO();Image.fromarray(sample[1]).save(b,format='PNG')
            frame=Path(self.config['prompt_frame_output']);frame.parent.mkdir(parents=True,exist_ok=True);frame.write_bytes(b.getvalue())
            roles={'red_mug':'whole red mug including handle','white_mug':'whole white mug, static background object; no manipulation requested','box':'whole wooden placement box'}
            payload=proposal(frame,roles,frame.parent/'RED_ONLY_GROUNDING/attempt_01')
            self.config['role_boxes']=payload['role_boxes'];self.config['prompt_frame_sha256']=payload['frame_sha256']
            Path(self.config['prompt_reply_path']).write_text(json.dumps(payload,indent=2))
            (frame.parent/'AUTONOMOUS_GROUNDING').mkdir(exist_ok=True)
            (frame.parent/'AUTONOMOUS_GROUNDING/INITIAL_GROUNDING_SOURCE.json').write_text(json.dumps({'source':IDENTITY,'human_root':False,'logical_task':'put_red_mug_on_box','white_is_background_only':True}))
            class Once:
                def __init__(self):self.once=True
                def __getattr__(self,name):return getattr(real,name)
                def capture(self):
                    if self.once:self.once=False;return sample
                    return real.capture()
            self.camera=Once()
            try:return super().capture()
            finally:self.camera=real
        return super().capture()

class RedOnlyExecutor(BaseExecutor):
    def start(self):
        # Bypass white-part, white-table and white size-prior initialization only.
        CommonExecutor.start(self)
        self.vision.initial_static_clouds={}
SharedCupExecutor=RedOnlyExecutor
SharedSAMVision=RedOnlyVision
