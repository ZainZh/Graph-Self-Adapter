"""Select actual current Codex Root directly or bounded Gemini then Root fallback."""
from pathlib import Path
from cup_v25_root_grounding_vision_v2 import RootFallbackGroundingVision as V2Vision
from prove.xtrainer.io import SafetyError

class RootFallbackGroundingVision(V2Vision):
    def _initialize_autonomously(self):
        provider=self.config.get('initial_grounding_provider','gemini_then_codex_root')
        if provider=='gemini_then_codex_root':
            obs=super()._initialize_autonomously()
        elif provider=='codex_root':
            self._actual_root_provider=True
            self.config['prompt_frame_output']=str(Path(self.config['prompt_frame_output']).with_name('INITIAL_ROOT_RGB.png'))
            self.config['prompt_reply_path']=str(Path(self.config['prompt_reply_path']).with_name('INITIAL_ROOT_BOXES.json'))
            obs=self._initialize_selected_provider()
        else:raise SafetyError('unsupported initial grounding provider')
        if self._actual_root_provider:obs.source='RGBD+actual_SAM2+actual_current_Codex_Root_grounding+functional_part_gate'
        return obs
