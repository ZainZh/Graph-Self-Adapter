"""Frozen visual frontends for P21; no dynamics, labels, or actions are inputs."""
from pathlib import Path
import importlib.util
import numpy as np
import torch
from prove.fast_adapter.visual_alignment import FrozenDinoROI, DEFAULT_MODEL

SOURCE = Path('data/sources/lila-wam-reference-20260923/models/vla_model_fm.py')
WEIGHTS = Path('data/sources/lila-wam-reference-20260923/VISUAL_WEIGHTS.pt')
FRONTENDS = ('last', 'middle17', 'lila')

class ThreeFrontends(FrozenDinoROI):
    def __init__(self, model_dir=DEFAULT_MODEL, device='cuda', microbatch=16,
                 selected=None):
        super().__init__(model_dir, device, microbatch)
        self.selected = selected
        if selected in (None, 'lila'):
            spec = importlib.util.spec_from_file_location('p21_official_lila', SOURCE)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.fusion = module.MultiLayerConcatFusion(1024, 3, 1024,
                                proj_type='linear', pre_norm=True).to(self.device).eval()
            self.adapter = module.VisualFeatureAdapter(1024, 768, num_queries=64,
                         num_heads=8, num_layers=4, dropout=0.).to(self.device).eval()
            state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
            self.fusion.load_state_dict(state['fusion'], strict=True)
            self.adapter.load_state_dict(state['adapter'], strict=True)
            self.fusion.requires_grad_(False); self.adapter.requires_grad_(False)

    def pool(self, tokens, boxes):
        patches = tokens[:, self.prefix:].float().reshape(-1, 14, 14, 1024)
        b = torch.as_tensor(boxes, device=self.device)
        lo = torch.arange(14, device=self.device) * 16
        wx = (torch.minimum(b[:, :, 2, None], lo + 16) -
              torch.maximum(b[:, :, 0, None], lo)).clamp(min=0)
        wy = (torch.minimum(b[:, :, 3, None], lo + 16) -
              torch.maximum(b[:, :, 1, None], lo)).clamp(min=0)
        weights = wy[:, :, :, None] * wx[:, :, None, :]
        return torch.einsum('bryx,byxc->brc', weights, patches) / weights.sum((-1,-2)).clamp(min=1)[..., None]

    def encode_all(self, rgb, boxes):
        names = (self.selected,) if self.selected else FRONTENDS
        collected = {k: [] for k in names}
        for start in range(0, len(rgb), self.microbatch):
            stop = start + self.microbatch
            with torch.inference_mode():
                pixels = torch.as_tensor(np.array(rgb[start:stop], copy=True),
                            device=self.device).permute(0, 3, 1, 2).float() / 255
                pixels = ((pixels-self.mean)/self.std).to(torch.bfloat16)
                out = self.model(pixel_values=pixels, output_hidden_states=True, return_dict=True)
                assert len(out.hidden_states) == 25
                for name in names:
                    if name == 'last':
                        value = self.pool(out.last_hidden_state, boxes[start:stop])
                    elif name == 'middle17':
                        value = self.pool(out.hidden_states[17], boxes[start:stop])
                    else:
                        fused = self.fusion([out.hidden_states[i].float() for i in (-12,-8,-4)])
                        value = self.adapter(fused).flatten(1).unsqueeze(1)
                    assert torch.isfinite(value).all()
                    collected[name].append(value.detach().clone())
        return {k: torch.cat(v) for k, v in collected.items()}

    def encode(self, rgb, boxes):
        assert self.selected in FRONTENDS
        return self.encode_all(rgb, boxes)[self.selected]
