"""P29 tool candidate2: preserve grip and anchor known canonical behavior."""
import torch
from prove.fast_adapter.object_adapter_v2 import ObjectAdapterV2

class ObjectToolAdapterV2(ObjectAdapterV2):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.correction_scale.copy_(torch.tensor([.5,.5,.5,.1]))
