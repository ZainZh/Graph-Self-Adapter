"""P29 cup candidate2: more Cartesian correction range, less gripper disturbance."""
import torch
from prove.fast_adapter.object_interaction_adapter import ObjectAdapter
class ObjectAdapterV2(ObjectAdapter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.register_buffer('correction_scale',torch.tensor([1.5,1.5,1.5,.25]))
    def action(self,v,p,prev,base,z):
        u=self.encode(v,p,prev);value=self.actor(torch.cat((u,base,z),-1));prior=self.actor(torch.cat((u,base,self.z0.expand_as(z)),-1))
        return (base+self.correction_scale*torch.tanh(value-prior)).clamp(-1,1)
