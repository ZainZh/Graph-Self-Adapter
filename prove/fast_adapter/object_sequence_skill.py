"""Small causal RGB/proprio action model; ordinary recurrent inference, no TTT.
Used by the P29 hammer development repair. The optional outer adapter still only
updates z in support; model weights and z remain fixed during query execution.
"""
import torch
from torch import nn
from prove.fast_adapter.object_interaction_adapter import inputs

class ObjectSequenceSkill(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(150,128),nn.SiLU(),nn.LayerNorm(128))
        self.gru=nn.GRU(128,64,batch_first=True)
        self.head=nn.Sequential(nn.Linear(64,64),nn.SiLU(),nn.Linear(64,4));self._hidden=None
    def reset_rollout(self):self._hidden=None
    def forward_sequence(self,v,p,previous):
        u=self.encoder(inputs(v,p,previous));h,_=self.gru(u);return self.head(h).tanh()
    def forward(self,v,p,previous):
        if v.ndim==3:return self.forward_sequence(v,p,previous)
        u=self.encoder(inputs(v,p,previous))[:,None]
        h,self._hidden=self.gru(u,self._hidden)
        return self.head(h[:,0]).tanh()
