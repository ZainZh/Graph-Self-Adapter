"""Frozen local visual tokens with optional second, train-only PCA projection."""
import torch
from prove.fast_adapter.skill_tokens_v02 import LocalVision

class DualLocalVision(LocalVision):
    def __init__(self,pca,secondary=None):
        super().__init__(None);self.primary_table=pca['local'];self.secondary_table=None if secondary is None else secondary['local']

    def encode(self,images,masks):
        primary=[];secondary=[]
        for start in range(0,len(images),8):
            raw=super().encode(images[start:start+8],masks[start:start+8])['local']
            p=self.primary_table;primary.append((raw-p['mean'])@p['basis']/p['scale'])
            if self.secondary_table is not None:
                p=self.secondary_table;secondary.append((raw-p['mean'])@p['basis']/p['scale'])
        out={'local':torch.cat(primary)}
        if secondary:out['secondary']=torch.cat(secondary)
        return out
