"""Describe actual native pixel coverage and role visibility without outcome labels."""
import numpy as np
import torch
from tools.fast_adapter.p35_common import *


def main():
    torch.set_num_threads(2);rows=[];sources=[]
    for task in TASKS:
        b=load_data(task,'g224','cpu');keep=b['object_ids']>=48
        streams={'global224':{'tokens':b['tokens'][:,:,0],'boxes':b['boxes'][:,:,0],'presence':b['presence'][:,:,0]}}
        for stream in ('global448','wrist224','wrist448'):
            p=ROOT/task/'data'/f'{stream}.pt';streams[stream]=torch.load(p,weights_only=True);sources.append(p)
        for stream,data in streams.items():
            size=448 if stream.endswith('448') else 224
            for name,sl in (('early_0_60',slice(0,61)),('whole_0_200',slice(None))):
                present=data['presence'][keep,sl,1]>0
                pixel_box=(data['boxes'][keep,sl,1,2:]*size)[present]
                patches=data['tokens'][keep,sl,1,:,-1].sum(-1)
                rows.append({'task':task,'stream':stream,'window':name,'frames':present.numel(),
                    'object_visible_fraction':float(present.float().mean()),
                    'median_visible_bbox_pixels':pixel_box.median(0).values.tolist(),
                    'median_valid_spatial_tokens':float(patches[present].median())})
    dump(ROOT/'VISIBILITY_DIAGNOSTIC.json',{'rows':rows,'scope':'fixed validation objects; no physical success or contact labels',
        'box_pixels':'RGB-derived tight image box, width and height; camera dimensions are native',
        'bindings':bindings([ROOT/t/'data/RECEIPT.json' for t in TASKS]+[Path(__file__)])})
    print('P35_VISIBILITY_COMPLETE',flush=True)


if __name__=='__main__':main()
