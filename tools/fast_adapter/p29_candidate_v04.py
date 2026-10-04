"""Explicit candidate binding for reused P29 functions; no change to sealed v1 code.

A candidate JSON chooses task-local base checkpoint, meta-training bank and adapter
round. Dependency injection happens before importing the selected pipeline module.
Every referenced input is hash-checked. It never changes physical task/evaluation.
"""
import argparse,json,runpy,sys
from pathlib import Path
import torch
from prove.fast_adapter.object_interaction_adapter import ObjectSkill

def configure(path):
    import tools.fast_adapter.p29_common as common
    cfg=json.loads(Path(path).read_text())
    for b in cfg['bindings']:assert common.digest(b['path'])==b['sha256'],b['path']
    old_base=common.load_base
    def selected_base(task,round_name='base_final'):
        if task!=cfg['task'] or round_name!='base_final':return old_base(task,round_name)

        if cfg.get('base_kind')=='recurrent':
            from prove.fast_adapter.object_sequence_skill import ObjectSequenceSkill
            net=ObjectSequenceSkill().cuda()
        else:net=ObjectSkill().cuda()
        net.load_state_dict(torch.load(cfg['base_checkpoint'],map_location='cpu',weights_only=True)['state'])
        return net.eval().requires_grad_(False)
    common.load_base=selected_base
    import tools.fast_adapter.p29_data as data
    old_collect=data.collect_block
    def selected_collect(*args,**kwargs):
        base=kwargs.get('base',args[6] if len(args)>6 else None)
        if base is not None and hasattr(base,'reset_rollout'):base.reset_rollout()
        return old_collect(*args,**kwargs)
    data.collect_block=selected_collect
    old_load=data.load_stage
    def selected_stage(task,stage,role=None):
        if task==cfg['task'] and stage=='meta':stage=cfg['meta_stage']
        return old_load(task,stage,role)
    data.load_stage=selected_stage
    return cfg
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('module');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args();configure(a.candidate)
    if a.module=='tools.fast_adapter.p29_data':
        # Invoke the configured imported module, avoiding runpy creating a second
        # unconfigured collector whose recurrent state would cross reset boundaries.
        import tools.fast_adapter.p29_data as data
        q=argparse.ArgumentParser();q.add_argument('mode',choices=['collect','merge']);q.add_argument('--task',required=True);q.add_argument('--stage');q.add_argument('--dino',default='assets/dinov3');q.add_argument('--count',type=int);x=q.parse_args(a.args)
        assert data.collect.__globals__['collect_block'] is data.collect_block
        if x.mode=='collect':data.collect(x.task,x.stage,x.dino,x.count)
        else:data.merge(x.task)
    else:
        sys.argv=[a.module,*a.args];runpy.run_module(a.module,run_name='__main__')
