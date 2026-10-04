"""Load the prospectively frozen r2 adapter class without changing sealed r1 code."""
import argparse,runpy,sys
import torch
from prove.fast_adapter.object_adapter_v2 import ObjectAdapterV2

def configure():
    import tools.fast_adapter.p29_common as common
    old=common.load_adapter
    def load(task,seed,variant,round_name='r1'):
        if round_name!='r2':return old(task,seed,variant,round_name)
        blob=torch.load(common.task_root(task)/'models/r2'/f'seed{seed}'/f'{variant}.pt',map_location='cpu',weights_only=True)
        net=ObjectAdapterV2(history=variant=='history',feedback_mode='robot' if variant=='robot' else 'object').cuda();net.load_state_dict(blob['state']);return net.eval().requires_grad_(False),blob
    common.load_adapter=load
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('module');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args();configure();sys.argv=[a.module,*a.args];runpy.run_module(a.module,run_name='__main__')
