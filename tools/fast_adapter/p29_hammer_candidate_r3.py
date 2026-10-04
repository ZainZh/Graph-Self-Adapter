"""Explicit GRU base and tool-r3 adapter factory, configured before pipeline imports."""
import argparse,runpy,sys
import torch
from prove.fast_adapter.object_tool_adapter_v2 import ObjectToolAdapterV2

def configure():
    from tools.fast_adapter.p29_candidate_v04 import configure as base_configure
    base_configure('evaluations/fast_adapter_p29_object_transfer_v01/tasks/hammer-block/candidates/base_v03/CANDIDATE_V2.json')
    import tools.fast_adapter.p29_common as common
    old=common.load_adapter
    def load(task,seed,variant,round_name='r1'):
        if task!='hammer-block' or round_name!='r3':return old(task,seed,variant,round_name)
        blob=torch.load(common.task_root(task)/'models/r3'/f'seed{seed}'/f'{variant}.pt',map_location='cpu',weights_only=True)
        net=ObjectToolAdapterV2(history=variant=='history',feedback_mode='robot' if variant=='robot' else 'object').cuda();net.load_state_dict(blob['state']);return net.eval().requires_grad_(False),blob
    common.load_adapter=load
    import tools.fast_adapter.p29_data as data
    old_stage=data.load_stage
    def corrected_stage(task,stage,role=None):
        if task=='hammer-block' and stage=='meta' and role=='base':stage='meta_relabel_r3'
        return old_stage(task,stage,role)
    data.load_stage=corrected_stage
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('module');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args();configure();sys.argv=[a.module,*a.args];runpy.run_module(a.module,run_name='__main__')
