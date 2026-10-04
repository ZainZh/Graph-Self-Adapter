"""Training-only implementation qualification; discard models afterwards."""
import subprocess
import sys
import torch
from tools.fast_adapter.p32_common import *
from prove.fast_adapter.learned_interaction_loss import LossState


def main():
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    assert not (ROOT/'QUALIFICATION.json').exists()
    test=subprocess.run(['data/runtime/libero-pi05-venv-clean/bin/python','-m','pytest','-q','tests/test_learned_interaction_loss.py'],
                        capture_output=True,text=True)
    (ROOT/'QUALIFICATION_TESTS.log').write_text(test.stdout+test.stderr)
    assert test.returncode==0,test.stdout+test.stderr
    for b in json.loads((P31/'VISUAL_FREEZE.json').read_text())['files']:
        assert digest(b['path'])==b['sha256'],b['path']
    assert json.loads((P31/'OBSERVER_QUALIFICATION.json').read_text())['passed']
    bank={k:v[[0,1,64,65]].cuda() for k,v in torch.load(P31/'data/own.pt',weights_only=True).items()}
    base=base_model(TASK);obs=observer();nets,counts=make_pair(3232)
    prev=torch.cat((torch.zeros_like(bank['actions'][:,:1]),bank['actions'][:,:-1]),1)
    with torch.no_grad():ba=base(bank['visual'][:,:-1],bank['proprio'][:,:-1],prev);gates=obs.active(bank['visual'][:,:-1],bank['proprio'][:,:-1],prev)
    values={}
    for family,net in nets.items():
        opt=torch.optim.Adam(net.parameters(),lr=3e-4);s=LossState(net,4,meta=True);grad_sum=0.
        for start in range(0,200,20):
            loss=0.
            for t in range(start,start+20):
                v,p=bank['visual'][:,t],bank['proprio'][:,t]
                a=s.action(v,p,prev[:,t],ba[:,t],gates[:,t])
                loss=loss+((a-bank['teacher'][:,t])/.5).square().mean()/20
                s.observe(v,p,prev[:,t],bank['actions'][:,t],bank['visual'][:,t+1],bank['proprio'][:,t+1],gates[:,t])
            opt.zero_grad(set_to_none=True);loss.backward()
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in net.parameters())
            if family=='learned':grad_sum+=sum(float(p.grad.abs().sum()) for p in net.energy.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();s.detach_segment()
        assert bool(torch.isfinite(s.z).all()) and int(s.counts.sum())>0
        if family=='learned':assert grad_sum>1e-10,'Energy has no action meta-gradient'
        values[family]={'update_counts':s.counts.tolist(),'latent_rms':float(s.z.detach().square().mean().sqrt()),'energy_action_meta_gradient_l1_sum':grad_sum if family=='learned' else None}
    dump(ROOT/'QUALIFICATION.json',{'passed':True,'training_only_discarded_preflight':True,
         'tests':test.stdout.strip(),'parameter_counts':counts,'values':values,
         'visual_hashes_match_parent':True,'bindings':bindings(source_paths()+[Path(__file__),Path('tests/test_learned_interaction_loss.py')])})
    print('QUALIFICATION_PASS',counts,values,flush=True)


if __name__=='__main__':main()
