"""Reuse the sealed P20 train/eval implementation with explicit P21 dependencies."""
import argparse,json
from pathlib import Path
from prove.fast_adapter.visual_alignment import DEFAULT_MODEL
from prove.fast_adapter.lila_frontend import ThreeFrontends
R=Path('evaluations/fast_adapter_p21_frontends_v01')

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','eval'])
    p.add_argument('--frontend',required=True,choices=['last','middle17','lila'])
    p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',default='dev',choices=['dev','final'])
    p.add_argument('--stratum',default='train',choices=['train','ood']);p.add_argument('--dino',default=DEFAULT_MODEL)
    a=p.parse_args();root=R/'frontends'/a.frontend
    selection=json.loads((R/'diagnostic/RESULT.json').read_text())
    assert selection['control_qualified'] and a.frontend in selection['control_frontends']
    if a.mode=='train':
        from tools.fast_adapter import p20_train as runner
        runner.R=root;runner.main(a.seed,'cuda')
    else:
        if a.stage=='final':assert (R/'FINAL_FREEZE.json').exists(), 'final must be frozen before rollout'
        from tools.fast_adapter import p20_eval as runner
        protocol=json.loads((R/'PROTOCOL.json').read_text())
        def setting(stage,stratum):
            q=protocol[stage]
            if stage=='dev':return q['domains'],q['domain_seed'],q['start_seed'],'train'
            return q['domains_per_stratum'],q['domain_seeds'][stratum],q['start_seeds'][stratum],stratum
        def encoder(model_dir=DEFAULT_MODEL,device='cuda',microbatch=16):
            return ThreeFrontends(model_dir,device,microbatch,selected=a.frontend)
        runner.R=root;runner.setting=setting;runner.FrozenDinoROI=encoder
        runner.main(a.stage,a.stratum,a.seed,a.dino)
if __name__=='__main__':main()
