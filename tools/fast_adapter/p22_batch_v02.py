"""Corrected renderer evaluation of unchanged P22 weights on fresh conditions."""
from pathlib import Path
import json,time
from tools.fast_adapter.p18_run import save_json,digest
from tools.fast_adapter import p22_batch

R=Path('evaluations/fast_adapter_p22_geometry_v02')

def main():
    p22_batch.R=R;p22_batch.verify_sources();began=time.time()
    for stage in ('dev','final'):
        if stage=='final':
            refs={s:json.loads((R/'dev'/s/'seed0/RESULT.json').read_text())['metrics'] for s in ('id','position','scale')}
            passed=refs['id']['base']['successes']>=8 and all(r['oracle']['successes']>=15 for r in refs.values())
            save_json(R/'BASE_GATE.json',{'passed':passed,'metrics':refs,'rule':'Same v01 gate; new dev conditions; unchanged base'})
            if not passed:
                save_json(R/'CAMPAIGN_DONE.json',{'status':'completed_base_qualification_failed','final_run':False,'adapter_trained':True,'seconds':time.time()-began});return
            paths=[R/'SOURCE_FREEZE.json',R/'PROTOCOL.json',R/'bank/PCA.pt',R/'models/base/model.pt']
            paths += [R/'models'/f'seed{s}'/f'{a}.pt' for s in (11,22,33) for a in ('history','hybrid')]
            save_json(R/'FINAL_FREEZE.json',{'time':time.time(),'outcomes_opened':False,'no_parameter_updates':True,
                'bindings':[{'path':str(p),'sha256':digest(p)} for p in paths]})
        for seed in (0,11,22,33):
            p22_batch.parallel([(g,f'{stage}_{st}_seed{seed}',['tools.fast_adapter.p22_eval_v02','--stage',stage,
                 '--stratum',st,'--seed',str(seed),'--dino','assets/dinov3']) for g,st in enumerate(('id','position','scale'))])
    save_json(R/'CAMPAIGN_DONE.json',{'status':'completed_pending_root_audit','final_run':True,'seconds':time.time()-began,
      'reused_training':'v01 training valid; exact same checkpoints; evaluation-only correction'})
    print('CAMPAIGN_COMPLETE',flush=True)

if __name__=='__main__':main()
