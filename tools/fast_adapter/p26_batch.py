"""Finite three-GPU campaign; teacher qualification is a mandatory gate."""
import json,time
from tools.fast_adapter import p22_batch
from tools.fast_adapter.p26_common import R,SEEDS,ARMS
from tools.fast_adapter.p18_run import save_json,digest


def seal(name,paths,**meta):
    save_json(R/name,{'time':time.time(),**meta,'bindings':[{'path':str(p),'sha256':digest(p)} for p in paths]})


def main():
    p22_batch.R=R;p22_batch.verify_sources();began=time.time()
    assert json.loads((R/'qualification/RESULT.json').read_text())['passed']
    p22_batch.parallel([(g,f'train_a_seed{s}',['tools.fast_adapter.p26_train','--seed',str(s),'--stage','a']) for g,s in enumerate(SEEDS)])
    seal('STAGE_A_FREEZE.json',[R/'models'/f'seed{s}'/'a'/f'{name}.pt' for s in SEEDS for name in ('oldlabel','relabel')],training_refresh_not_started=True)
    p22_batch.run(['tools.fast_adapter.p26_rollout','collect'],2,'training_refresh')
    p22_batch.parallel([(g,f'train_final_seed{s}',['tools.fast_adapter.p26_train','--seed',str(s),'--stage','final']) for g,s in enumerate(SEEDS)])
    p22_batch.verify_sources()
    seal('MODEL_FREEZE.json',[R/'PROTOCOL.json',R/'SOURCE_FREEZE.json',R/'training_refresh/COMPLETE.json']+
        [R/'models'/f'seed{s}'/'final'/f'{name}.pt' for s in SEEDS for name in ARMS],development_not_opened=True)
    p22_batch.parallel([(g,f'dev_{st}',['tools.fast_adapter.p26_rollout','evaluate','--stratum',st]) for g,st in enumerate(('id','position','scale'))])
    p22_batch.verify_sources()
    save_json(R/'CAMPAIGN_DONE.json',{'status':'small_development_complete_pending_review','seconds':time.time()-began,
        'queries':540,'development_supports':36,'training_queries':64,'training_supports':32,'formal_final':False})
    print('CAMPAIGN_COMPLETE',flush=True)

if __name__=='__main__':main()
