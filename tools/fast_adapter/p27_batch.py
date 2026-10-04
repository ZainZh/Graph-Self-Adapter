"""One fixed-checkpoint attribution batch, no adaptive stopping or extension."""
import time
from tools.fast_adapter import p22_batch
from tools.fast_adapter.p27_common import R,STRATA
from tools.fast_adapter.p18_run import save_json


def main():
    p22_batch.R=R;p22_batch.verify_sources();began=time.time()
    p22_batch.parallel([(g,f'eval_{st}',['tools.fast_adapter.p27_eval','--stratum',st]) for g,st in enumerate(STRATA)])
    p22_batch.verify_sources()
    save_json(R/'CAMPAIGN_DONE.json',{'status':'fixed_checkpoint_attribution_complete_pending_review','seconds':time.time()-began,
        'query_episodes':6528,'independent_target_conditions':384,'support_trajectories':1152,'training_updates':0,'physical_actions':350208})
    print('CAMPAIGN_COMPLETE',flush=True)

if __name__=='__main__':main()
