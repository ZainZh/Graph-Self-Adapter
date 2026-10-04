"""Result-free P67 scope only; all runtime mathematics belongs to frozen P66."""
from prove.fast_adapter.p66_cup import ARMS

SPLITS=('id','size','position')
SEED_START={'id':670100,'size':670200,'position':670300}
CONDITIONS_PER_SPLIT=64
CONDITION_COUNT=192
CELL_COUNT=768
HORIZON=240
FREEZE_SCHEMA='prove.p67-cup-confirmation-freeze.v1'
QUEUE=dict(first_index_alone=0,max_workers_after_first=2,cpu_threads=1,automatic_retries=0,
    on_program_error='stop_new_submissions_and_await_running')


def cells():
    return [dict(split=split,seed=seed,arm=arm) for split in SPLITS
        for seed in range(SEED_START[split],SEED_START[split]+CONDITIONS_PER_SPLIT) for arm in ARMS]


def selected_videos():
    return [dict(split=split,seed=SEED_START[split]) for split in SPLITS]
