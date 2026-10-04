"""P40 explicit missing-role semantics, without evaluator truth fallback."""
import numpy as np

def normalize_observation(task,obs):
    out={k:np.array(v,copy=True) for k,v in obs.items()}
    if task=='insert':
        # The peg is rigidly attached to the actuated stage. Stage x/z are
        # declared proprioception, available even when the peg enters the hole.
        # No synthetic visual feature/mask is filled; absent extent is zero.
        if not out['visible'][1]:
            out['positions'][1]=out['proprio'][:2];out['sizes'][1]=0.
        if not out['visible'][0]:
            out['positions'][0]=out['proprio'][:2];out['sizes'][0]=0.
        valid=bool(out['visible'][2] and np.isfinite(out['positions']).all())
    else:
        valid=bool(out['visible'].all() and np.isfinite(out['positions']).all())
    return out,valid
