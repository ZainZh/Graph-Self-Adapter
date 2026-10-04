"""Paired environment-cluster inference conditional on two fixed checkpoints."""
import numpy as np


def paired_clusters(a,b,seed=27027,resamples=20000,family=9):
    a,b=np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    if a.ndim==1:a,b=a[None],b[None]
    assert a.ndim==2 and a.shape[1]>=2
    diff=a-b;cluster=diff.mean(axis=0)
    rng=np.random.default_rng(seed)
    indices=rng.integers(len(cluster),size=(resamples,len(cluster)))
    boot=cluster[indices].mean(axis=1)*100
    tail=.05/(2*family)
    ci=np.quantile(boot,[tail,1-tail]);per_seed=diff.mean(axis=1)*100
    return {'delta_pp':float(cluster.mean()*100),'paired_ci95':np.quantile(boot,[.025,.975]).tolist(),
        'multiplicity_adjusted_ci':ci.tolist(),'adjusted_interval_level':1-.05/family,'family_size':family,
        'seed_deltas_pp':per_seed.tolist(),'positive_model_condition_differences':int((diff>0).sum()),
        'negative_model_condition_differences':int((diff<0).sum()),'independent_clusters':len(cluster),
        'conditional_positive_gate':bool(ci[0]>0 and np.all(per_seed>0)),
        'at_least_5pp':bool(cluster.mean()>=.05),
        'inference_scope':'Paired percentile bootstrap of independent condition clusters, fixed model seeds averaged inside each cluster; approximate, not exact finite-sample familywise coverage.'}
