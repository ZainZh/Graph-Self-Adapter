"""Predeclared success and paired uncertainty calculations for P42-A."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from tools.fast_adapter.p41_run import js, sha
from tools.fast_adapter.p42_batch import ROOT, verify


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [float(max(0, center-half)), float(min(1, center+half))]


def paired(current, reference, seed):
    d = np.asarray(current, dtype=int)-np.asarray(reference, dtype=int)
    n = len(d)
    wins, losses = int((d == 1).sum()), int((d == -1).sum())
    # Resample whole paired geometry conditions, preserving five-arm pairing.
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, n, size=(100000, n))].mean(axis=1)
    return dict(n=n, wins=wins, losses=losses, ties=n-wins-losses,
                delta=float(d.mean()),
                paired_bootstrap_95=np.quantile(boot, [.025, .975]).tolist(),
                paired_bootstrap_bonferroni_98_75=np.quantile(boot, [.00625, .99375]).tolist(),
                exact_mcnemar_one_sided_p=(float(binomtest(wins, wins+losses, .5,
                                                         alternative='greater').pvalue)
                                          if wins+losses else 1.0))


def main():
    freeze, protocol = verify()
    rows = []
    exceptions = []
    for task in protocol['tasks']:
        done = json.loads((ROOT / 'independent_v01' / f'{task}_DONE.json').read_text())
        assert len(done['rows']) == 640
        for split in protocol['splits']:
            for seed in protocol['seeds']:
                for arm in protocol['arms']:
                    path = ROOT / 'independent_v01' / task / f'{split}_{seed}_{arm}'
                    if (path / 'RESULT.json').exists():
                        r = json.loads((path / 'RESULT.json').read_text())
                        rows.append(dict(r, cohort_scope='independent_frozen_prototype_replication',
                                         execution_status='complete', source=str(path/'RESULT.json')))
                    else:
                        e = json.loads((path/'EXCEPTION.json').read_text())
                        r = dict(task=task, split=split, seed=seed, arm=arm,
                                 success=False, execution_status='execution_exception',
                                 source=str(path/'EXCEPTION.json'), exception=e)
                        rows.append(r)
                        exceptions.append(r)
    cells, pairs = [], []
    for task in protocol['tasks']:
        for split in protocol['splits']:
            for arm in protocol['arms']:
                group = [r for r in rows if (r['task'], r['split'], r['arm']) == (task, split, arm)]
                k = sum(r['success'] for r in group)
                cells.append(dict(task=task, split=split, arm=arm, n=len(group),
                                  successes=k, rate=k/len(group), wilson_95=wilson(k, len(group)),
                                  exceptions=sum(r['execution_status'] != 'complete' for r in group)))
            current = {r['seed']:r for r in rows if (r['task'],r['split'],r['arm']) == (task,split,'gradient')}
            for arm in ['fixed', 'single', 'prefix', 'multiview']:
                ref = {r['seed']:r for r in rows if (r['task'],r['split'],r['arm']) == (task,split,arm)}
                value = paired([current[s]['success'] for s in protocol['seeds']],
                               [ref[s]['success'] for s in protocol['seeds']],
                               protocol['statistics']['bootstrap_seed']+len(pairs))
                primary = split == 'geometry' and arm in ['fixed', 'single']
                pairs.append(dict(task=task, split=split, reference=arm, primary=primary, **value))
    primary = sorted([p for p in pairs if p['primary']], key=lambda x:x['exact_mcnemar_one_sided_p'])
    running = 0.
    for i, item in enumerate(primary):
        running = max(running, (4-i)*item['exact_mcnemar_one_sided_p'])
        item['holm_adjusted_p'] = min(1., running)
        item['positive_primary_comparison'] = bool(item['delta'] > 0 and item['holm_adjusted_p'] < .05)
    js(ROOT/'summary/RESULTS.json', dict(scope='independent frozen analytic geometry prototype; not DINO or meta-learning',
        protocol_sha256=sha(ROOT/'PROTOCOL.json'), freeze_sha256=sha(ROOT/'FREEZE.json'),
        rows=rows, cells=cells, paired=pairs, exceptions=exceptions,
        all_four_primary_positive=all(p['positive_primary_comparison'] for p in primary),
        uncertainty='Percentile paired bootstrap is discrete and may degenerate when all pair differences are equal; Wilson intervals describe arm rates. Exact paired tests control the four primary comparisons with Holm.'))
    print(json.dumps(dict(cells=cells, primary=primary, exceptions=len(exceptions)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
