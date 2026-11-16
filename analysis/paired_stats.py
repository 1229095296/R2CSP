"""Paired per-structure statistics over per_struct_<label>.csv files (same test set, same order).

  * pairwise exact McNemar (two-sided binomial on discordant pairs b, c) + MR difference
    with a paired bootstrap 95% CI;
  * MR by atom count N (bins) per run, and paired difference vs a reference run;
  * RMSE on the jointly matched subset (paired Wilcoxon).
Usage: python paired_stats.py --out OUT.json name=path.csv [name=path.csv ...] [--ref name]
"""
import argparse, itertools, json

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

BINS = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 20)]


def boot_ci(d, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar(a, b):
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
    p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    d = a.astype(float) - b.astype(float)
    return dict(mr_a=float(a.mean()), mr_b=float(b.mean()), diff=float(d.mean()), ci95=boot_ci(d),
                a_only=n10, b_only=n01, p=float(p))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--ref', default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--bins', default=None, help='e.g. 1-10,11-20,21-30,31-40,41-52 (MPTS52)')
    a = p.parse_args()
    dfs = {}
    for r in a.runs:
        k, path = r.split('=', 1)
        dfs[k] = pd.read_csv(path)
    names = list(dfs)
    base = dfs[names[0]]
    for k, df in dfs.items():
        assert (df['material_id'].values == base['material_id'].values).all(), f'order mismatch {k}'
    N = base['n_atoms'].values
    M = {k: df['match'].values.astype(bool) for k, df in dfs.items()}
    rms = {k: df['rms'].values for k, df in dfs.items()}

    res = dict(n=len(base), runs=names, mr={k: float(v.mean()) for k, v in M.items()}, pairs={}, byN={})
    for x, y in itertools.combinations(names, 2):
        r = mcnemar(M[x], M[y])
        both = M[x] & M[y]
        if both.sum() > 10:
            r['rms_joint'] = [float(np.nanmean(rms[x][both])), float(np.nanmean(rms[y][both]))]
            r['rms_wilcoxon_p'] = float(wilcoxon(rms[x][both], rms[y][both]).pvalue)
        res['pairs'][f'{x}|{y}'] = r
    bins = [tuple(int(x) for x in s.split('-')) for s in a.bins.split(',')] if a.bins else BINS
    for lo, hi in bins:
        m = (N >= lo) & (N <= hi)
        row = dict(count=int(m.sum()), mr={k: float(v[m].mean()) for k, v in M.items()})
        if a.ref:
            row['vs_ref'] = {k: mcnemar(M[k][m], M[a.ref][m]) for k in names if k != a.ref}
        res['byN'][f'{lo}-{hi}'] = row
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)

    print(f"n={res['n']}")
    print('MR: ' + '  '.join(f'{k} {100 * v:.2f}' for k, v in res['mr'].items()))
    print(f"{'pair':45s} {'diff':>7s} {'95% CI':>17s} {'a_only':>6s} {'b_only':>6s} {'p':>9s}")
    for k, r in res['pairs'].items():
        print(f"{k:45s} {100 * r['diff']:+7.2f} [{100 * r['ci95'][0]:+6.2f},{100 * r['ci95'][1]:+6.2f}] "
              f"{r['a_only']:6d} {r['b_only']:6d} {r['p']:9.2e}")
    print('MR by N: ' + '  '.join(names))
    for b, row in res['byN'].items():
        print(f"  N {b:6s} (n={row['count']:4d}) " + '  '.join(f"{100 * row['mr'][k]:5.1f}" for k in names))


if __name__ == '__main__':
    main()
