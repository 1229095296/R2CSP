"""Fit the dose--response law dMR = a * (|dm|/|m|)^2 + c over the inference-time interventions.

x: relative change of the last-layer aggregate, from eval_fmloss.py, averaged over t in
   {0.3, 0.4, 0.5, 0.6} and all size bins, weighted by the number of atoms.
y: paired match-rate change in points against the unmodified model, from paired_stats.py with a run
   named ref. With several paired files, each mode is taken from the first file that contains it,
   against the ref of that file.
  python fit_dose.py --fm fm_s123.json --paired dose_s123_paired.json --out dose_fit_s123.json
"""
import argparse, json

import numpy as np

MODES = ['uniform', 'selfkeep', 'selfflat', 'normprojN', 'residN', 'shuffle',
         'bias:1', 'bias:2', 'bias:4', 'bias:-2', 'shelloracle', 'oracle']
T_WINDOW = [0.3, 0.4, 0.5, 0.6]


def rel_move(stats, mode):
    num = den = 0.0
    for key, s in stats.items():
        m, t, _ = key.split('|')
        if m == mode and float(t) in T_WINDOW:
            num += s['dm'] * s['n']
            den += s['n']
    return num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fm', required=True)
    p.add_argument('--paired', required=True, nargs='+')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    fm = json.load(open(a.fm))['stats']
    mrs = [json.load(open(f))['mr'] for f in a.paired]
    x = np.array([rel_move(fm, m) for m in MODES])
    y = []
    for m in MODES:
        mr = next(r for r in mrs if m.replace(':-', 'm').replace(':', '') in r)
        y.append(100 * (mr[m.replace(':-', 'm').replace(':', '')] - mr['ref']))
    y = np.array(y)
    A = np.stack([x ** 2, np.ones_like(x)], 1)
    (slope, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ [slope, c]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    out = dict(x=x.tolist(), y=y.tolist(), a=float(slope), c=float(c), r2=float(r2), modes=MODES)
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f"a={slope:.3f} c={c:.3f} R2={r2:.3f} r(x,y)={np.corrcoef(x, y)[0, 1]:.3f}")


if __name__ == '__main__':
    main()
