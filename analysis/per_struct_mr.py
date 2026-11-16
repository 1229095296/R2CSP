#!/usr/bin/env python3
"""Per-structure match/RMSE from an existing eval_diff_<label>.pt, using the exact
compute_metrics.py code path (same validity + StructureMatcher), so the overall MR
reproduces eval_metrics_<label>.json. Writes per_struct_<label>.csv in the eval dir.
Run from the CrystalFlow root: python per_struct_mr.py --eval_dir D --label L
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from p_tqdm import p_map

sys.path.insert(0, 'scripts')
import compute_metrics as cm  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval_dir', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--gt_file', default='data/mp_20/test.csv')
    p.add_argument('--njobs', type=int, default=16)
    a = p.parse_args()
    d = Path(a.eval_dir)

    path = cm.get_file_paths(str(d), 'diff', a.label)
    arrs, _ = cm.get_crystal_array_list(path, batch_idx=0)
    preds = p_map(lambda x: cm.Crystal(x), arrs, num_cpus=a.njobs)
    csv = pd.read_csv(a.gt_file)
    gts = p_map(cm.get_gt_crys_ori, csv['cif'], num_cpus=a.njobs)

    ev = cm.RecEval(preds, gts, njobs=a.njobs)
    m = ev.get_match_rate_and_rms()
    rms = np.array([np.nan if r is None else r for r in ev.rms_dists], dtype=float)

    out = pd.DataFrame({
        'material_id': csv['material_id'],
        'n_atoms': [len(g.atom_types) for g in gts],
        'spacegroup': csv['spacegroup.number'] if 'spacegroup.number' in csv else -1,  # absent in mpts_52
        'pred_valid': [c.valid for c in preds],
        'match': ~np.isnan(rms),
        'rms': rms,
    })
    out.to_csv(d / f'per_struct_{a.label}.csv', index=False)

    ref = d / f'eval_metrics_{a.label}.json'
    if ref.exists():
        r = json.loads(ref.read_text())
        print('stored  ', r)
    print('recomputed', {k: float(v) for k, v in m.items()})


if __name__ == '__main__':
    main()
