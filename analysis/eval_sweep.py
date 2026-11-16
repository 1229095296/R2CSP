"""Sampler-setting sweep on a fixed VALIDATION subset (model selection without touching test).

Same sampler as scripts/evaluate.py (Euler ODE, anneal_coords with slope s, no lattice anneal),
but on the first --max_n structures of the val split, seeded so every setting and every model
sees the same noise. Writes <eval_dir>/eval_diff_<label>.pt in evaluate.py's format, so
per_struct_mr.py works with --gt_file = the matching head-of-val csv (see --write_gt).

  python eval_sweep.py -m <eval_dir> --split val --max_n 2000 -N 100 --slope 5 --label v_s5_N100
"""
import argparse, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import hydra
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, 'scripts')
from eval_utils import load_model  # noqa: E402
from evaluate import diffusion  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--model_path', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--max_n', type=int, default=2000)
    p.add_argument('-N', '--steps', type=int, default=100)
    p.add_argument('--slope', type=float, default=5.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--write_gt', default='', help='also write the head-of-split gt csv here')
    p.add_argument('--gt_src', default='', help='full split csv (for --write_gt)')
    p.add_argument('--zero_gate', action='store_true', help='zero the distance gates (attn_dist_bias)')
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    mp = Path(a.model_path)
    if a.split == 'test':
        model, loader, cfg = load_model(mp, load_data=True, testing=True)
        ds = loader.dataset
    else:
        # load_model(testing=False) also builds the train set (and has a typo on that path), so
        # build only the val datasets, with the scalers the model was trained with
        model, _, cfg = load_model(mp, load_data=False)
        dm = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False, scaler_path=mp)
        vals = [hydra.utils.instantiate(c) for c in dm.datasets.val]
        for v in vals:
            v.lattice_scaler, v.scaler, v.scalers = dm.lattice_scaler, dm.scaler, dm.scalers
        ds = ConcatDataset(vals)
    n = min(a.max_n, len(ds)) if a.max_n > 0 else len(ds)
    loader = DataLoader(Subset(ds, range(n)), batch_size=cfg.data.datamodule.batch_size.test)
    model.to('cuda').eval()
    if a.zero_gate:     # distance-gate knockout: g(h_i) = 0, i.e. plain softmax attention
        gates = [m.attn_bias_gate for m in model.modules() if getattr(m, 'attn_bias_gate', None) is not None]
        assert gates
        for g in gates:
            torch.nn.init.zeros_(g.weight); torch.nn.init.zeros_(g.bias)
    if a.write_gt:
        pd.read_csv(a.gt_src).head(n).to_csv(a.write_gt, index=False)
    print(f'{a.split} n={n} steps={a.steps} slope={a.slope}', flush=True)

    t0 = time.time()
    fc, at, lat, lengths, angles, na, inp = diffusion(
        loader, model, num_evals=1, step_lr=1 / a.steps, N=a.steps,
        anneal_lattice=False, anneal_coords=a.slope > 0, anneal_type=False,
        anneal_slope=a.slope, anneal_offset=0.0, guide_factor=None)
    torch.save({'eval_setting': a, 'input_data_batch': inp, 'frac_coords': fc, 'num_atoms': na,
                'atom_types': at, 'lattices': lat, 'lengths': lengths, 'angles': angles,
                'time': time.time() - t0}, mp / f'eval_diff_{a.label}.pt')
    print('DONE', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
