"""What does the distance gate (attn_dist_bias) learn? On training interpolants of the MP-20 test set, per (t, N bin):
mean / sd of the gate g(h_i) (> 0 = sharpen toward the nearest neighbour, since psi <= 0), the
std over non-self edges of the bias g*psi vs of the content logit w.e, and the attention mass on the
first shell (d <= 1.25 d_min, current coords) with and without the bias.

  python probe_gate.py -m <ckpt dir> --out gate.json --max_batches 8
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import torch
from torch_scatter import scatter, scatter_softmax

sys.path.insert(0, 'scripts')
from eval_utils import load_model  # noqa: E402
from diffcsp.common.data_utils import lattice_polar_build_torch  # noqa: E402

BINS = (4, 8, 12, 16, 20)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--model_path', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max_batches', type=int, default=8)
    p.add_argument('--tgrid', default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    a = p.parse_args()
    model, loader, _ = load_model(Path(a.model_path), load_data=True)
    model.to('cuda').eval()
    dec = model.decoder
    assert dec.attn_dist_bias
    layers = [m for n, m in dec.named_modules() if getattr(m, 'attn_bias_gate', None) is not None]
    st = {}
    for li, L in enumerate(layers):
        orig = L.node_model

        def nm(node_features, edge_features, edge_index, attn_psi=None, _o=orig, _L=L, _i=li):
            st[_i] = dict(g=_L.attn_bias_gate(node_features)[:, 0], z=_L.attn_score_proj(edge_features)[:, 0],
                          psi=attn_psi, ei=edge_index)
            return _o(node_features, edge_features, edge_index, attn_psi)
        L.node_model = nm
    A = defaultdict(lambda: defaultdict(float))
    g0 = torch.Generator(device='cuda').manual_seed(0)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= a.max_batches:
                break
            batch = batch.to('cuda')
            B, fT, lT = batch.num_graphs, batch.frac_coords, batch.lattice_polar
            f0 = torch.rand(fT.shape, generator=g0, device='cuda')
            l0 = torch.randn((B, 6), generator=g0, device='cuda') * model.lattice_polar_sigma
            tf, tl = (fT - f0 - 0.5) % 1 - 0.5, lT - l0
            bn = torch.bucketize(batch.num_atoms[batch.batch], torch.tensor(BINS[:-1], device='cuda'))
            for t in [float(x) for x in a.tgrid.split(',')]:
                in_f, in_l = f0 + t * tf, l0 + t * tl
                dec(t=model.time_embedding(torch.full((B,), t, device='cuda')), atom_types=batch.atom_types,
                    frac_coords=in_f, lattices_rep=in_l, num_atoms=batch.num_atoms, node2graph=batch.batch,
                    lattices_mat=lattice_polar_build_torch(in_l), cemb=None, guide_indicator=None)
                for li, s in st.items():
                    src, dst = s['ei']; n = s['g'].shape[0]; ns = src != dst
                    bias = s['g'][src] * s['psi']
                    shell = (s['psi'] >= -torch.log(torch.tensor(1.25, device='cuda'))) & ns
                    w0, w1 = scatter_softmax(s['z'], src), scatter_softmax(s['z'] + bias, src)
                    m0 = scatter(w0 * shell, src, dim=0, dim_size=n, reduce='sum')
                    m1 = scatter(w1 * shell, src, dim=0, dim_size=n, reduce='sum')
                    zc = s['z'] - scatter(s['z'] * ns, src, dim=0, dim_size=n, reduce='sum')[src] / \
                        scatter(ns.float(), src, dim=0, dim_size=n, reduce='sum')[src].clamp_min(1)
                    bc = bias - scatter(bias * ns, src, dim=0, dim_size=n, reduce='sum')[src] / \
                        scatter(ns.float(), src, dim=0, dim_size=n, reduce='sum')[src].clamp_min(1)
                    for k in range(len(BINS)):
                        m = bn == k; me = (bn[src] == k) & ns
                        if not m.any():
                            continue
                        D = A[f'L{li}|{t}|{k}']
                        D['n'] += m.sum().item(); D['ne'] += me.sum().item()
                        D['g'] += s['g'][m].sum().item(); D['g2'] += (s['g'][m] ** 2).sum().item()
                        D['zc2'] += (zc[me] ** 2).sum().item(); D['bc2'] += (bc[me] ** 2).sum().item()
                        D['shell0'] += m0[m].sum().item(); D['shell1'] += m1[m].sum().item()
            print('batch', bi, flush=True)
    out = {}
    for key, D in A.items():
        n, ne = max(D['n'], 1), max(D['ne'], 1)
        out[key] = dict(n=D['n'], g=D['g'] / n, g_sd=max(D['g2'] / n - (D['g'] / n) ** 2, 0) ** .5,
                        zc_sd=(D['zc2'] / ne) ** .5, bias_sd=(D['bc2'] / ne) ** .5,
                        shell_mass=D['shell0'] / n, shell_mass_bias=D['shell1'] / n)
    Path(a.out).write_text(json.dumps(dict(model=a.model_path, bins=BINS, stats=out)))
    print('DONE', a.out, flush=True)


if __name__ == '__main__':
    main()
