"""Flow-matching loss and relative change of the aggregate under the inference-time interventions.

Training-style interpolants on the TEST set (so u is known), fixed t grid, noise shared by every mode
(paired). Interventions are those of eval_intervene_shell.py at layer 5, applied at every t. Per
(mode, t, N bin): coord/lattice loss, dl = ||dv||^2 + cross with cross = 2 dv.(v_ref - u) (a local
minimum predicts cross ~ 0), ||dm||/||m||, D_self = (dw_ii)^2 ||e_ii - mean_{j!=i} e_ij||^2, and
KL / JS of the non-self allocation vs the learned one. --grad adds, on the ref pass, with
g_ij = dl/dz_ij = w_ij (q_ij - sum_k w_ik q_ik), q_ij = (dl/dm_i) . e_ij:
  dbeta = sum_j g_ij zc_ij per node (dl/dbeta at beta=1: < 0 means this bin wants sharper logits)
  corr(-g, zc), corr(-g, resid logit), corr(-g, 1[shell]) over non-self edges, centred per node.
Stationarity of the shared w only forces the POOLED w-gradient to 0, so per-bin signs are informative.

  python eval_fmloss.py -m <eval_dir> --out fm_s42.json --grad
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path

import torch
from torch_scatter import scatter

sys.path.insert(0, 'scripts')
sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import load_model  # noqa: E402
from diffcsp.common.data_utils import lattice_polar_build_torch  # noqa: E402
from diffcsp.pl_modules import r2csp as R  # noqa: E402
import eval_intervene_shell as S  # noqa: E402

MODES = ['ref', 'uniform', 'selfkeep', 'selfflat', 'normprojN', 'residN', 'shuffle', 'bias:1', 'bias:2',
         'bias:4', 'bias:-2', 'shelloracle', 'oracle', 'beta:0.5', 'beta:2']
BINS = ((1, 4), (5, 8), (9, 12), (13, 16), (17, 20), (21, 999))


class Scaled(torch.nn.Module):
    def __init__(self, lin, beta):
        super().__init__(); self.lin, self.beta = lin, beta

    def forward(self, x):
        return self.lin(x) * self.beta


def nbin(n):
    return torch.bucketize(n, torch.tensor([b[1] for b in BINS[:-1]], device=n.device), right=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--model_path', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tgrid', default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    p.add_argument('--modes', default=','.join(MODES))
    p.add_argument('--max_batches', type=int, default=0)
    p.add_argument('--grad', action='store_true')
    a = p.parse_args()
    R.R2CSPLayer.node_model = S.node_model
    model, loader, _ = load_model(Path(a.model_path), load_data=True)
    model.to('cuda').eval().requires_grad_(True)
    torch.set_grad_enabled(False)     # re-enabled only for the --grad ref pass
    assert model.lattice_polar and not model.pred_type
    dec = model.decoder
    for i in range(dec.num_layers):
        dec._modules['csp_layer_%d' % i]._idx = i
    L5 = dec._modules['csp_layer_5']
    proj0 = L5.attn_score_proj
    S.CFG.update(t_lo=0.0, t_hi=1.01, layers={5})
    S.GEN['g'] = torch.Generator(device='cuda').manual_seed(a.seed + 1000)
    tg = [float(x) for x in a.tgrid.split(',')]
    modes = a.modes.split(',')
    A = defaultdict(lambda: defaultdict(float))
    for bi, batch in enumerate(loader):
        if a.max_batches and bi >= a.max_batches:
            break
        batch = batch.to('cuda')
        g = torch.Generator(device='cuda').manual_seed(a.seed * 100003 + bi)
        B, fT, lT = batch.num_graphs, batch.frac_coords, batch.lattice_polar
        f0 = torch.rand(fT.shape, generator=g, device='cuda')
        l0 = torch.randn((B, 6), generator=g, device='cuda') * model.lattice_polar_sigma
        tar_f, tar_l = (fT - f0 - 0.5) % 1 - 0.5, lT - l0
        nn_ = batch.num_atoms[batch.batch]; bn, bg = nbin(nn_), nbin(batch.num_atoms)
        S.STATE['shell'] = S.compute_shell(model, batch)
        for t in tg:
            in_f = f0 + t * tar_f
            in_l = l0 + t * tar_l
            temb = model.time_embedding(torch.full((B,), t, device='cuda'))
            S.STATE['t'], S.STATE['step'] = t, int(round(t * 100))
            ref = None
            for md in modes:
                name, _, arg = md.partition(':')
                L5.attn_score_proj = Scaled(proj0, float(arg)) if name == 'beta' else proj0
                S.CFG.update(mode='ref' if name == 'beta' else name, lam=float(arg) if name == 'bias' else 0.0)
                S.STATE['stash'] = st = {'layer': 5}
                with torch.set_grad_enabled(a.grad and name == 'ref'):
                    pl, pf = dec(t=temb, atom_types=batch.atom_types, frac_coords=in_f, lattices_rep=in_l,
                                 num_atoms=batch.num_atoms, node2graph=batch.batch,
                                 lattices_mat=lattice_polar_build_torch(in_l), cemb=None, guide_indicator=None)
                    lf = (pf - tar_f).pow(2).mean(1)
                    ll = (pl - tar_l).pow(2).mean(1)
                    if name == 'ref' and a.grad:
                        r = torch.autograd.grad(lf.sum(), st['agg'])[0]
                pf, pl, agg = pf.detach(), pl.detach(), st['agg'].detach()
                src, dst = st['ei']; n = agg.shape[0]
                w = st['w'][:, 0]; sf = src == dst
                wself = scatter(w * sf, src, dim=0, reduce='sum', dim_size=n)
                if name == 'ref':
                    ref = dict(pf=pf, lf=lf.detach(), ll=ll.detach(), agg=agg, w=w, wself=wself)
                    if a.grad:
                        grad_stats(A, t, bn, st, r, w, src, dst, sf, n)
                dv = pf - ref['pf']
                dl, dv2 = lf.detach() - ref['lf'], dv.pow(2).mean(1)
                cross = 2 * (dv * (ref['pf'] - tar_f)).mean(1)
                dm = (agg - ref['agg']).norm(dim=1) / ref['agg'].norm(dim=1).clamp_min(1e-12)
                e = st['e']; ns = (~sf).float()
                cnt = scatter(ns, src, dim=0, reduce='sum', dim_size=n).clamp_min(1)
                eo = scatter(e * ns[:, None], src, dim=0, reduce='sum', dim_size=n) / cnt[:, None]
                es = scatter(e * sf[:, None].float(), src, dim=0, reduce='sum', dim_size=n)
                dself = (wself - ref['wself']).pow(2) * (es - eo).pow(2).sum(1)
                pc = (w / (1 - wself[src]).clamp_min(1e-12)) * ns
                pr = (ref['w'] / (1 - ref['wself'][src]).clamp_min(1e-12)) * ns
                xl = lambda x, y: torch.where(x > 0, x * (x.clamp_min(1e-30).log() - y.clamp_min(1e-30).log()), 0 * x)
                kl = scatter(xl(pc, pr), src, dim=0, reduce='sum', dim_size=n)
                mid = 0.5 * (pc + pr)
                js = scatter(0.5 * xl(pc, mid) + 0.5 * xl(pr, mid), src, dim=0, reduce='sum', dim_size=n)
                for k in range(len(BINS)):
                    m = bn == k
                    if not m.any():
                        continue
                    D = A[f'{md}|{t}|{k}']
                    D['n'] += m.sum().item()
                    for nm, v in (('lf', lf.detach()), ('dl', dl), ('dl2', dl * dl), ('dv2', dv2), ('cross', cross),
                                  ('dm', dm), ('dself', dself), ('kl', kl), ('js', js)):
                        D[nm] += v[m].sum().item()
                    mg = bg == k
                    D['ng'] += mg.sum().item(); D['ll'] += ll.detach()[mg].sum().item()
                    D['dll'] += (ll.detach() - ref['ll'])[mg].sum().item()
        print('batch', bi, flush=True)
    L5.attn_score_proj = proj0
    out = {}
    for key, D in A.items():
        o = {k: v / max(D['ng' if k in ('ll', 'dll') else 'n'], 1) for k, v in D.items() if k not in ('n', 'ng')}
        o['n'] = D['n']
        if 'dl2' in o:
            o['dl_se'] = math.sqrt(max(o['dl2'] - o['dl'] ** 2, 0) / max(D['n'], 1))
        for c in ('zc', 'res', 'sh'):
            if f'gy_{c}' in D:
                o[f'corr_{c}'] = D[f'gy_{c}'] / math.sqrt(max(D['gg'] * D[f'yy_{c}'], 1e-30))
        out[key] = o
    Path(a.out).write_text(json.dumps(dict(model=a.model_path, tgrid=tg, modes=modes, bins=BINS, stats=out)))
    print('DONE', a.out, flush=True)


@torch.no_grad()
def grad_stats(A, t, bn, st, r, w, src, dst, sf, n):
    e, z = st['e'], st['z'][:, 0]
    q = (r[src] * e).sum(1)
    gz = w * (q - scatter(w * q, src, dim=0, reduce='sum', dim_size=n)[src])      # dl/dz_ij
    zc = z - scatter(z, src, dim=0, reduce='mean', dim_size=n)[src]
    ns = (~sf).float(); cnt = scatter(ns, src, dim=0, reduce='sum', dim_size=n).clamp_min(1)
    cen = lambda y: (y - (scatter(y * ns, src, dim=0, reduce='sum', dim_size=n) / cnt)[src]) * ns
    y, zn = cen(-gz), cen(z)
    x = cen(e.norm(dim=1).clamp_min(1e-6).log())
    sh = cen(S.STATE['shell'].float())
    be = bn[src]
    dbeta = scatter(gz * zc, src, dim=0, reduce='sum', dim_size=n)
    for k in range(len(BINS)):
        m = be == k
        if not m.any():
            continue
        D = A[f'grad|{t}|{k}']
        D['n'] += (bn == k).sum().item(); D['dbeta'] += dbeta[bn == k].sum().item()
        D['gnorm'] += scatter(gz.abs(), src, dim=0, reduce='sum', dim_size=n)[bn == k].sum().item()
        a_ = (x[m] * zn[m]).sum() / (x[m] * x[m]).sum().clamp_min(1e-12)
        res = zn[m] - a_ * x[m]
        D['gg'] += (y[m] ** 2).sum().item()
        for c, v in (('zc', zn[m]), ('res', res), ('sh', sh[m])):
            D[f'gy_{c}'] += (y[m] * v).sum().item(); D[f'yy_{c}'] += (v * v).sum().item()


if __name__ == '__main__':
    main()
