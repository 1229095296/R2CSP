"""Inference-time interventions on the last-layer softmax weights of R2CSPNet.

Inside the flow-time window [t_lo, t_hi) and on the layers in --layers, the learned weights are
replaced as below; outside the window the model is untouched. The sampling noise is seeded, so
runs with the same --seed are paired structure by structure.

  ref          unmodified model
  uniform      alpha_ij = 1/N for all j, self edge included (mean aggregation)
  selfkeep     learned self weight kept, the rest spread evenly over the N-1 other edges
  selfflat     self weight set to 1/N, the other edges keep their learned ratios
  normprojN    non-self logits := atom mean + a_N * centred log||e_ij||, with a_N the within-atom
               least-squares slope of the logit on log||e_ij|| per cell size N; self logit kept
  residN       the same fit, keeping the residual instead of the projection
  normproj, resid   as above with one slope pooled over all N
  shuffle      non-self logits permuted among the non-self edges of each atom (own generator, so
               the global RNG and the pairing are untouched)
  bias         logits += lam * 1[j in S_i]
  shelloracle  learned logits renormalized over S_i and the self edge
  oracle       learned self weight kept, the rest spread evenly over S_i
  --shell_src  S_i from the final structures of another run (self-oracle)

S_i = {j != i : d_ij <= 1.25 min_k d_ik} in the reference structure, minimum-image Cartesian
distance. Atoms with N = 1, or without a shell neighbour for the shell-based modes, keep their
learned weights. --geo records, on unmodified calls, the correlation of log||e_ij|| with the
current distance (also for the mean-aggregation network).

Run from the CrystalFlow root:
  python eval_intervene_shell.py -m <ckpt dir> --label L --mode bias --lam 2 --layers 5
Writes <ckpt dir>/eval_diff_<L>.pt (format of scripts/evaluate.py) and attn_stats_<L>.json.
"""
import argparse, json, random, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax

sys.path.insert(0, 'scripts')
from eval_utils import load_model  # noqa: E402
from evaluate import diffusion  # noqa: E402
from diffcsp.common.data_utils import lattice_polar_build_torch  # noqa: E402
from diffcsp.pl_modules import r2csp as R  # noqa: E402

CFG = dict(mode='ref', lam=0.0, t_lo=0.0, t_hi=1.01, layers=None)
STATE = dict(t=None, step=-1, shell=None)
STATS = defaultdict(lambda: [0.0, 0.0, 0.0])  # (layer, step) -> [sum normH, sum eff_frac, n]
R2S = defaultdict(lambda: [0.0, 0.0])  # (layer, step) -> [resid SS, total SS] of the normproj fit
GEN = dict(g=None)
SELFW = defaultdict(lambda: [0.0, 0.0])  # (layer, step) -> [sum deg * w_self (1 = uniform), n]
GEO = defaultdict(lambda: defaultdict(float))  # (layer, step, Nbin) -> sums of geometry stats (--geo)


def active(layer):
    if CFG['mode'] == 'ref':
        return False
    if CFG['layers'] is not None and layer._idx not in CFG['layers']:
        return False
    return CFG['t_lo'] <= STATE['t'] < CFG['t_hi']


def node_model(self, node_features, edge_features, edge_index):
    if not self.use_attention:
        if CFG.get('geo') and self._idx == 5:
            geo_stats(self, torch.zeros_like(edge_features[:, :1]), edge_features, edge_index,
                      node_features.shape[0])
        agg = scatter(edge_features, edge_index[0], dim=0, reduce="mean", dim_size=node_features.shape[0])
        return self.node_mlp(torch.cat([node_features, agg], dim=1))
    src = edge_index[0]
    n = node_features.shape[0]
    logits = self.attn_score_proj(edge_features)
    on = active(self)
    sh = None
    if on:
        sh = STATE['shell']
        assert sh is not None and sh.shape[0] == edge_features.shape[0], \
            f'shell mask {None if sh is None else tuple(sh.shape)} vs {edge_features.shape[0]} edges'
        sh = sh.float()
        if CFG['mode'] == 'bias':
            logits = logits + CFG['lam'] * sh[:, None]
        elif CFG['mode'] == 'uniform':
            logits = torch.zeros_like(logits)
        elif CFG['mode'] in ('normproj', 'resid', 'normprojN', 'residN'):
            # Fit and substitute on the NON-self edges only, centred per node; the self edge keeps
            # its learned logit and every node keeps its non-self mean, so the self weight is
            # roughly preserved and only the allocation among neighbours changes. (The self edge's
            # ||e_ii|| is an outlier: including it lets the fit hand the self edge all the mass.)
            ns = (src != edge_index[1])
            nsf = ns.to(logits.dtype)
            cnt = scatter(nsf, src, dim=0, reduce='sum', dim_size=n).clamp_min(1.0)
            x = edge_features.norm(dim=1).clamp_min(1e-6).log()
            mx = scatter(x * nsf, src, dim=0, reduce='sum', dim_size=n) / cnt
            ml = scatter(logits * nsf[:, None], src, dim=0, reduce='sum', dim_size=n) / cnt[:, None]
            xc = (x - mx[src]) * nsf
            lc = (logits - ml[src]) * nsf[:, None]
            if CFG['mode'].endswith('N'):
                # slope fitted separately for every crystal size N (deg incl. self = N), since
                # the pooled slope mixes a ~ -0.3 (small N) with a ~ -4.7 (N 17-20)
                dg = scatter(torch.ones_like(nsf), src, dim=0, reduce='sum', dim_size=n).long()
                de = dg[src]
                nb = int(de.max()) + 1
                sxz = scatter(lc * xc[:, None], de, dim=0, reduce='sum', dim_size=nb)          # (nb, H)
                sxx = scatter(xc.pow(2), de, dim=0, reduce='sum', dim_size=nb).clamp_min(1e-12)
                fit = xc[:, None] * (sxz / sxx[:, None])[de]
            else:
                a = (lc * xc[:, None]).sum(0) / xc.pow(2).sum().clamp_min(1e-12)    # (H,)
                fit = xc[:, None] * a[None, :]
            r = R2S[(self._idx, STATE['step'])]
            r[0] += (lc - fit).pow(2).sum().item(); r[1] += lc.pow(2).sum().item()
            new = ml[src] + (fit if CFG['mode'].startswith('normproj') else lc - fit)
            logits = torch.where(ns[:, None], new, logits)
        elif CFG['mode'] == 'shuffle':
            notself = (src != edge_index[1]).double()
            key = torch.rand(src.shape[0], generator=GEN['g'], device=src.device, dtype=torch.float64)
            base = torch.argsort(src.double() * 3 + notself, stable=True)
            perm = torch.argsort(src.double() * 3 + notself * (1 + key), stable=True)
            new = torch.empty_like(logits)
            new[base] = logits[perm]                     # self edge stays put, others permuted
            logits = new
    w = scatter_softmax(logits, src, dim=0)  # (E, H)
    if CFG.get('geo') and not on:
        geo_stats(self, logits, edge_features, edge_index, n)
    if on and CFG['mode'] == 'shelloracle':
        # perfect selection with the learned gain preserved: keep the learned logits but
        # renormalize them over {true shell} u {self} only. Separates "which neighbours" from
        # "how much each one is scaled", which plain oracle destroys together.
        keep = sh.bool() | (src == edge_index[1])
        w_sel = scatter_softmax(logits.masked_fill(~keep[:, None], float('-inf')), src, dim=0)
        cnt = scatter(sh, src, dim=0, reduce='sum', dim_size=n)
        w = torch.where((cnt > 0)[src][:, None], w_sel, w)   # no shell neighbour -> leave learned
    if on and CFG['mode'] in ('selfkeep', 'selfflat'):
        selfe = (src == edge_index[1])[:, None]
        dg = scatter(torch.ones_like(w[:, 0]), src, dim=0, reduce="sum", dim_size=n)[:, None]   # (N, 1)
        if CFG['mode'] == 'selfkeep':
            w_self = scatter(w * selfe, src, dim=0, reduce='sum', dim_size=n)                     # (N, H)
            per = (1.0 - w_self) / (dg - 1.0).clamp_min(1.0)
            w = torch.where(selfe, w, per[src])
        else:
            w_ns = w * (~selfe)
            s_ns = scatter(w_ns, src, dim=0, reduce='sum', dim_size=n).clamp_min(1e-30)
            w = torch.where(selfe, (1.0 / dg)[src].expand_as(w), w_ns / s_ns[src] * (1.0 - 1.0 / dg)[src])
        w = torch.where((dg > 1)[src], w, scatter_softmax(logits, src, dim=0))
    if on and CFG['mode'] == 'oracle':
        selfe = (src == edge_index[1])[:, None]
        w_self = scatter(w * selfe, src, dim=0, reduce='sum', dim_size=n)       # (N, H)
        cnt = scatter(sh, src, dim=0, reduce='sum', dim_size=n)                 # (N,)
        per = (1.0 - w_self)[src] / cnt.clamp_min(1.0)[src][:, None]            # mass per shell edge
        w = torch.where(selfe, w, sh[:, None] * per)
        w = torch.where((cnt > 0)[src][:, None], w, scatter_softmax(logits, src, dim=0))
    # attention statistics (normalized entropy, effective-neighbour fraction), head-averaged
    ent = scatter(-(w * w.clamp_min(1e-30).log()), src, dim=0, reduce="sum", dim_size=n).mean(1)
    deg = scatter(torch.ones_like(w[:, 0]), src, dim=0, reduce="sum", dim_size=n)
    m = deg > 1
    s = STATS[(self._idx, STATE['step'])]
    s[0] += (ent[m] / deg[m].log()).sum().item()
    s[1] += (ent[m].exp() / deg[m]).sum().item()
    s[2] += m.sum().item()
    selfm = (src == edge_index[1])
    ws = scatter(w.mean(1) * selfm, src, dim=0, reduce='sum', dim_size=n)
    q = SELFW[(self._idx, STATE['step'])]
    q[0] += (ws[m] * deg[m]).sum().item(); q[1] += m.sum().item()
    chunk = edge_features.shape[1] // self.num_heads
    agg = torch.cat([
        scatter(edge_features[:, h * chunk:(h + 1) * chunk] * w[:, h:h + 1], src, dim=0,
                reduce="sum", dim_size=n)
        for h in range(self.num_heads)
    ], dim=1)
    st = STATE.get('stash')
    if st is not None and self._idx == st.get('layer', 5):
        # for eval_fmloss.py: learned (pre-intervention) logits, final weights, messages, aggregate
        if agg.requires_grad:
            agg.retain_grad()
        st.update(z=self.attn_score_proj(edge_features).detach(), w=w.detach(), e=edge_features.detach(),
                  ei=edge_index, agg=agg)
    return self.node_mlp(torch.cat([node_features, agg], dim=1))


@torch.no_grad()
def geo_stats(self, logits, e, edge_index, n):
    """Theory inputs on the unperturbed trajectory, head 0, per (layer, step, N bin):
    within-node OLS slope a and R^2 of the logit on log||e|| (non-self), self-vs-rest gaps of
    log||e|| and of the logit, cos(e_ii, mean_{j!=i} e_ij), and the norm share of the self term
    in the plain mean aggregate, ||e_ii / N|| / ||mean_j e_ij||."""
    src, dst = edge_index
    sf = src == dst
    ns = (~sf).to(e.dtype)
    deg = scatter(torch.ones_like(ns), src, dim=0, reduce='sum', dim_size=n)
    cnt = (deg - 1).clamp_min(1.0)
    x = e.norm(dim=1).clamp_min(1e-6).log()
    z = logits[:, 0]
    mx = scatter(x * ns, src, dim=0, reduce='sum', dim_size=n) / cnt
    mz = scatter(z * ns, src, dim=0, reduce='sum', dim_size=n) / cnt
    xc, zc = (x - mx[src]) * ns, (z - mz[src]) * ns
    xs = scatter(x * sf, src, dim=0, reduce='sum', dim_size=n)
    zs = scatter(z * sf, src, dim=0, reduce='sum', dim_size=n)
    es = scatter(e * sf[:, None].to(e.dtype), src, dim=0, reduce='sum', dim_size=n)
    eo = scatter(e * ns[:, None], src, dim=0, reduce='sum', dim_size=n) / cnt[:, None]
    em = scatter(e, src, dim=0, reduce='mean', dim_size=n)
    cos = F.cosine_similarity(es, eo, dim=1)
    share = (es.norm(dim=1) / deg) / em.norm(dim=1).clamp_min(1e-12)
    ok = deg > 2
    for lo, hi in ((1, 4), (5, 8), (9, 12), (13, 16), (17, 20), (21, 999)):
        m = ok & (deg >= lo) & (deg <= hi)
        if not m.any():
            continue
        me = m[src] & (~sf)
        g = GEO[(self._idx, STATE['step'], f'{lo}-{hi}')]
        g['n'] += m.sum().item()
        g['sxz'] += (xc * zc)[me].sum().item(); g['sxx'] += (xc * xc)[me].sum().item()
        g['szz'] += (zc * zc)[me].sum().item()
        g['dx'] += (xs - mx)[m].sum().item(); g['dz'] += (zs - mz)[m].sum().item()
        g['sdx'] += xc.pow(2)[me].sum().item()
        g['ne'] += me.sum().item()
        g['cos'] += cos[m].sum().item(); g['share'] += share[m].sum().item()
        # within-node centred correlations of log||e|| (and of the logit) with the GT-shell
        # indicator and with log d^t (current geometry), non-self edges only
        for nm, y in (('sh', STATE.get('shell')), ('ld', STATE.get('logd'))):
            if y is None or y.shape[0] != e.shape[0]:
                continue
            y = y.to(e.dtype)
            my = scatter(y * ns, src, dim=0, reduce='sum', dim_size=n) / cnt
            yc = (y - my[src]) * ns
            g[f'xy_{nm}'] += (xc * yc)[me].sum().item(); g[f'zy_{nm}'] += (zc * yc)[me].sum().item()
            g[f'yy_{nm}'] += (yc * yc)[me].sum().item()


def geo_summary():
    out = {}
    for (l, s, b), g in sorted(GEO.items()):
        a = g['sxz'] / max(g['sxx'], 1e-30)
        out[f'{l}_{s}_{b}'] = dict(
            n=g['n'], slope=a, r2=a * a * g['sxx'] / max(g['szz'], 1e-30),
            self_minus_rest_logn=g['dx'] / g['n'], self_minus_rest_logit=g['dz'] / g['n'],
            sd_logn_rest=(g['sdx'] / max(g['ne'], 1)) ** 0.5,
            cos_self_rest=g['cos'] / g['n'], self_norm_share_in_mean=g['share'] / g['n'])
        for nm in ('sh', 'ld'):
            if g.get(f'yy_{nm}'):
                out[f'{l}_{s}_{b}'][f'corr_logn_{nm}'] = g[f'xy_{nm}'] / (g['sxx'] * g[f'yy_{nm}']) ** 0.5
                out[f'{l}_{s}_{b}'][f'corr_logit_{nm}'] = g[f'zy_{nm}'] / max(g['szz'] * g[f'yy_{nm}'], 1e-30) ** 0.5
    return out


@torch.no_grad()
def compute_shell(model, batch):
    """First coordination shell of the GROUND-TRUTH structure, per edge of the fc graph.

    The 'fc' edge list depends only on num_atoms (block-diagonal complete graph), so it is the
    same tensor the decoder builds at every ODE step regardless of the current coordinates.
    """
    n = batch.num_nodes
    L = lattice_polar_build_torch(batch.lattice_polar)
    edges, _ = model.decoder.gen_edges(batch.num_atoms, batch.frac_coords, L, batch.batch)
    src, dst = edges
    g = batch.batch[src]
    notself = src != dst
    fd = (batch.frac_coords[dst] - batch.frac_coords[src] + 0.5) % 1 - 0.5
    d = torch.linalg.norm(torch.einsum('ei,eij->ej', fd, L[g]), dim=-1)
    dm = scatter(torch.where(notself, d, torch.full_like(d, 1e9)), src, dim=0,
                 reduce='min', dim_size=n)
    return notself & (d <= 1.25 * dm[src])


@torch.no_grad()
def compute_shell_from(model, batch, frac, L):
    """Same shell rule, but on a given structure (frac (n,3), lattice matrices L (B,3,3)) with the
    batch's atom order -- e.g. the reference run's own final sample (--shell_src)."""
    n = batch.num_nodes
    edges, _ = model.decoder.gen_edges(batch.num_atoms, frac, L, batch.batch)
    src, dst = edges
    notself = src != dst
    fd = (frac[dst] - frac[src] + 0.5) % 1 - 0.5
    d = torch.linalg.norm(torch.einsum('ei,eij->ej', fd, L[batch.batch[src]]), dim=-1)
    dm = scatter(torch.where(notself, d, torch.full_like(d, 1e9)), src, dim=0, reduce='min', dim_size=n)
    return notself & (d <= 1.25 * dm[src])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--model_path', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mode', choices=('ref', 'bias', 'oracle', 'shelloracle',
                                           'uniform', 'normproj', 'resid', 'shuffle',
                                           'selfkeep', 'selfflat', 'normprojN', 'residN'),
                   default='ref')
    p.add_argument('--lam', type=float, default=0.0, help='bias mode: logit += lam * 1[shell]')
    p.add_argument('--t_lo', type=float, default=0.3)
    p.add_argument('--t_hi', type=float, default=0.7)
    p.add_argument('--layers', default='', help='comma list; empty = all attention layers')
    p.add_argument('-N', '--ode_int_steps', type=int, default=100)
    p.add_argument('--anneal_slope', type=float, default=5.0)
    p.add_argument('--max_batches', type=int, default=0, help='screening only')
    p.add_argument('--geo', action='store_true', help='collect geo_stats on unperturbed calls')
    p.add_argument('--shell_src', default='', help='eval_diff_*.pt whose FINAL samples define the shell '
                   '(self-oracle) instead of the ground truth; same loader order and noise seed')
    a = p.parse_args()
    CFG.update(geo=a.geo, mode=a.mode, lam=a.lam, t_lo=a.t_lo, t_hi=a.t_hi,
               layers=set(int(x) for x in a.layers.split(',')) if a.layers else None)

    random.seed(a.seed); np.random.seed(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    GEN['g'] = torch.Generator(device='cuda').manual_seed(a.seed + 1000)
    R.R2CSPLayer.node_model = node_model
    model_path = Path(a.model_path)
    model, loader, _ = load_model(model_path, load_data=True)
    model.to('cuda').eval()
    if a.max_batches:
        loader = [b for _, b in zip(range(a.max_batches), loader)]
    dec = model.decoder
    for i in range(dec.num_layers):
        dec._modules['csp_layer_%d' % i]._idx = i
    attn_layers = [i for i in range(dec.num_layers) if dec._modules['csp_layer_%d' % i].use_attention]
    print('attention layers:', attn_layers, 'CFG:', CFG, flush=True)

    # the hook cannot see the batch, so stash the ground-truth shell mask when sample() gets it
    _orig_sample = model.sample
    n_shell, n_edges = [0], [0]
    SRC = None
    if a.shell_src:
        d = torch.load(a.shell_src, map_location='cpu', weights_only=False)
        SRC = dict(f=d['frac_coords'][0], L=d['lattices'][0], na=d['num_atoms'][0], node=0, g=0)

    def sample(batch, *args, **kwargs):
        if SRC is None:
            sh = compute_shell(model, batch)
        else:
            B, nn_ = batch.num_graphs, batch.num_nodes
            assert torch.equal(SRC['na'][SRC['g']:SRC['g'] + B], batch.num_atoms.cpu()), 'shell_src order mismatch'
            f = SRC['f'][SRC['node']:SRC['node'] + nn_].to(batch.frac_coords)
            L = SRC['L'][SRC['g']:SRC['g'] + B].to(batch.frac_coords)
            SRC['node'] += nn_; SRC['g'] += B
            sh = compute_shell_from(model, batch, f, L)
        STATE['shell'] = sh
        n_shell[0] += int(sh.sum()); n_edges[0] += int(sh.numel())
        return _orig_sample(batch, *args, **kwargs)
    model.sample = sample

    te_forward = model.time_embedding.forward

    if a.geo:   # stash log d^t (current minimum-image distance) per fc edge for geo_stats
        dec_forward = model.decoder.forward

        def dec_tracked(*args, **kw):
            fcur, Lm, na, n2g = kw['frac_coords'], kw['lattices_mat'], kw['num_atoms'], kw['node2graph']
            ed, _ = model.decoder.gen_edges(na, fcur, Lm, n2g)
            fd = (fcur[ed[1]] - fcur[ed[0]] + 0.5) % 1 - 0.5
            d = torch.linalg.norm(torch.einsum('ei,eij->ej', fd, Lm[n2g[ed[0]]]), dim=-1)
            STATE['logd'] = d.clamp_min(1e-3).log()
            return dec_forward(*args, **kw)
        model.decoder.forward = dec_tracked

    def tracked(times):
        tv = float(times[0])
        STATE['step'] = int(round(tv * a.ode_int_steps))
        STATE['t'] = tv
        return te_forward(times)
    model.time_embedding.forward = tracked

    t0 = time.time()
    fc, at, lat, lengths, angles, na, inp = diffusion(
        loader, model, num_evals=1, step_lr=1 / a.ode_int_steps, N=a.ode_int_steps,
        anneal_lattice=False, anneal_coords=True, anneal_type=False,
        anneal_slope=a.anneal_slope, anneal_offset=0.0, guide_factor=None)
    torch.save({'eval_setting': a, 'input_data_batch': inp, 'frac_coords': fc, 'num_atoms': na,
                'atom_types': at, 'lattices': lat, 'lengths': lengths, 'angles': angles,
                'time': time.time() - t0}, model_path / f'eval_diff_{a.label}.pt')
    stats = {f'{l}_{s}': [v[0] / max(v[2], 1), v[1] / max(v[2], 1)] for (l, s), v in sorted(STATS.items())}
    (model_path / f'attn_stats_{a.label}.json').write_text(json.dumps(
        dict(cfg={k: (sorted(v) if isinstance(v, set) else v) for k, v in CFG.items()},
             seed=a.seed, attn_layers=attn_layers,
             shell_pos_rate=n_shell[0] / max(n_edges[0], 1),
             stats_layer_step_normH_effrac=stats,
             selfw_ratio_layer_step={f'{l}_{s}': v[0] / max(v[1], 1)
                                     for (l, s), v in sorted(SELFW.items())},
             normproj_r2_layer_step={f'{l}_{s}': 1 - v[0] / max(v[1], 1e-30)
                                     for (l, s), v in sorted(R2S.items())},
             geo=geo_summary())))
    print('DONE', time.time() - t0, 'shell_pos_rate', n_shell[0] / max(n_edges[0], 1), flush=True)


if __name__ == '__main__':
    main()
