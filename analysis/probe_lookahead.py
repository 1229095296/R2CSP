"""First-shell information along the trajectory: current distance, predicted endpoint, learned weight.

Per flow time t on the training interpolation path (f_t = f0 + t*((f1-f0-.5)%1-.5), same for the
polar lattice rep), one decoder forward gives the raw velocity (v_f, v_l) -- which in this
formulation IS the f0->f1 displacement -- so the self-conditioning endpoint is
    f1_hat = f_t + (1-t) * v_f,   l1_hat = l_t + (1-t) * v_l        (no anneal: matches training)

AUROC for "j is in i's first coordination shell of the TARGET structure" from
       (a) -d^t   current noisy minimum-image distance        (what the input already shows)
       (b) -d_hat predicted-endpoint distance                 (the lookahead signal)
       (b') -d_hat computed with the TRUE lattice             (coords-only lookahead)
       (c) the learned last-layer attention weight
       (d) a 2-feature logistic probe on (d^t, d_hat)         (does d_hat add information?)
       (e) a probe on the last attention layer's edge feature  (what the scorer could reach)
     both pooled and per-node (per-node = what a within-node softmax actually ranks).
Sharpness: normalized entropy of the learned attention vs the calibrated-posterior
     ("Bayes") weights p(shell | d^t) and vs the uniform-on-true-shell upper bound.
Dilution: |S_i| / deg distribution vs N.
Also reports endpoint error vs t (raw and translation-aligned).

Run from the CrystalFlow root:
  python probe_lookahead.py -m <ckpt dir> --label L --out_dir OUT
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax

sys.path.insert(0, 'scripts')
from eval_utils import load_model  # noqa: E402
from diffcsp.common.data_utils import lattice_polar_build_torch  # noqa: E402
from diffcsp.pl_modules import r2csp as R  # noqa: E402

REC = []
_orig_node_model = R.R2CSPLayer.node_model


def node_model(self, node_features, edge_features, edge_index):
    if self.use_attention:
        logits = self.attn_score_proj(edge_features)
        w = scatter_softmax(logits, edge_index[0], dim=0).mean(1)
        REC.append((edge_features.detach().half(), w.detach(), logits.detach().mean(1),
                    edge_features.detach().float().norm(dim=1)))
    return _orig_node_model(self, node_features, edge_features, edge_index)


def mic_dist(fd, L):  # fd (E,3) frac diff, L (E,3,3)
    return torch.linalg.norm(torch.einsum('ei,eij->ej', (fd + 0.5) % 1 - 0.5, L), dim=-1)


def shell(d, src, n, notself):
    dm = scatter(torch.where(notself, d, torch.full_like(d, 1e9)), src, dim=0, reduce='min', dim_size=n)
    return notself & (d <= 1.25 * dm[src])


def segments(src):
    """Indices sorted by src plus split boundaries."""
    order = np.argsort(src, kind='stable')
    bounds = np.flatnonzero(np.diff(src[order])) + 1
    return order, bounds


def node_auc(score, src, y, order, bounds, min_deg=4):
    """Mean within-node ROC AUC (rank-sum form); nodes with both classes and deg >= min_deg."""
    s_o, y_o = score[order], y[order]
    aucs = []
    for ss, yy in zip(np.split(s_o, bounds), np.split(y_o, bounds)):
        npos = int(yy.sum())
        if npos == 0 or npos == len(yy) or len(yy) < min_deg:
            continue
        r = rankdata(ss)
        aucs.append((r[yy].sum() - npos * (npos + 1) / 2) / (npos * (len(yy) - npos)))
    return float(np.mean(aucs)), len(aucs)


def probe(X, y, sid, seed, Xall=None):
    """Logistic probe, structure-level 70/30 split, standardized; returns (test AUROC, p_all)."""
    rng = np.random.RandomState(seed)
    us = np.unique(sid); te_s = set(rng.choice(us, int(0.3 * len(us)), replace=False))
    te = torch.tensor([s in te_s for s in sid]).cuda()
    X = X.float().cuda(); y = y.float().cuda()
    mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-6
    Xs = (X - mu) / sd
    lin = torch.nn.Linear(X.shape[1], 1).cuda()
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=200, line_search_fn='strong_wolfe')
    Xtr, ytr = Xs[~te], y[~te]

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lin(Xtr).squeeze(1), ytr) + 1e-4 * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        auc = float(roc_auc_score(y[te].cpu().numpy(), lin(Xs[te]).squeeze(1).cpu().numpy()))
        p = None if Xall is None else torch.sigmoid(lin((Xall.float().cuda() - mu) / sd).squeeze(1)).cpu()
    return auc, p


@torch.no_grad()
def collect(model, batches, t):
    """One forward per batch at flow time t; edge-level tensors concatenated over batches."""
    out = defaultdict(list)
    noff = goff = 0
    for b in batches:
        B, n = b.num_graphs, b.num_nodes
        l0 = model.sample_lattice_polar(B)
        lt = l0 + t * (b.lattice_polar - l0)
        Lt, LT = lattice_polar_build_torch(lt), lattice_polar_build_torch(b.lattice_polar)
        f0 = torch.rand_like(b.frac_coords)
        ft = f0 + t * ((b.frac_coords - f0 - 0.5) % 1 - 0.5)
        REC.clear()
        pred = model.decoder(t=model.time_embedding(torch.full((B,), t, device='cuda')),
                             atom_types=b.atom_types, frac_coords=ft, lattices_rep=lt,
                             num_atoms=b.num_atoms, node2graph=b.batch, lattices_mat=Lt,
                             cemb=None, guide_indicator=None)
        pred_l, pred_f = pred[0], pred[1]
        # self-conditioning endpoint: the velocity target is the (wrapped) f0 -> f1 displacement
        fh = ft + (1.0 - t) * pred_f
        lh = lt + (1.0 - t) * pred_l
        Lh = lattice_polar_build_torch(lh)

        edges, fd_t = model.decoder.gen_edges(b.num_atoms, ft, Lt, b.batch)
        src, dst = edges
        g, notself = b.batch[src], src != dst
        dT = mic_dist(b.frac_coords[dst] - b.frac_coords[src], LT[g])
        dt = mic_dist(fd_t, Lt[g])
        fd_h = fh[dst] - fh[src]
        dh = mic_dist(fd_h, Lh[g])
        dh_Ltrue = mic_dist(fd_h, LT[g])
        yT = shell(dT, src, n, notself)

        # endpoint error (per atom, Angstrom): raw and after removing the per-structure translation
        df = (fh - b.frac_coords + 0.5) % 1 - 0.5
        err = torch.linalg.norm(torch.einsum('ni,nij->nj', df, LT[b.batch]), dim=-1)
        dfc = df - scatter(df, b.batch, dim=0, reduce='mean')[b.batch]
        err_al = torch.linalg.norm(torch.einsum('ni,nij->nj', (dfc + 0.5) % 1 - 0.5, LT[b.batch]), dim=-1)
        lerr = (lh - b.lattice_polar).abs().mean(tuple(range(1, b.lattice_polar.dim())))

        out['src'].append((src + noff).cpu()); out['sid'].append((g + goff).cpu())
        out['N'].append(b.num_atoms[g].cpu()); out['keep'].append(notself.cpu())
        out['yT'].append(yT.cpu()); out['dT'].append(dT.cpu()); out['dt'].append(dt.cpu())
        out['dh'].append(dh.cpu()); out['dhL'].append(dh_Ltrue.cpu())
        out['err'].append(err.cpu()); out['err_al'].append(err_al.cpu()); out['lerr'].append(lerr.cpu())
        out['same'].append((b.atom_types[src] == b.atom_types[dst]).cpu())
        for li, (ef, w, lg, en) in enumerate(REC):
            out[f'ef{li}'].append(ef.cpu()); out[f'w{li}'].append(w.cpu())
            out[f'lg{li}'].append(lg.cpu()); out[f'en{li}'].append(en.cpu())
        noff += n; goff += B
    return {k: torch.cat(v) for k, v in out.items()}


def bayes_weights(dt, src, y, n, order, bounds, nbin=40):
    """Nonparametric calibration of p(shell | r), r = d^t / min_j d^t_ij, then weights ~ p."""
    dm = scatter(dt, src, dim=0, reduce='min', dim_size=n)
    r = (dt / dm[src].clamp_min(1e-9)).numpy()
    q = np.quantile(r, np.linspace(0, 1, nbin + 1)[1:-1])
    bi = np.searchsorted(q, r)
    yy = y.numpy().astype(np.float64)
    num = np.bincount(bi, weights=yy, minlength=nbin)
    den = np.bincount(bi, minlength=nbin)
    p = torch.tensor((num + 0.5) / (den + 1.0))[bi].float()
    return p, float(node_auc(-r, None, y.numpy(), order, bounds)[0])


def norm_entropy(w, src, n, deg):
    p = w / scatter(w, src, dim=0, reduce='sum', dim_size=n)[src].clamp_min(1e-30)
    H = scatter(-(p * p.clamp_min(1e-30).log()), src, dim=0, reduce='sum', dim_size=n)
    return (H / deg.clamp_min(2).log()).numpy()


def ctr(x, src, n):
    """Within-node centering: the softmax is shift-invariant per node, so only this part acts."""
    return x - scatter(x, src, dim=0, reduce='mean', dim_size=n)[src]


def r2(y, X):
    """R^2 of a least-squares fit of already-centered y on already-centered columns of X."""
    sol = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1.0 - ((y - X @ sol) ** 2).sum() / (y ** 2).sum())


def pe5_stats(lg, dt, dh, en, same, src, y, n):
    """What does the within-node logit variation actually key on?"""
    c = {'dt': ctr(dt, src, n), 'dhat': ctr(dh, src, n),
         'log_enorm': ctr(en.clamp_min(1e-6).log(), src, n), 'same_species': ctr(same.float(), src, n)}
    yl = ctr(lg, src, n).numpy().astype(np.float64)
    feats = {k: v.numpy().astype(np.float64)[:, None] for k, v in c.items()}
    out = {f'r2_{k}': r2(yl, v) for k, v in feats.items()}
    out['r2_all'] = r2(yl, np.concatenate(list(feats.values()), 1))
    out['r2_geom_only'] = r2(yl, np.concatenate([feats['dt'], feats['dhat']], 1))
    out['r2_enorm_species'] = r2(yl, np.concatenate([feats['log_enorm'], feats['same_species']], 1))
    # is the species relation itself predictive of the target shell, and does it explain the sign?
    yn, sm = y.numpy(), same.numpy()
    o, b = segments(src.numpy())
    out['auc_node_unlike_for_shell'] = node_auc((~sm).astype(float), src, yn, o, b)[0]
    out['auc_node_attn_for_unlike'] = node_auc(lg.numpy(), src, ~sm, o, b)[0]
    for nm, msk in (('unlike', ~sm), ('like', sm)):
        if msk.sum() < 1000:
            continue
        s2 = src.numpy()[msk]
        o2, b2 = segments(s2)
        out[f'auc_node_attn_shell_{nm}'] = node_auc(lg.numpy()[msk], s2, yn[msk], o2, b2)[0]
        out[f'auc_node_dt_shell_{nm}'] = node_auc(-dt.numpy()[msk], s2, yn[msk], o2, b2)[0]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--model_path', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--n_batches', type=int, default=4)
    p.add_argument('--ts', default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    p.add_argument('--max_probe_edges', type=int, default=200000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', required=True)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    R.R2CSPLayer.node_model = node_model
    model, loader, _ = load_model(Path(a.model_path), load_data=True)
    model.to('cuda').eval()
    batches = [b.cuda() for _, b in zip(range(a.n_batches), loader)]
    res = dict(label=a.label, model_path=a.model_path, seed=a.seed,
               n_structs=int(sum(b.num_graphs for b in batches)), per_t={})

    for t in (float(x) for x in a.ts.split(',')):
        torch.manual_seed(a.seed)
        D = collect(model, batches, t)
        k = D['keep']
        src = D['src'][k]
        n = int(D['src'].max()) + 1
        y = D['yT'][k]
        yn = y.numpy()
        order, bounds = segments(src.numpy())
        deg = scatter(torch.ones(len(src)), src, dim=0, reduce='sum', dim_size=n)
        sfrac = scatter(y.float(), src, dim=0, reduce='sum', dim_size=n) / deg.clamp_min(1)
        Nn = scatter(D['N'][k].float(), src, dim=0, reduce='max', dim_size=n)
        nlayer = len([key for key in D if key.startswith('w')])
        wl = D[f'w{nlayer - 1}'][k]  # last attention layer

        row = dict(pos_rate=float(y.float().mean()),
                   endpoint_err=float(D['err'].mean()), endpoint_err_aligned=float(D['err_al'].mean()),
                   lattice_err=float(D['lerr'].mean()))
        scores = {'dt': -D['dt'][k].numpy(), 'dhat': -D['dh'][k].numpy(),
                  'dhat_Ltrue': -D['dhL'][k].numpy(), 'attn': wl.numpy()}
        row['auc_pooled'] = {name: float(roc_auc_score(yn, s)) for name, s in scores.items()}
        row['auc_node'] = {}
        for name, s in scores.items():
            auc, nn = node_auc(s, src, yn, order, bounds)
            row['auc_node'][name] = auc
        row['n_nodes_auc'] = nn

        sel = torch.arange(len(src))
        if len(sel) > a.max_probe_edges:
            sel = sel[torch.randperm(len(sel))[:a.max_probe_edges]].sort().values
        sid = D['sid'][k][sel].numpy()
        feats = {'dt': D['dt'][k][sel, None], 'dhat': D['dh'][k][sel, None],
                 'dt+dhat': torch.stack([D['dt'][k][sel], D['dh'][k][sel]], 1),
                 'edge_feat': D[f'ef{nlayer - 1}'][k][sel]}
        row['auc_probe'] = {name: probe(X, y[sel], sid, a.seed)[0] for name, X in feats.items()}

        pb, auc_r = bayes_weights(D['dt'][k], src, y, n, order, bounds)
        m = (deg.numpy() >= 4)
        shell_size = (sfrac * deg).numpy()
        row['entropy'] = dict(attn=float(norm_entropy(wl, src, n, deg)[m].mean()),
                              bayes_dt=float(norm_entropy(pb, src, n, deg)[m].mean()),
                              true_shell=float(np.mean((np.log(np.maximum(shell_size, 1.0))
                                                        / np.log(np.maximum(deg.numpy(), 2.0)))[m])),
                              uniform=1.0)
        row['auc_node_ratio'] = auc_r
        row['pe5'] = pe5_stats(D[f'lg{nlayer - 1}'][k], D['dt'][k], D['dh'][k],
                               D[f'en{nlayer - 1}'][k], D['same'][k], src, y, n)
        row['dilution'] = {f'{lo}-{hi}': dict(n=int(mm.sum()), shell_frac=float(sfrac.numpy()[mm].mean()),
                                              deg=float(deg.numpy()[mm].mean()))
                           for lo, hi in ((1, 4), (5, 8), (9, 12), (13, 16), (17, 20), (21, 999))
                           for mm in [m & (Nn.numpy() >= lo) & (Nn.numpy() <= hi)] if mm.sum() >= 20}
        res['per_t'][f'{t:g}'] = row
        print(f'=== t={t:g}', json.dumps({kk: vv for kk, vv in row.items() if kk != 'dilution'}), flush=True)
        del D

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f'lookahead_{a.label}.json').write_text(json.dumps(res, indent=1))
    print('DONE', out / f'lookahead_{a.label}.json')


if __name__ == '__main__':
    main()
