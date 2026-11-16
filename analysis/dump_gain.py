"""Per-edge learned gain N*alpha_ij of the last layer vs. current distance.

Unmodified R2CSPNet, first --max_batches test batches. At the flow-time steps in
--steps, layer 5 records for every edge: N (deg incl. self), self flag, current minimum-image
distance d_ij, d_ij / min_k d_ik, ground-truth shell flag, gain N*alpha_ij, logit, log||e_ij||.
Reuses eval_intervene_shell.py unchanged (its node_model is wrapped, not edited).
  python dump_gain.py -m <eval dir> --out gain_dump.npz --max_batches 8 --steps 30,50,70,90
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_intervene_shell as S  # noqa: E402

REC = []
WANT = set()
_node_model = S.node_model


def node_model(self, node_features, edge_features, edge_index):
    if self.use_attention and self._idx == 5 and S.STATE['step'] in WANT:
        with torch.no_grad():
            src, dst = edge_index
            n = node_features.shape[0]
            z = self.attn_score_proj(edge_features)[:, 0]
            w = scatter_softmax(z, src, dim=0)
            deg = scatter(torch.ones_like(w), src, dim=0, reduce='sum', dim_size=n)
            d = S.STATE['logd'].exp()
            sf = src == dst
            dmin = scatter(torch.where(sf, torch.full_like(d, 1e9), d), src, dim=0, reduce='min', dim_size=n)
            REC.append(dict(
                step=np.full(z.shape[0], S.STATE['step'], np.int16),
                N=deg[src].cpu().numpy().astype(np.int16), self=sf.cpu().numpy(),
                d=d.cpu().numpy().astype(np.float32), drel=(d / dmin[src]).cpu().numpy().astype(np.float32),
                shell=S.STATE['shell'].cpu().numpy(), gain=(w * deg[src]).cpu().numpy().astype(np.float32),
                z=z.cpu().numpy().astype(np.float32),
                logn=edge_features.norm(dim=1).clamp_min(1e-6).log().cpu().numpy().astype(np.float32)))
    return _node_model(self, node_features, edge_features, edge_index)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--steps', default='30,50,70,90')
    a, rest = p.parse_known_args()
    WANT.update(int(x) for x in a.steps.split(','))
    S.node_model = node_model
    sys.argv = [sys.argv[0]] + rest + ['--mode', 'ref', '--geo', '--label', 'gaindump', '--layers', '5']
    S.main()
    out = {k: np.concatenate([r[k] for r in REC]) for k in REC[0]}
    np.savez_compressed(a.out, **out)
    print('SAVED', a.out, {k: v.shape for k, v in out.items()})
