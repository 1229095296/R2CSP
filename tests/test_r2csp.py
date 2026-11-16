"""Unit tests of R2CSPNet against the CSPNet of CrystalFlow. Run from the CrystalFlow root on CPU:
  python -m pytest ../R2CSP/tests/test_r2csp.py      (or: python ../R2CSP/tests/test_r2csp.py)
"""
import importlib.util
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)  # the embeddings build their buffers in the default dtype

from diffcsp.pl_modules.cspnet import CSPNet  # noqa: E402

_spec = importlib.util.spec_from_file_location('r2csp_release', Path(__file__).resolve().parents[1] / 'r2csp' / 'r2csp.py')
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

KW = dict(hidden_dim=64, latent_dim=16, lattice_dim=9, num_layers=4, act_fn='silu', dis_emb='sin', num_freqs=16,
          rec_emb='sin', num_millers=3, edge_style='fc', ln=True, ip=True)
PAPER = dict(use_attention=True, num_heads=1, attn_last_k=1)


def batch(sizes=(1, 3, 7, 12), seed=0):
    g = torch.Generator().manual_seed(seed)
    na = torch.tensor(sizes)
    B, N = len(sizes), int(na.sum())
    L = torch.diag_embed(4 + 4 * torch.rand(B, 3, generator=g)) + 0.3 * torch.randn(B, 3, 3, generator=g)
    return dict(t=torch.randn(B, KW['latent_dim'], generator=g), atom_types=torch.randint(1, 95, (N,), generator=g),
                frac_coords=torch.rand(N, 3, generator=g), lattices_rep=L, num_atoms=na,
                node2graph=torch.repeat_interleave(torch.arange(B), na), lattices_mat=L)


def build(cls, seed=0, **kw):
    torch.manual_seed(seed)
    return cls(**KW, **kw).eval()


def close(a, b):
    return all(torch.allclose(x, y, rtol=0, atol=1e-10) for x, y in zip(a, b))


def test_all_off_equals_cspnet():
    base, net = build(CSPNet), build(R.R2CSPNet)
    net.load_state_dict(base.state_dict(), strict=True)
    b = batch()
    with torch.no_grad():
        assert close(base(**b), net(**b))


def test_zero_init_equals_mean():
    base, net = build(CSPNet), build(R.R2CSPNet, **PAPER)
    missing, unexpected = net.load_state_dict(base.state_dict(), strict=False)
    assert missing == ['csp_layer_%d.attn_score_proj.weight' % (KW['num_layers'] - 1)] and not unexpected
    assert torch.count_nonzero(net._modules['csp_layer_%d' % (KW['num_layers'] - 1)].attn_score_proj.weight) == 0
    b = batch()
    with torch.no_grad():
        assert close(base(**b), net(**b))


def test_parameter_count():
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    base = n(build(R.R2CSPNet))
    assert n(build(R.R2CSPNet, **PAPER)) - base == KW['hidden_dim']
    assert n(build(R.R2CSPNet, use_attention=True, num_heads=8)) - base == 8 * KW['hidden_dim'] * KW['num_layers']


def test_logit_receives_gradient():
    net = build(R.R2CSPNet, **PAPER).train()
    lat, coord = net(**batch())
    (lat.square().sum() + coord.square().sum()).backward()
    w = net._modules['csp_layer_%d' % (KW['num_layers'] - 1)].attn_score_proj.weight
    assert w.grad is not None and w.grad.abs().sum() > 0


def test_nonzero_logit_changes_output():
    net = build(R.R2CSPNet, **PAPER)
    layer = net._modules['csp_layer_%d' % (KW['num_layers'] - 1)]
    b = batch()
    with torch.no_grad():
        ref = net(**b)
        torch.nn.init.normal_(layer.attn_score_proj.weight, std=0.1)
        out = net(**b)
    assert not close(ref, out)
    assert all(torch.isfinite(x).all() for x in out)


def test_noself_mean():
    net = build(R.R2CSPNet, noself_last_k=1)
    b = batch(sizes=(1, 2, 5))  # a one-atom cell keeps its self edge
    with torch.no_grad():
        out = net(**b)
    assert all(torch.isfinite(x).all() for x in out)
    base = build(R.R2CSPNet)
    base.load_state_dict(net.state_dict(), strict=True)
    with torch.no_grad():
        assert not close(base(**b), out)


def test_distance_gate_starts_closed_and_keeps_rng():
    net = build(R.R2CSPNet, **PAPER)
    r_plain = torch.rand(3)
    gated = build(R.R2CSPNet, **PAPER, attn_dist_bias=True)
    r_gated = torch.rand(3)
    assert torch.equal(r_plain, r_gated)  # the gate does not shift the initialization stream
    sg = gated.state_dict()
    assert all(torch.equal(v, sg[k]) for k, v in net.state_dict().items())
    b = batch()
    with torch.no_grad():
        assert close(net(**b), gated(**b))


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            print('ok', name)
