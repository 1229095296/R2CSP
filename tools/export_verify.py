"""Export weights-only release checkpoints and verify them against the training-time module.

For each run: check that the decoder flags removed from the release module are off, write a
cleaned hparams.yaml and a weights-only checkpoint, then
  (1) strict-load the original checkpoint with the training-time r2csp.py,
  (2) strict-load the exported checkpoint with the release r2csp.py through scripts/eval_utils.load_model,
  (3) compare the two state_dicts and the decoder outputs on fixed synthetic batches bit for bit,
  (4) build both decoders from the same seed and compare the initial weights and the RNG state.
The training-time module is the one installed as diffcsp.pl_modules.r2csp when the runs were trained.
Run from the CrystalFlow root on CPU:
  python export_verify.py --release_module ../R2CSP/r2csp/r2csp.py --out OUT --report REPORT.json name=RUN_DIR:CKPT ...
"""
import argparse, copy, importlib.util, json, re, shutil, sys, traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf, DictConfig, ListConfig

sys.path.insert(0, 'scripts')
import diffcsp.pl_modules.r2csp as OLD  # noqa: E402
import hydra  # noqa: E402

REMOVED_OFF = dict(use_cart_rbf=False, use_recip_kernel=False, use_equivariant_coord=False,
                   shared_edge_proj_dim=0, use_recip_mixer=False)
REMOVED = list(REMOVED_OFF) + ['rbf_centers', 'rbf_cutoff', 'kernel_sigmas', 'kernel_order', 'attn_type',
                               'attn_qk_dim', 'mixer_dim', 'mixer_sigmas', 'mixer_order', 'mixer_stable',
                               'mixer_sigma_min']
KEEP_DEFAULTS = dict(use_attention=False, num_heads=8, attn_last_k=0, attn_dist_bias=False,
                     noself_last_k=0, attn_logn_scale=False)  # defaults of the training-time module
SCALERS = ['lattice_scaler.pt', 'prop_scaler.pt', 'prop_scalers.pt']


def load_new(path):
    spec = importlib.util.spec_from_file_location('diffcsp.pl_modules.r2csp', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def use(mod):
    # hydra resolves the target through the parent package attribute, so set both
    import diffcsp.pl_modules
    sys.modules['diffcsp.pl_modules.r2csp'] = mod
    diffcsp.pl_modules.r2csp = mod


def check_flags(dec):
    bad = {k: dec.get(k) for k, v in REMOVED_OFF.items() if k in dec and dec.get(k) != v}
    if dec.get('attn_type', 'softmax') != 'softmax':
        bad['attn_type'] = dec.get('attn_type')
    return bad


def clean_decoder(dec):
    dec = dict(dec)
    for k in REMOVED:
        dec.pop(k, None)
    for k, v in KEEP_DEFAULTS.items():
        dec.setdefault(k, v)
    return dec


def strip_root(x):
    # resolved data paths of the training machine -> paths relative to the CrystalFlow root
    if isinstance(x, str):
        return re.sub(r'^/\S*?/CrystalFlow[^/]*/', '', x)
    if isinstance(x, (DictConfig, ListConfig)):
        return OmegaConf.create(strip_root(OmegaConf.to_container(x, resolve=False)))
    if isinstance(x, dict):
        return {k: strip_root(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(strip_root(v) for v in x)
    return x


def clean_hp(hp):
    hp = copy.deepcopy(hp)
    if isinstance(hp, DictConfig):
        hp = OmegaConf.to_container(hp, resolve=False)
    hp = strip_root(dict(hp))
    hp['decoder'] = clean_decoder(hp['decoder'])
    return hp


def batches(lattice_dim, max_n, seed=0):
    g = torch.Generator().manual_seed(seed)
    sizes = [[1, 2, 5, 13, 20], [3, 8, 17, max_n]] if max_n > 20 else [[1, 2, 5, 13, 20], [4, 9, 16, 20]]
    out = []
    for na in sizes:
        na = torch.tensor(na)
        B, N = len(na), int(na.sum())
        L = torch.diag_embed(4 + 4 * torch.rand(B, 3, generator=g)) + 0.5 * torch.randn(B, 3, 3, generator=g)
        rep = L.clone() if lattice_dim == 9 else torch.randn(B, lattice_dim, generator=g)
        out.append(dict(t_raw=torch.rand(B, generator=g), atom_types=torch.randint(1, 95, (N,), generator=g),
                        frac_coords=torch.rand(N, 3, generator=g), lattices_rep=rep, num_atoms=na,
                        node2graph=torch.repeat_interleave(torch.arange(B), na), lattices_mat=L))
    return out


def run_decoder(model, b):
    t = model.time_embedding(b['t_raw'])
    kw = {k: v for k, v in b.items() if k != 't_raw'}
    with torch.no_grad():
        return model.decoder(t=t, cemb=None, guide_indicator=None, **kw)


def init_check(cfg, new_mod):
    dec = cfg.model.decoder
    common = dict(type_encoding=None, latent_dim=cfg.model.latent_dim + cfg.model.time_dim, cemb_dim=1,
                  _recursive_=False)
    use(OLD)
    torch.manual_seed(0)
    a = hydra.utils.instantiate(dec, **common)
    ra = torch.rand(4)
    use(new_mod)
    torch.manual_seed(0)
    b = hydra.utils.instantiate(OmegaConf.create(clean_decoder(OmegaConf.to_container(dec, resolve=True))), **common)
    rb = torch.rand(4)
    assert type(a) is not type(b)
    sa, sb = a.state_dict(), b.state_dict()
    same = list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    return dict(init_equal=bool(same), rng_equal=bool(torch.equal(ra, rb)))


def process(name, run_dir, ckpt_name, out_root, new_mod):
    import eval_utils
    run_dir, out = Path(run_dir), Path(out_root) / name
    rep = dict(name=name, run_dir=str(run_dir), ckpt=ckpt_name)
    cfg = OmegaConf.load(run_dir / 'hparams.yaml')
    dec = cfg.model.decoder
    rep['decoder'] = {k: OmegaConf.to_container(dec, resolve=True).get(k) for k in KEEP_DEFAULTS}
    rep['bad_flags'] = check_flags(dec)
    if rep['bad_flags']:
        return rep

    # export: cleaned hparams.yaml, weights-only checkpoint, scalers
    out.mkdir(parents=True, exist_ok=True)
    ccfg = copy.deepcopy(cfg)
    ccfg.model.decoder = OmegaConf.create(clean_decoder(OmegaConf.to_container(dec, resolve=False)))
    OmegaConf.save(ccfg, out / 'hparams.yaml')
    ck = torch.load(run_dir / ckpt_name, map_location='cpu', weights_only=False)
    slim = {k: ck[k] for k in ('epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'hparams_name')
            if k in ck}
    slim['hyper_parameters'] = clean_hp(ck['hyper_parameters'])
    torch.save(slim, out / ckpt_name)
    for f in SCALERS:
        if (run_dir / f).exists():
            shutil.copy2(run_dir / f, out / f)
    rep.update(epoch=ck['epoch'], global_step=ck['global_step'])
    rep['no_abs_paths'] = re.search(r'/(home|data1?)/', repr(slim['hyper_parameters'])) is None

    # (1) original checkpoint, training-time module
    use(OLD)
    cls = hydra.utils.get_class(cfg.model._target_)
    m_old = cls.load_from_checkpoint(str(run_dir / ckpt_name), hparams_file=str(run_dir / 'hparams.yaml'),
                                     strict=True, map_location='cpu').eval()
    # (2) exported checkpoint, release module, the loader used by scripts/evaluate.py
    use(new_mod)
    m_new = eval_utils.load_model(out)[0].eval()
    m_new_strict = cls.load_from_checkpoint(str(out / ckpt_name), hparams_file=str(out / 'hparams.yaml'),
                                            strict=True, map_location='cpu')
    use(OLD)
    rep['decoder_class'] = type(m_new.decoder).__module__ + '.' + type(m_new.decoder).__name__
    rep['decoder_file'] = type(m_new.decoder).__init__.__code__.co_filename
    # (3) parameters and outputs
    sa, sb = m_old.state_dict(), m_new.state_dict()
    rep['n_params'] = sum(p.numel() for p in m_new.parameters())
    rep['keys_equal'] = list(sa) == list(sb) == list(m_new_strict.state_dict())
    rep['params_equal'] = rep['keys_equal'] and all(torch.equal(sa[k], sb[k]) for k in sa)
    md = 0.0
    eq = True
    ld = m_old.decoder.lattice_out.out_features
    max_n = int(cfg.data.get('max_atoms', 20))
    for b in batches(ld, max_n):
        oa, ob = run_decoder(m_old, b), run_decoder(m_new, b)
        for x, y in zip(oa, ob):
            eq &= torch.equal(x, y)
            md = max(md, (x - y).abs().max().item())
    rep['outputs_bit_equal'] = bool(eq)
    rep['outputs_max_abs_diff'] = md
    # (4) initialization
    rep.update(init_check(cfg, new_mod))
    use(OLD)
    rep['release_module_loaded'] = rep['decoder_file'] == new_mod.__file__ and \
        type(m_old.decoder).__init__.__code__.co_filename == OLD.__file__
    rep['ok'] = all(rep[k] for k in ('no_abs_paths', 'release_module_loaded', 'keys_equal', 'params_equal', 'outputs_bit_equal', 'init_equal', 'rng_equal'))
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release_module', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('runs', nargs='+', help='name=RUN_DIR:CKPT')
    a = p.parse_args()
    torch.set_num_threads(4)
    new_mod = load_new(a.release_module)
    reps = []
    for r in a.runs:
        name, rest = r.split('=', 1)
        run_dir, ckpt_name = rest.rsplit(':', 1)
        try:
            rep = process(name, run_dir, ckpt_name, a.out, new_mod)
        except Exception:
            rep = dict(name=name, error=traceback.format_exc())
        print(json.dumps(rep), flush=True)
        reps.append(rep)
        Path(a.report).write_text(json.dumps(reps, indent=1))
    print('ALL', sum(r.get('ok', False) for r in reps), '/', len(reps), 'ok', flush=True)


if __name__ == '__main__':
    main()
