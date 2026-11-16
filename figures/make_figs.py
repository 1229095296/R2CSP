"""Vector PDF figures of the paper (IEEE two-column: column 3.5 in, page 7.16 in).

  python make_figs.py --data <figure_inputs dir of the results release asset> --out <dir>
Inputs: dose_fit_s42_s123.json ({'s42': ..., 's123': ...}, each an output of analysis/fit_dose.py),
mpts52_2seed.json (two-seed MPTS-52 match rates by cell size, at slopes 15 and 5),
mpts_csv/{off,k1}_s{42,123}_s15.csv (per-structure MPTS-52 results, mean aggregation and R2CSPNet),
gain_dump_k1s42.npz (analysis/dump_gain.py on mp20_r2csp_s42_ep2421, 4 test batches),
geo_off_last.json, geo_ref_k1.json (analysis/eval_intervene_shell.py --geo, mode ref),
gate_k1G.json (analysis/probe_gate.py), lookahead_k1s42.json (summary of analysis/probe_lookahead.py).
"""
import argparse, csv, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'lines.linewidth': 1.0,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
# Okabe-Ito
C_MEAN, C_R2, C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_GREY = (
    '#7F7F7F', '#D55E00', '#0072B2', '#E69F00', '#009E73', '#CC79A7', '#BBBBBB')
BINS = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 52)]


def by_bin(path):
    rows = list(csv.DictReader(open(path)))
    out = []
    for lo, hi in BINS:
        s = [r for r in rows if lo <= int(r['n_atoms']) <= hi]
        out.append(100.0 * sum(r['match'] == 'True' for r in s) / len(s))
    return np.array(out)


def fig_pern(data, out):
    j = json.load(open(data / 'mpts52_2seed.json'))['s15']
    g42 = np.array([j['s42']['byN'][f'{lo}-{hi}'] for lo, hi in BINS])
    g123 = np.array([j['s123']['byN'][f'{lo}-{hi}'] for lo, hi in BINS])
    base = np.mean([by_bin(data / f'mpts_csv/off_s{s}_s15.csv') for s in (42, 123)], axis=0)
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    x = np.arange(len(BINS)); w = 0.36
    ax.bar(x - w / 2, g42, w, color=C_R2, label='seed 42', zorder=2)
    ax.bar(x + w / 2, g123, w, color=C_R2, alpha=0.45, label='seed 123', zorder=2)
    ax.plot(x, (g42 + g123) / 2, 'D', color='k', ms=3, label='mean of seeds', zorder=3)
    top = max(g42.max(), g123.max())
    ax.set_xticks(x); ax.set_xticklabels([f'{lo}–{hi}\n({b:.1f}%)' for (lo, hi), b in zip(BINS, base)])
    ax.set_xlabel('Atoms per cell (MR of the mean model)'); ax.set_ylabel(r'$\Delta$MR (points)')
    ax.set_ylim(0, top + 0.5); ax.set_xlim(-0.65, len(BINS) - 0.35)
    ax.axhline(0, color='k', lw=0.6)
    ax.grid(axis='y', lw=0.3, color='#DDDDDD', zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(loc='upper right', frameon=False, ncol=1, handlelength=1.2, labelspacing=0.25)
    fig.savefig(out / 'fig_pern.pdf')
    plt.close(fig)


GROUPS = {  # family -> (colour, marker, modes)
    'flatten': (C_BLUE, 'o', ['uniform', 'selfkeep', 'selfflat']),
    'norm projection': (C_GREEN, 's', ['normprojN', 'residN']),
    'shuffle': (C_PURPLE, 'v', ['shuffle']),
    r'shell bias $\pm\lambda$': (C_ORANGE, '^', ['bias:1', 'bias:2', 'bias:4', 'bias:-2']),
    'true shell': (C_R2, 'D', ['shelloracle', 'oracle']),
}
LABEL = {'uniform': 'uniform', 'oracle': 'oracle', 'shelloracle': 'shelloracle',
         'bias:4': r'$\lambda{=}4$', 'residN': 'orth. to norm', 'selfflat': 'self at $1/N$'}


def fig_dose(data, out):
    d = json.load(open(data / 'dose_fit_s42_s123.json'))
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    xs = np.linspace(0, 1.08, 200)
    a, c = -d['s42']['a'], d['s42']['c']
    ax.plot(xs, -a * xs ** 2 + c, color='k', lw=0.9, zorder=1,
            label=rf'fit, seed 42: $-{a:.2f}\,x^2{c:+.2f}$')
    for fam, (col, mk, modes) in GROUPS.items():
        for k, (seed, fill) in enumerate((('s42', True), ('s123', False))):
            m = d[seed]['modes']
            idx = [m.index(q) for q in modes]
            xv = np.array(d[seed]['x'])[idx]; yv = np.array(d[seed]['y'])[idx]
            ax.scatter(xv, yv, s=16, marker=mk, zorder=3, linewidths=0.8,
                       facecolors=col if fill else 'white', edgecolors=col,
                       label=fam if k == 0 else None)
    m = d['s42']['modes']
    off = {'uniform': (5, -3), 'oracle': (-6, -6), 'shelloracle': (6, -1), 'bias:4': (6, 0)}
    for q, (dx, dy) in off.items():
        i = m.index(q)
        ax.annotate(LABEL[q], (d['s42']['x'][i], d['s42']['y'][i]), xytext=(dx, dy),
                    textcoords='offset points', fontsize=6, ha='left' if dx >= 0 else 'right', color='#333333')
    ax.set_xlabel(r'$x=\|\delta m\|/\|m\|$ (relative change of the aggregate)')
    ax.set_ylabel(r'$\Delta$MR (points)')
    ax.set_xlim(0, 1.08); ax.set_ylim(-3.25, 0.25)
    ax.axhline(0, color='k', lw=0.4, ls=':')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    h.append(plt.Line2D([], [], ls='', marker='o', mfc='white', mec='k', ms=4)); l.append('open: seed 123')
    ax.legend(h, l, loc='lower left', frameon=False, ncol=1, handletextpad=0.3, borderaxespad=0.1,
              labelspacing=0.2, bbox_to_anchor=(0.0, -0.02))
    fig.savefig(out / 'fig_dose.pdf')
    plt.close(fig)


def _clean(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def fig_norm(data, out, nbin='17-20'):
    """Last-layer message statistics along the sampling trajectory, cells of 17-20 atoms."""
    off = json.load(open(data / 'geo_off_last.json'))['geo']
    k1 = json.load(open(data / 'geo_ref_k1.json'))['geo']
    steps = np.arange(5, 96, 5); t = steps / 100
    get = lambda d, key: np.array([d[f'5_{s}_{nbin}'][key] for s in steps])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 1.65))
    a1.plot(t, get(off, 'corr_logn_ld'), color=C_MEAN, label=r'mean model, $\log\|e_{ij}\|$')
    a1.plot(t, get(k1, 'corr_logn_ld'), color=C_R2, ls='--', label=r'R2CSPNet, $\log\|e_{ij}\|$')
    a1.plot(t, get(k1, 'corr_logit_ld'), color=C_R2, label=r'R2CSPNet, logit $z_{ij}$')
    a1.axhline(0, color='k', lw=0.4, ls=':')
    a1.set_ylim(-0.85, 1.0); a1.set_xlim(0, 0.95); a1.set_yticks([-0.5, 0, 0.5])
    a1.set_xlabel('flow time $t$'); a1.set_ylabel(r'corr. with $\log d_{ij}$')
    a1.set_title('(a) what carries distance', fontsize=8)
    a2.plot(t, get(off, 'sd_logn_rest'), color=C_MEAN, label='mean model')
    a2.plot(t, get(k1, 'sd_logn_rest'), color=C_R2, ls='--', label='R2CSPNet')
    a2.set_ylim(0, 0.85); a2.set_xlim(0, 0.95)
    a2.set_xlabel('flow time $t$'); a2.set_ylabel(r's.d. of $\log\|e_{ij}\|$')
    a2.set_title('(b) spread of message norms', fontsize=8)
    for ax in (a1, a2):
        _clean(ax)
    a1.legend(loc='upper left', frameon=False, handlelength=1.4, labelspacing=0.2, fontsize=6,
              borderaxespad=0.1)
    a2.legend(loc='upper left', frameon=False, handlelength=1.4, labelspacing=0.2, fontsize=6)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(out / 'fig_norm.pdf')
    plt.close(fig)


def fig_time(data, out):
    """First-shell ranking along training interpolants, and the opening of the distance gate."""
    L = json.load(open(data / 'lookahead_k1s42.json'))
    G = json.load(open(data / 'gate_k1G.json'))['stats']
    tg = np.round(np.arange(0.1, 0.95, 0.1), 1)
    gate = []
    for tt in tg:
        rows = [G[f'L0|{tt:g}|{b}'] for b in range(5)]
        gate.append(sum(r['g'] * r['n'] for r in rows) / sum(r['n'] for r in rows))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 1.65))
    t = np.array(L['t'])
    a1.plot(t, L['auc_node_dt'], '-o', ms=2.5, color=C_BLUE, label=r'current distance $d_{ij}$')
    a1.plot(t, L['auc_node_attn'], '-s', ms=2.5, color=C_R2, label='learned weight')
    a1.axhline(0.5, color='k', lw=0.4, ls=':')
    a1.text(0.97, 0.505, 'chance', ha='right', va='bottom', fontsize=6, color='#555555')
    a1.set_ylim(0.3, 1.0); a1.set_xlim(0.05, 0.95)
    a1.set_xlabel('flow time $t$'); a1.set_ylabel('first-shell AUROC')
    a1.set_title('(a) ranking the first shell', fontsize=8)
    a1.legend(loc='upper left', frameon=False, handlelength=1.4, labelspacing=0.2, fontsize=6)
    a2.plot(tg, gate, '-o', ms=2.5, color=C_GREEN)
    a2.axhline(0, color='k', lw=0.4, ls=':')
    a2.set_xlim(0.05, 0.95); a2.set_ylim(-0.3, 6.5)
    a2.set_xlabel('flow time $t$'); a2.set_ylabel(r'gate $g(h_i)$')
    a2.set_title('(b) learned distance gate', fontsize=8)
    for ax in (a1, a2):
        _clean(ax); ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    fig.tight_layout(w_pad=0.8)
    fig.savefig(out / 'fig_time.pdf')
    plt.close(fig)


def graph_panel(ax, gains, self_gain, eq1, eq2):
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.72, 1.05); ax.set_aspect('equal'); ax.axis('off')
    ang = np.deg2rad([90, 150, 210, 270, 330, 30])
    rad = np.array([0.72, 0.78, 1.12, 0.70, 1.15, 1.08])
    pos = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    for p, g in zip(pos, gains):
        ax.add_patch(FancyArrowPatch(p, (0, 0), arrowstyle='-|>', mutation_scale=6,
                                     lw=0.4 + 1.5 * g, color=C_BLUE if self_gain is None else C_R2,
                                     shrinkA=6.5, shrinkB=7.5, zorder=1))
    for p in pos:
        ax.add_patch(Circle(p, 0.15, fc='#F2F2F2', ec='k', lw=0.6, zorder=2))
    ax.add_patch(Circle((0, 0), 0.19, fc='#FFE3CC' if self_gain is not None else '#DCEBF7', ec='k',
                        lw=0.8, zorder=2))
    ax.text(0, 0, '$i$', ha='center', va='center', fontsize=8, zorder=3)
    sg = 1.0 if self_gain is None else self_gain
    loop = FancyArrowPatch((0.15, 0.12), (0.15, -0.12), connectionstyle='arc3,rad=-2.4',
                           arrowstyle='-|>', mutation_scale=5, lw=0.4 + 1.5 * sg,
                           color=(C_BLUE if self_gain is None else C_GREY),
                           ls='-' if self_gain is None else (0, (2, 1.5)), zorder=1)
    ax.add_patch(loop)
    ax.text(0.62, 0.0, 'self' + ('' if self_gain is None else '\n' + r'$\approx0$'),
            ha='left', va='center', fontsize=7, color='#555555')
    ax.text(0, -1.33, eq1, ha='center', va='center', fontsize=8.5)
    ax.text(0, -1.66, eq2, ha='center', va='center', fontsize=7.5, color='#444444')


def fig_overview(data, out):
    d = np.load(data / 'gain_dump_k1s42.npz')
    fig = plt.figure(figsize=(7.16, 2.3))
    ax1 = fig.add_axes([0.0, 0.0, 0.27, 0.88])
    ax2 = fig.add_axes([0.29, 0.0, 0.27, 0.88])
    ax3 = fig.add_axes([0.665, 0.2, 0.325, 0.68])
    for x, t in ((0.135, '(a) Mean aggregation, layers 1–5'), (0.425, '(b) R2CSPNet, last layer'),
                 (0.8275, '(c) Learned gain, 9–20 atoms')):
        fig.text(x, 0.985, t, ha='center', va='top', fontsize=8.5, weight='bold')
    graph_panel(ax1, [0.5] * 6, None,
                r'$m_i=\frac{1}{N}\sum_{j}e_{ij}$', 'every edge has gain $N\\alpha_{ij}=1$')
    graph_panel(ax2, [1.0, 0.95, 0.3, 1.05, 0.25, 0.4], 0.0,
                r'$\alpha_{ij}=\mathrm{softmax}_j(w^{\top}\!e_{ij}),\;\; m_i=\sum_{j}\alpha_{ij}e_{ij}$',
                'same $e_{ij}$; only the gain $N\\alpha_{ij}$ is learned')
    fig.add_artist(FancyArrowPatch((0.268, 0.55), (0.298, 0.55), transform=fig.transFigure,
                                   arrowstyle='-|>', mutation_scale=8, lw=0.8, color='k'))
    # (c) learned gain vs current relative distance, non-self edges of cells with 9-20 atoms
    edges = np.array([1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0])
    ctr = np.sqrt(edges[:-1] * edges[1:])
    for st, col, lab in ((50, C_BLUE, '$t=0.5$'), (90, C_R2, '$t=0.9$')):
        m = (d['step'] == st) & ~d['self'] & (d['N'] >= 9)
        dr, g = d['drel'][m], d['gain'][m]
        med, q1, q3 = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = g[(dr >= lo) & (dr < hi)]
            med.append(np.median(s)); q1.append(np.percentile(s, 25)); q3.append(np.percentile(s, 75))
        ax3.fill_between(ctr, q1, q3, color=col, alpha=0.12, lw=0)
        ax3.plot(ctr, med, '-o', color=col, ms=2.5, label=lab)
    ms = (d['step'] == 90) & d['self'] & (d['N'] >= 9)
    ax3.plot([0.8], [np.median(d['gain'][ms])], marker='*', color='k', ms=6, ls='', clip_on=False,
             label='self edge')
    ax3.axhline(1.0, color='k', lw=0.6, ls=':')
    ax3.text(5.9, 1.04, 'mean', ha='right', va='bottom', fontsize=7, color='#555555')
    ax3.set_xscale('log'); ax3.set_xlim(0.72, 6.2)
    ax3.set_xticks([0.8, 1, 1.5, 2, 3, 4, 6]); ax3.set_xticklabels(['self', '1', '1.5', '2', '3', '4', '6'])
    ax3.minorticks_off()
    ax3.set_ylim(-0.05, 2.0)
    ax3.set_xlabel(r'distance / nearest-neighbour distance, $d_{ij}/d_i^{\mathrm{nn}}$')
    ax3.set_ylabel(r'learned gain $N\alpha_{ij}$')
    for s in ('top', 'right'):
        ax3.spines[s].set_visible(False)
    ax3.legend(loc='upper right', frameon=False, handlelength=1.2, labelspacing=0.25)
    fig.savefig(out / 'fig_overview.pdf')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    data, out = Path(a.data), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig_pern(data, out); fig_dose(data, out); fig_overview(data, out)
    fig_norm(data, out); fig_time(data, out)
    print('ok')
