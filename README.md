# R2CSPNet

Code for *Last-Layer Attention as Normalized Message Gain in Flow-Matching Crystal Structure Prediction*.

R2CSPNet is the CSPNet decoder of [CrystalFlow](https://github.com/ixsluo/CrystalFlow) with one change in the
last message-passing layer: atom i aggregates its edge messages e_ij with weights
alpha_ij = softmax_j(w^T e_ij) instead of the mean. w is initialized at zero, so training starts from mean
aggregation; the change adds 512 parameters.

```
r2csp/r2csp.py            R2CSPNet (subclass of CrystalFlow's CSPNet)
conf/r2csp.yaml           decoder config (all options off = cspnet.yaml)
patches/crystalflow.patch small fixes to CrystalFlow used for training and evaluation
scripts/                  training, evaluation, interventions, dataset preprocessing
analysis/                 per-structure match, paired tests, interventions, probes, dose-response fit
figures/make_figs.py      the figures of the paper
tests/test_r2csp.py       unit tests against CSPNet
tools/export_verify.py    export of the released checkpoints and their bit-exact check
```

## Installation

R2CSPNet runs inside a CrystalFlow checkout. We used Python 3.11.9, PyTorch 2.3.1 (CUDA 12.1),
torch_geometric 2.5.3 and lightning 2.3.2.

```bash
git clone https://github.com/ixsluo/CrystalFlow.git
git clone https://github.com/1229095296/R2CSP.git
cd CrystalFlow
git apply ../R2CSP/patches/crystalflow.patch
cp ../R2CSP/r2csp/r2csp.py diffcsp/pl_modules/r2csp.py
cp ../R2CSP/conf/r2csp.yaml conf/model/decoder/r2csp.yaml
# environment as in the CrystalFlow README, then
pip install -e .
python ../R2CSP/tests/test_r2csp.py      # or: python -m pytest ../R2CSP/tests/test_r2csp.py
```

Copy `.env.template` to `.env` and set `PROJECT_ROOT` and `HYDRA_JOBS`. MP-20 and MPTS-52 are in `data/`
of CrystalFlow. All commands below run from the CrystalFlow root.

## Training

```bash
GPUS=0 bash ../R2CSP/scripts/train.sh mp_20 r2csp 42      # R2CSPNet, MP-20, seed 42
GPUS=0 bash ../R2CSP/scripts/train.sh mp_20 mean 42       # the same network with mean aggregation
GPUS=0 bash ../R2CSP/scripts/train.sh mpts_52 r2csp 123
```

MP-20 trains for 3000 epochs with a global batch of 256, MPTS-52 for 2000 epochs with 128. The seeds of the
paper are 42, 123, 2026 and 7 on MP-20 and 42 and 123 on MPTS-52. The ablations of Table III are overrides
of the decoder, listed at the top of `scripts/train.sh`.

## Evaluation

Download the checkpoint assets (below) and unzip them in the CrystalFlow root; they unpack into `ckpt/`. Each directory holds `hparams.yaml`, a
weights-only checkpoint and the scalers, and loads with CrystalFlow's `scripts/evaluate.py`.

```bash
GPU=0 bash ../R2CSP/scripts/eval_mp20.sh ckpt/mp20_r2csp_s42 last       # Table I, MP-20 (s = 5)
GPU=0 bash ../R2CSP/scripts/eval_mpts52.sh ckpt/mpts52_r2csp_s42 15     # Table I, MPTS-52 (s = 15)
python ../R2CSP/analysis/paired_stats.py --out paired.json \
  r2csp=ckpt/mp20_r2csp_s42/per_struct_last.csv mean=ckpt/mp20_mean_s42/per_struct_last.csv  # McNemar
```

Both write a per-structure CSV next to the checkpoint. Sampling uses 100 Euler steps and one sample per test
composition.

## Interventions and figures

```bash
GPU=0 OUT=results/s42 bash ../R2CSP/scripts/interventions.sh ckpt/mp20_r2csp_s42_ep2421 s42
GPU=0 OUT=results/s123 bash ../R2CSP/scripts/interventions.sh ckpt/mp20_r2csp_s123 s123
python ../R2CSP/figures/make_figs.py --data r2csp_results/figure_inputs --out figs
```

`interventions.sh` runs the twelve interventions on the last-layer weights in t in [0.3, 0.7), the paired
tests, the flow-matching loss under each intervention and the dose-response fit. The probes of the
first-shell analysis are `analysis/probe_lookahead.py`, `analysis/probe_gate.py` and `analysis/dump_gain.py`;
their usage is in each file.

## Released checkpoints and results

The [Releases](https://github.com/1229095296/R2CSP/releases) hold:

| asset | size | contents | sha256 |
|---|---|---|---|
| `r2csp_ckpt_mp20_main.zip` | 623 MB | Table I, MP-20: R2CSPNet and mean model, seeds 42, 123, 2026, 7 (last checkpoint) | `56b1e57b3fb02d46ea48d320c397deaf3f45f4e4a25dd21c3d6b1150bde29d75` |
| `r2csp_ckpt_mpts52.zip` | 390 MB | Table I, MPTS-52: R2CSPNet and mean model, seeds 42, 123; R2CSPNet with log N scaling, seed 42 | `6b318ae020ff74531ffa6b065ad845ac2ec1da17cad6ea2a43fb7bdba270aa6d` |
| `r2csp_ckpt_mp20_ablations.zip` | 546 MB | Table III: no self edge, last 2/3/4 layers, all 6 layers, all 6 layers with 8 heads, distance gate (seed 42) | `86d7e5585a9c3506c4a08634e13c2c616998d9bf743a1b95efd0a4f220d69a4b` |
| `r2csp_ckpt_diagnostics.zip` | 78 MB | R2CSPNet MP-20 seed 42 at epoch 2421, used by the interventions and probes | `88bcd2637f96cabff1bc91015b305b8e584e9a7b457c7e448b905b9af0d3647c` |
| `r2csp_results.zip` | 14 MB | per-structure results, paired tests, intervention outputs, figure inputs, checkpoint verification | `9dc05688d503338d4761c665f9a23108d93818533b0ce0987d42f15f315a6fd1` |

`r2csp_results.zip` contains the per-structure results of every run, the paired tests, the outputs of the
interventions and the inputs of every figure; its `MANIFEST.md` maps each file to the paper.
Check the downloads with `shasum -a 256 -c SHA256SUMS`.

## Citation

```bibtex
@article{lin2026r2csp,
  title   = {Last-Layer Attention as Normalized Message Gain in Flow-Matching Crystal Structure Prediction},
  author  = {Lin, Zihan and Fang, Zhi and Sun, Jiayang and Cao, Jie and Huang, Huaibo and Li, Linjing and Duan, Junxian and He, Ran},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`. R2CSPNet builds on CrystalFlow and DiffCSP, both MIT.
