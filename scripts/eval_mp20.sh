#!/usr/bin/env bash
# MP-20 test match rate of one checkpoint directory: 100 Euler steps, coordinate annealing slope 5.
# Writes eval_metrics_<label>.json and per_struct_<label>.csv into the directory. Run from the CrystalFlow root.
#   GPU=0 bash ../R2CSP/scripts/eval_mp20.sh ckpt/mp20_r2csp_s42 last
set -euo pipefail
R2CSP=$(cd "$(dirname "$0")/.." && pwd)
D=$1; L=${2:-last}; SLOPE=${SLOPE:-5}
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONUNBUFFERED=1
python scripts/evaluate.py -m "$D" --dataset mp_20 --ode-int-steps 100 --anneal_coords --anneal_slope "$SLOPE" --label "$L"
python scripts/compute_metrics.py --root_path "$D" --tasks csp --gt_file data/mp_20/test.csv --njobs 16 --label "$L"
python "$R2CSP/analysis/per_struct_mr.py" --eval_dir "$D" --label "$L" --gt_file data/mp_20/test.csv --njobs 16
cat "$D/eval_metrics_$L.json"; echo
