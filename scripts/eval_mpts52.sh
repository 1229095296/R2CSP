#!/usr/bin/env bash
# MPTS-52 test match rate of one checkpoint directory: 100 Euler steps, coordinate annealing slope
# 15 (main protocol) or 5. Writes per_struct_t_s<slope>_N100.csv into the directory. Run from the CrystalFlow root.
#   GPU=0 bash ../R2CSP/scripts/eval_mpts52.sh ckpt/mpts52_r2csp_s42 15
set -euo pipefail
R2CSP=$(cd "$(dirname "$0")/.." && pwd)
D=$1; S=${2:-15}; L=t_s${S}_N100
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONUNBUFFERED=1
python "$R2CSP/analysis/eval_sweep.py" -m "$D" --label "$L" --split test --max_n 0 -N 100 --slope "$S"
python "$R2CSP/analysis/per_struct_mr.py" --eval_dir "$D" --label "$L" --gt_file data/mpts_52/test.csv --njobs 16
