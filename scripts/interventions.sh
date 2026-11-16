#!/usr/bin/env bash
# The twelve inference-time interventions on the last-layer weights in t in [0.3, 0.7) (paired
# match rates and the dose--response figure) and the flow-matching loss under each of them (relative change
# of the aggregate), followed by the dose--response fit. Run from the CrystalFlow root.
#   GPU=0 OUT=results/s123 bash ../R2CSP/scripts/interventions.sh ckpt/mp20_r2csp_s123 s123
# The paper uses mp20_r2csp_s42_ep2421 (tag s42) and mp20_r2csp_s123 (tag s123).
# Other windows: change T_LO / T_HI, e.g. T_LO=0.7 T_HI=1.01 for the late window.
set -uo pipefail
R2CSP=$(cd "$(dirname "$0")/.." && pwd)
M=$1; TAG=$2; OUT=${OUT:-results/$TAG}; T_LO=${T_LO:-0.3}; T_HI=${T_HI:-0.7}
A=$R2CSP/analysis
mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONUNBUFFERED=1
CONDS="ref:ref:0 uniform:uniform:0 selfkeep:selfkeep:0 selfflat:selfflat:0 normprojN:normprojN:0 residN:residN:0
shuffle:shuffle:0 bias1:bias:1 bias2:bias:2 bias4:bias:4 biasm2:bias:-2 shelloracle:shelloracle:0 oracle:oracle:0"
ARGS=""
for c in $CONDS; do
  IFS=: read -r name mode lam <<< "$c"
  l=d${TAG}_$name
  if [ ! -f "$M/per_struct_$l.csv" ]; then
    echo "START $l $(date)"
    python "$A/eval_intervene_shell.py" -m "$M" --label "$l" --mode "$mode" --lam "$lam" \
      --layers 5 --t_lo "$T_LO" --t_hi "$T_HI" --seed 0 > "$OUT/logs/$l.log" 2>&1 || { echo "FAIL $l"; continue; }
    python "$A/per_struct_mr.py" --eval_dir "$M" --label "$l" --gt_file data/mp_20/test.csv --njobs 16 \
      >> "$OUT/logs/$l.log" 2>&1 || echo "FAIL per_struct $l"
  fi
  ARGS="$ARGS $name=$M/per_struct_$l.csv"
done
python "$A/paired_stats.py" --out "$OUT/dose_${TAG}_paired.json" $ARGS --ref ref
python "$A/eval_fmloss.py" -m "$M" --out "$OUT/fm_${TAG}.json" --grad > "$OUT/logs/fm.log" 2>&1
python "$A/fit_dose.py" --fm "$OUT/fm_${TAG}.json" --paired "$OUT/dose_${TAG}_paired.json" --out "$OUT/dose_fit_${TAG}.json"
