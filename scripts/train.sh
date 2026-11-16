#!/usr/bin/env bash
# Train R2CSPNet or the mean-aggregation network on MP-20 or MPTS-52. Run from the CrystalFlow root.
#   GPUS=0 bash ../R2CSP/scripts/train.sh mp_20 r2csp 42       # R2CSPNet, MP-20, seed 42
#   GPUS=0,1 bash ../R2CSP/scripts/train.sh mpts_52 mean 123   # mean aggregation, MPTS-52, seed 123
# Extra arguments are passed to diffcsp/run.py, e.g. the ablations of Table II:
#   model.decoder.attn_last_k=2                        (gain in the last two layers)
#   model.decoder.attn_last_k=0 model.decoder.num_heads=8   (every layer, eight heads)
#   model.decoder.use_attention=false model.decoder.noself_last_k=1   (mean without the self edge)
#   model.decoder.attn_dist_bias=true                  (explicit distance term)
#   model.decoder.attn_logn_scale=true                 (logit times log N)
# Global batch: 256 on MP-20, 128 on MPTS-52, split evenly over the GPUs.
set -eo pipefail
R2CSP=$(cd "$(dirname "$0")/.." && pwd)
[ $# -ge 3 ] || { echo "usage: train.sh mp_20|mpts_52 r2csp|mean SEED [overrides]"; exit 1; }
DATA=$1; MODEL=$2; SEED=$3; shift 3
[ -n "$GPUS" ] || { echo "set GPUS, e.g. GPUS=0"; exit 1; }
case $DATA in
  mp_20)   GLOBAL_BS=256; EPOCHS=3000 ;;
  mpts_52) GLOBAL_BS=128; EPOCHS=2000 ;;
  *) echo "dataset must be mp_20 or mpts_52"; exit 1 ;;
esac
case $MODEL in
  r2csp) DEC="model.decoder.use_attention=true model.decoder.num_heads=1 model.decoder.attn_last_k=1" ;;
  mean)  DEC="model.decoder.use_attention=false" ;;
  *) echo "model must be r2csp or mean"; exit 1 ;;
esac
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l | tr -d ' ')
BS=$((GLOBAL_BS / NGPU))
if [ "$NGPU" -gt 1 ]; then
  DIST="train.pl_trainer.devices=$NGPU +train.pl_trainer.strategy=ddp_find_unused_parameters_true"
else
  DIST="train.pl_trainer.devices=1"
fi
EXPNAME=${EXPNAME:-R2CSP-$DATA-$MODEL-s$SEED}

# build the dataset caches once, before any DDP launch
for s in train val test; do
  [ -f data/$DATA/${s}_ori.pt ] || { python "$R2CSP/scripts/preprocess.py" "$DATA"; break; }
done

mkdir -p log
CUDA_VISIBLE_DEVICES=$GPUS HYDRA_FULL_ERROR=1 python diffcsp/run.py \
  data=$DATA data.train_max_epochs=$EPOCHS \
  data.datamodule.batch_size.train=$BS \
  model=flow_polar \
  model/decoder=r2csp \
  optim.optimizer.lr=1e-3 \
  optim.optimizer.weight_decay=0 \
  optim.lr_scheduler.factor=0.6 \
  +model.lattice_polar_sigma=0.1 \
  model.cost_coord=10 model.cost_lattice=1 \
  model.decoder.num_freqs=256 \
  model.decoder.rec_emb=sin model.decoder.num_millers=8 \
  +model.decoder.na_emb=0 \
  model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
  train.random_seed=$SEED \
  train.model_checkpoints.save_last=true \
  +train.model_checkpoints.every_n_epochs=500 \
  $DIST $DEC \
  logging.wandb.mode=offline \
  expname=$EXPNAME \
  "$@" \
  2>&1 | tee "log/$EXPNAME.log"
