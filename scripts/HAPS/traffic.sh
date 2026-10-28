#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ROOT_PATH="${ROOT_PATH:-./dataset/traffic/}"
DATA_PATH="${DATA_PATH:-traffic.csv}"
USE_AMP="${USE_AMP:-1}"

AMP_ARGS=()
if [[ "$USE_AMP" == "1" ]]; then
    AMP_ARGS+=(--use_amp)
fi

if [[ "$#" -gt 0 ]]; then
    HORIZONS=("$@")
else
    HORIZONS=(96 192 336 720)
fi

for H in "${HORIZONS[@]}"; do
    case "$H" in
        96)
            D_FF=2048
            DROPOUT=0.1
            HAPS_RANK=32
            ;;
        192)
            D_FF=2048
            DROPOUT=0.2
            HAPS_RANK=64
            ;;
        336)
            D_FF=2048
            DROPOUT=0.3
            HAPS_RANK=32
            ;;
        720)
            D_FF=1024
            DROPOUT=0.2
            HAPS_RANK=64
            ;;
        *)
            echo "Unsupported prediction horizon: $H"
            echo "Supported horizons: 96, 192, 336, 720"
            exit 1
            ;;
    esac

    echo "============================================================"
    echo "Traffic | H=$H | d_ff=$D_FF | dropout=$DROPOUT | rank=$HAPS_RANK"
    echo "============================================================"

    python -u run_haps.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "$ROOT_PATH" \
      --data_path "$DATA_PATH" \
      --model_id "traffic_96_${H}_haps" \
      --model haps \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len "$H" \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --d_model 512 \
      --d_ff "$D_FF" \
      --dropout "$DROPOUT" \
      --learning_rate 5e-3 \
      --train_epochs 20 \
      --patience 5 \
      --batch_size 16 \
      --itr 1 \
      "${AMP_ARGS[@]}" \
      --haps_variant full \
      --haps_rank "$HAPS_RANK" \
      --haps_n_bands 4 \
      --haps_lambda_pred 0.20 \
      --haps_lambda_dyn 0.10 \
      --haps_lambda_rank 1e-3 \
      --haps_lambda_gate 1e-3 \
      --haps_lambda_innov 1e-4 \
      --lradj type3

done
