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

ALL_VARIANTS=(
    full
    no_subspace
    static_spectral
    no_spectral
    no_innovation
    fixed_rank
    generic_decoder
)

if [[ "$#" -gt 0 ]]; then
    VARIANTS=("$@")
else
    VARIANTS=("${ALL_VARIANTS[@]}")
fi

is_valid_variant() {
    local requested="$1"
    local valid
    for valid in "${ALL_VARIANTS[@]}"; do
        if [[ "$requested" == "$valid" ]]; then
            return 0
        fi
    done
    return 1
}

for VARIANT in "${VARIANTS[@]}"; do
    if ! is_valid_variant "$VARIANT"; then
        echo "Unknown variant: $VARIANT"
        echo "Valid variants: ${ALL_VARIANTS[*]}"
        exit 1
    fi

    echo "============================================================"
    echo "Traffic ablation | H=720 | variant=$VARIANT"
    echo "============================================================"

    python -u run_haps.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "$ROOT_PATH" \
      --data_path "$DATA_PATH" \
      --model_id "traffic_96_720_${VARIANT}" \
      --model haps \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 720 \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --d_model 512 \
      --d_ff 1024 \
      --dropout 0.2 \
      --learning_rate 5e-3 \
      --train_epochs 20 \
      --patience 5 \
      --batch_size 16 \
      --itr 1 \
      "${AMP_ARGS[@]}" \
      --haps_variant "$VARIANT" \
      --haps_rank 64 \
      --haps_n_bands 4 \
      --haps_lambda_pred 0.20 \
      --haps_lambda_dyn 0.10 \
      --haps_lambda_rank 1e-3 \
      --haps_lambda_gate 1e-3 \
      --haps_lambda_innov 1e-4 \
      --lradj type3

done
