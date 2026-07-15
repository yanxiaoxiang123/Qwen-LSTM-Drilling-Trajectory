#!/usr/bin/env bash
set -euo pipefail

: "${QWEN_FINETUNED_DIR:?Set QWEN_FINETUNED_DIR to the merged fine-tuned Qwen3.5-9B checkpoint}"

python src/train_online_rolling_h5.py \
  --runs-dir runs_paper \
  --run-name online_16A_inc_qwen_lstm_finetuned_h5 \
  --well 16A \
  --target inclination_deg \
  --model qwen_lstm \
  --text-mode finetuned \
  --qwen-model-dir "$QWEN_FINETUNED_DIR" \
  --data-variant depth1ft_clean \
  --seq-len 50 \
  --horizon 5 \
  --cold-start-ratio 0.10 \
  --target-scale-mode physical \
  --initial-epochs 40 \
  --online-epochs 1 \
  --online-window-labels 300 \
  --max-update-windows 256 \
  --batch-size 16 \
  --hidden-size 128 \
  --layers 2 \
  --text-dim 128 \
  --dropout 0.15 \
  --lr 5e-4 \
  --weight-decay 1e-4

