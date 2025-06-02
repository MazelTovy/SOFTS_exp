#!/bin/bash

# TimeMixer with AMRC (Adaptive Mask with Representation Consistency) on ETTm1 dataset
# This runs the full AMRC method with both mask penalty and embedding consistency penalty

export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id TimeMixer_ETTm1_AMRC \
  --model TimeMixer \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --d_model 16 \
  --d_ff 32 \
  --dropout 0.1 \
  --enc_in 7 \
  --c_out 7 \
  --itr 1 \
  --batch_size 32 \
  --des 'AMRC_Full' \
  --exp_name AMRC \
  --use_mask_penalty True \
  --mask_penalty_weight 0.1 \
  --use_emb_penalty True \
  --emb_penalty_weight 1.0 \
  --learning_rate 0.001 