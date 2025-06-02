#!/bin/bash

# TimeMixer with AML (Adaptive Mask Learning) on ETTm1 dataset
# This is the ablation version with only mask penalty (no embedding consistency penalty)

export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id TimeMixer_ETTm1_AML \
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
  --des 'AML_Ablation' \
  --exp_name AML \
  --use_mask_penalty True \
  --mask_penalty_weight 0.1 \
  --use_emb_penalty False \
  --learning_rate 0.001 