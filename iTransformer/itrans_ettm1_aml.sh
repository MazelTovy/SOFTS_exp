#!/bin/bash

# Example script for running iTransformer with AML (Adaptive Mask Learning)
# This is the ablation version of AMRC with only mask penalty (no embedding consistency penalty)
export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
seq_len=96
pred_len=96

for pred_len in 96 192 336 720
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm1.csv \
      --model_id ETTm1_$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'AML' \
      --d_model 128 \
      --d_ff 128 \
      --itr 1 \
      --exp_name 'AML' \
      --use_mask_penalty True \
      --mask_penalty_weight 0.1 \
      --use_emb_penalty False
done 