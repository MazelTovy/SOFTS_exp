#!/bin/bash

# PatchTST with AML (Adaptive Mask Learning) on ETTm1 dataset
# This is the ablation version with only mask penalty (no embedding consistency penalty)

export CUDA_VISIBLE_DEVICES=0

python -u run_longExp.py \
  --random_seed 2021 \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id PatchTST_ETTm1_AML \
  --model PatchTST \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8 \
  --d_model 512 \
  --d_ff 2048 \
  --dropout 0.05 \
  --e_layers 2 \
  --d_layers 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 20 \
  --revin 1 \
  --des 'AML_Ablation' \
  --exp_name AML \
  --use_mask_penalty True \
  --mask_penalty_weight 0.1 \
  --use_emb_penalty False \
  --itr 1 