if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=48

for pred_len in 48
do
    python -u run_longExp.py \
      --random_seed 2021 \
      --is_training 1 \
      --root_path ./dataset/Solar/ \
      --data_path solar_AL.txt \
      --model_id solar_$seq_len'_'$pred_len \
      --model PatchTST \
      --data Solar \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 137 \
      --e_layers 2 \
      --enc_in 137 \
      --dec_in 137 \
      --c_out 137 \
      --des 'Exp' \
      --d_model 512 \
      --d_ff 512 \
      --target 'OT' \
      --learning_rate 0.0005 \
      --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/PatchTST_solar_$seq_len'_'$pred_len.log 
done

# for pred_len in 48 72 96 120 144 192
# do
#     python -u run_longExp.py \
#       --random_seed 2021 \
#       --is_training 1 \
#       --root_path ./dataset/electricity/ \
#       --data_path electricity.csv \
#       --model_id Electricity_$seq_len'_'$pred_len \
#       --model PatchTST \
#       --data custom \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 321 \
#       --e_layers 3 \
#       --n_heads 16 \
#       --d_model 128 \
#       --d_ff 256 \
#       --dropout 0.2\
#       --fc_dropout 0.2\
#       --head_dropout 0\
#       --patch_len 16\
#       --stride 8\
#       --des 'Exp' \
#       --train_epochs 100\
#       --patience 10\
#       --lradj 'TST'\
#       --pct_start 0.2\
#       --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/PatchTST_electricity_$seq_len'_'$pred_len.log 
# done

# for pred_len in 48 72 96 120 144 192
# do
#     python -u run_longExp.py \
#       --random_seed 2021 \
#       --is_training 1 \
#       --root_path ./dataset/ETT/ \
#       --data_path ETTh1.csv \
#       --model_id ETTh1_$seq_len'_'$pred_len \
#       --model PatchTST \
#       --data ETTh1 \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 7 \
#       --e_layers 3 \
#       --n_heads 4 \
#       --d_model 16 \
#       --d_ff 128 \
#       --dropout 0.3\
#       --fc_dropout 0.3\
#       --head_dropout 0\
#       --patch_len 16\
#       --stride 8\
#       --des 'Exp' \
#       --train_epochs 100\
#       --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/PatchTST_ETTh1_$seq_len'_'$pred_len.log 
# done

# for pred_len in 48 72 96 120 144 192
# do
#     python -u run_longExp.py \
#       --random_seed 2021 \
#       --is_training 1 \
#       --root_path ./dataset/ETT/ \
#       --data_path ETTh2.csv \
#       --model_id ETTh2_$seq_len'_'$pred_len \
#       --model PatchTST \
#       --data ETTh2 \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 7 \
#       --e_layers 3 \
#       --n_heads 4 \
#       --d_model 16 \
#       --d_ff 128 \
#       --dropout 0.3\
#       --fc_dropout 0.3\
#       --head_dropout 0\
#       --patch_len 16\
#       --stride 8\
#       --des 'Exp' \
#       --train_epochs 100\
#       --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/PatchTST_ETTh2_$seq_len'_'$pred_len.log 
# done

# for pred_len in 48 72 96 120 144 192
# do
#     python -u run_longExp.py \
#       --random_seed 2021 \
#       --is_training 1 \
#       --root_path ./dataset/ETT/ \
#       --data_path ETTm1.csv \
#       --model_id ETTm1_$seq_len'_'$pred_len \
#       --model PatchTST \
#       --data ETTm1 \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 7 \
#       --e_layers 3 \
#       --n_heads 4 \
#       --d_model 16 \
#       --d_ff 128 \
#       --dropout 0.3\
#       --fc_dropout 0.3\
#       --head_dropout 0\
#       --patch_len 16\
#       --stride 8\
#       --des 'Exp' \
#       --train_epochs 100\
#       --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/PatchTST_ETTm1_$seq_len'_'$pred_len.log 
# done

# for pred_len in 48 72 96 120 144 192
# do
#     python -u run_longExp.py \
#       --random_seed 2021 \
#       --is_training 1 \
#       --root_path ./dataset/ETT/ \
#       --data_path ETTm2.csv \
#       --model_id ETTm2_$seq_len'_'$pred_len \
#       --model PatchTST \
#       --data ETTm2 \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 7 \
#       --e_layers 3 \
#       --n_heads 4 \
#       --d_model 16 \
#       --d_ff 128 \
#       --dropout 0.3\
#       --fc_dropout 0.3\
#       --head_dropout 0\
#       --patch_len 16\
#       --stride 8\
#       --des 'Exp' \
#       --train_epochs 100\
#       --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/PatchTST_ETTm2_$seq_len'_'$pred_len.log 
# done