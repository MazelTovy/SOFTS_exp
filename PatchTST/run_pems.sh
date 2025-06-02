if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=48

for pred_len in 12 24 36 48 72 96
do
    python -u run_longExp.py \
      --random_seed 2021 \
      --is_training 1 \
      --root_path ./dataset/PEMS/ \
      --data_path PEMS03.npz \
      --model_id PEMS03_$seq_len'_'$pred_len \
      --model PatchTST \
      --data PEMS \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 358 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/PatchTST_PEMS03_$seq_len'_'$pred_len.log 
done

for pred_len in 12 24 36 48 72 96
do
    python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_$seq_len'_'$pred_len \
    --model PatchTST \
    --data PEMS \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 307 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --pct_start 0.2\
    --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/PatchTST_PEMS04_$seq_len'_'$pred_len.log 
done

for pred_len in 12 24 36 48 72 96
do
    python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_$seq_len'_'$pred_len \
    --model PatchTST \
    --data PEMS \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 883 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --pct_start 0.2\
    --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/PatchTST_PEMS07_$seq_len'_'$pred_len.log 
done

for pred_len in 12 24 36 48 72 96
do
    python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id PEMS08_$seq_len'_'$pred_len \
    --model PatchTST \
    --data PEMS \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 170 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --pct_start 0.2\
    --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/PatchTST_PEMS08_$seq_len'_'$pred_len.log 
done

for pred_len in 48 72 96 120 144 192
do
    python -u run_longExp.py \
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
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --pct_start 0.2\
    --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/PatchTST_solar_$seq_len'_'$pred_len.log 
done
