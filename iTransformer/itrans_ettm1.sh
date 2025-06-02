model_name=iTransformer
# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=48

for pred_len in 48 72 96 120 144 168 192
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
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --train_epochs 100 \
        --patience 100 \
        --itr 1 >logs/LongForecasting/iTransformer_ETTm1_$seq_len'_'$pred_len.log
done
