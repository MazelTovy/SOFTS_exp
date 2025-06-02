model_name=iTransformer
# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=48

for pred_len in 48 72 96 120 144 192
do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/Solar/ \
        --data_path solar_AL.txt \
        --model_id solar_$seq_len'_'$pred_len \
        --model $model_name \
        --data Solar \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 137 \
        --dec_in 137 \
        --c_out 137 \
        --des 'Exp' \  
        --d_model 512 \
        --d_ff 512 \    
        --learning_rate 0.0005 \
        --itr 1 \
        --train_epochs 100 \
        --patience 100 \
        >logs/LongForecasting/iTransformer_solar_$seq_len'_'$pred_len.log
done
