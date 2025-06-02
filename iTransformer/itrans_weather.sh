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
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_$seq_len'_'$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --train_epochs 100 \
        --patience 100 \
        --itr 1 \
        >logs/LongForecasting/iTransformer_weather_$seq_len'_'$pred_len.log
done
