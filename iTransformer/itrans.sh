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
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id ECL_$seq_len'_'$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --batch_size 16 \
        --learning_rate 0.0005 \
        --train_epochs 100 \
        --patience 100 \
        --itr 1 >logs/LongForecasting/iTransformer_electricity_$seq_len'_'$pred_len.log
done

for pred_len in 48 72 96 120 144 168 192
do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_$seq_len'_'$pred_len \
        --model $model_name \
        --data ETTh1 \
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
        --itr 1 >logs/LongForecasting/iTransformer_ETTh1_$seq_len'_'$pred_len.log
done

for pred_len in 48 72 96 120 144 168 192
do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_$seq_len'_'$pred_len \
        --model $model_name \
        --data ETTh2 \
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
        --itr 1 >logs/LongForecasting/iTransformer_ETTh2_$seq_len'_'$pred_len.log
done

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

for pred_len in 48 72 96 120 144 168 192
do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm2_$seq_len'_'$pred_len \
        --model $model_name \
        --data ETTm2 \
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
        --itr 1 >logs/LongForecasting/iTransformer_ETTm2_$seq_len'_'$pred_len.log
done

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
