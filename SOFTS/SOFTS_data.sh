model_name=SOFTS
seq_len=48

for pred_len in 12 24 48 96
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id 'PEMS03_'$seq_len'_'$pred_len \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_core 256 \
    --d_ff 256 \
    --learning_rate 0.0003 \
    --lradj cosine \
    --train_epochs 300 \
    --patience 3 \
    --des 'Exp' \
    --itr 1
done

for pred_len in 12 24 48 96
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id 'PEMS04_'$seq_len'_'$pred_len \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_core 256 \
    --d_ff 256 \
    --learning_rate 0.0003 \
    --lradj cosine \
    --train_epochs 300 \
    --patience 3 \
    --des 'Exp' \
    --itr 1
done

for pred_len in 12 24 48 96
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id 'PEMS07_'$seq_len'_'$pred_len \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_core 256 \
    --d_ff 256 \
    --learning_rate 0.0003 \
    --lradj cosine \
    --train_epochs 300 \
    --patience 3 \
    --des 'Exp' \
    --itr 1
done

for pred_len in 12 24 48 96
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id 'PEMS08_'$seq_len'_'$pred_len \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_core 256 \
    --d_ff 256 \
    --learning_rate 0.0003 \
    --lradj cosine \
    --train_epochs 300 \
    --patience 3 \
    --des 'Exp' \
    --itr 1
done
