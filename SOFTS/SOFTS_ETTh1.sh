model_name=SOFTS
seq_len=96

for repeat in {1..5}; do
    echo "Starting round $repeat..."
    for pred_len in 96 192 336 720
    do
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT/ \
        --data_path ETTh1.csv \
        --model_id 'ETTh1_'$seq_len'_'$pred_len \
        --model $model_name \
        --data ETTh1 \
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
        --itr 1 \
        --lambda_penalty 10 \
        --sim_threshold 0.4 \
        --dist_threshold 0.3
    done

    for pred_len in 96 192 336 720
    do
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT/ \
        --data_path ETTh2.csv \
        --model_id 'ETTh2_'$seq_len'_'$pred_len \
        --model $model_name \
        --data ETTh2 \
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
        --itr 1 \
        --lambda_penalty 10 \
        --sim_threshold 0.4 \
        --dist_threshold 0.3
    done

    for pred_len in 96 192 336 720
    do
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT/ \
        --data_path ETTm1.csv \
        --model_id 'ETTm1_'$seq_len'_'$pred_len \
        --model $model_name \
        --data ETTm1 \
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
        --itr 1 \
        --lambda_penalty 10 \
        --sim_threshold 0.4 \
        --dist_threshold 0.3
    done

    for pred_len in 96 192 336 720
    do
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT/ \
        --data_path ETTm2.csv \
        --model_id 'ETTm2_'$seq_len'_'$pred_len \
        --model $model_name \
        --data ETTm2 \
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
        --itr 1 \
        --lambda_penalty 10 \
        --sim_threshold 0.4 \
        --dist_threshold 0.3
    done
done

echo "All rounds completed."