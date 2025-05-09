model_name=SOFTS
seq_len=96

# Beta and gamma values to search
beta_values=(5 10 15 20)
gamma_values=(5 10 15 20)

for beta in "${beta_values[@]}"
do
  for gamma in "${gamma_values[@]}"
  do
    echo "Running with beta=$beta, gamma=$gamma"
    
    for pred_len in 96 192 336 720
    do
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT/ \
        --data_path ETTh1.csv \
        --model_id 'ETTh1_'$seq_len'_'$pred_len'_beta'$beta'_gamma'$gamma \
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
        --beta $beta \
        --gamma $gamma \
        --des 'Exp' \
        --itr 1
    done

    for pred_len in 96 192 336 720
    do
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT/ \
        --data_path ETTh2.csv \
        --model_id 'ETTh2_'$seq_len'_'$pred_len'_beta'$beta'_gamma'$gamma \
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
        --beta $beta \
        --gamma $gamma \
        --des 'Exp' \
        --itr 1
    done

    for pred_len in 96 192 336 720
    do
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT/ \
        --data_path ETTm1.csv \
        --model_id 'ETTm1_'$seq_len'_'$pred_len'_beta'$beta'_gamma'$gamma \
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
        --beta $beta \
        --gamma $gamma \
        --des 'Exp' \
        --itr 1
    done

    for pred_len in 96 192 336 720
    do
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT/ \
        --data_path ETTm2.csv \
        --model_id 'ETTm2_'$seq_len'_'$pred_len'_beta'$beta'_gamma'$gamma \
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
        --beta $beta \
        --gamma $gamma \
        --des 'Exp' \
        --itr 1
    done
  done
done

echo "All rounds completed."
