# conda activate tsl
export CUDA_VISIBLE_DEVICES=0


model_name=Relateformer
root_directory='/home/xinz/ar57_scratch/xinz/HTSFB_project'
data_path_name=electricity.txt
model_id_name=Electricity
data_name=electricity-re-re
current_time=$(date +"%Y-%m-%d %H:%M:%S")

seq_len=336
for pred_len in 96 192 336 720
  do
  for corr in 0 2 4 6 8 10 12 14 16 18 20
    do
    nohup python -u $root_directory/baselines/Time-Series-Library/run_self_elec.py \
      --is_training 1 \
      --root_path  $root_directory/HTSFB_datasets/self/electricity/ \
      --data_path $data_path_name \
      --data_topk_path $root_directory/HTSFB_datasets/self/electricity/matrix_rank_electricity1.npy \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --patch_len 16\
      --patch_stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 3\
      --k $corr\
      --itr 1 --batch_size 128 --learning_rate 0.0001 > $root_directory/HTSFB_output/logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$corr'_'$current_time.log 
  done
done