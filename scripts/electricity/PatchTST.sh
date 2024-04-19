# conda activate tsl
export CUDA_VISIBLE_DEVICES=0


model_name=PatchTST
root_directory='/home/xinz/ar57_scratch/xinz/HTSFB_project'
data_path_name=electricity.txt
model_id_name=Electricity
data_name=electricity
current_time=$(date +"%Y-%m-%d %H:%M:%S")

seq_len=336
for pred_len in 96 192 336 720
do
    nohup python -u $root_directory/baselines/Time-Series-Library/run_self_elec.py \
      --is_training 1 \
      --root_path  $root_directory/HTSFB_datasets/self/electricity/ \
      --data_path $data_path_name \
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
      --itr 1 --batch_size 32 --learning_rate 0.0001 > $root_directory/HTSFB_output/logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$current_time.log 
done