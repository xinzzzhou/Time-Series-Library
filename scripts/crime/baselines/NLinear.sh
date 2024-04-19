export CUDA_VISIBLE_DEVICES=0

model_name=NLinear
# root_directory=$(dirname "$(dirname "$(pwd)")")
root_directory='/home/xinz/ar57_scratch/xinz/HTSFB_project/'
current_time=$(date +"%Y-%m-%d %H:%M:%S")

batch_sizes=(16 32 64 128)
for batch_size in "${batch_sizes[@]}"; do
  current_time=$(date +"%Y-%m-%d %H:%M:%S")
  nohup python -u $root_directory/baselines/Time-Series-Library/run_self.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
    --data_path Crimes-2001_2023.csv \
    --output_path $root_directory/HTSFB_output/\
    --checkpoints $root_directory/HTSFB_output/checkpoints/\
    --model_id crime_24_6 \
    --model $model_name \
    --data crime \
    --features M \
    --seq_len 24 \
    --label_len 6 \
    --pred_len 6 \
    --top_k 5 \
    --des 'Exp' \
    --dropout 0.6 \
    --batch_size $batch_size\
    --itr 1\
    --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_6_$current_time_"($batch_size)".out 

  nohup python -u $root_directory/baselines/Time-Series-Library/run_self.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
    --data_path Crimes-2001_2023.csv \
    --output_path $root_directory/HTSFB_output/\
    --checkpoints $root_directory/HTSFB_output/checkpoints/\
    --model_id crime_24_12 \
    --model $model_name \
    --data crime \
    --features M \
    --seq_len 24 \
    --label_len 6 \
    --pred_len 12 \
    --top_k 5 \
    --des 'Exp' \
    --dropout 0.6 \
    --batch_size $batch_size\
    --itr 1\
    --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_12_$current_time_"($batch_size)".out 

  nohup python -u $root_directory/baselines/Time-Series-Library/run_self.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
    --data_path Crimes-2001_2023.csv \
    --output_path $root_directory/HTSFB_output/\
    --checkpoints $root_directory/HTSFB_output/checkpoints/\
    --model_id crime_24_18 \
    --model $model_name \
    --data crime \
    --features M \
    --seq_len 24 \
    --label_len 6 \
    --pred_len 18 \
    --top_k 5 \
    --des 'Exp' \
    --dropout 0.6 \
    --batch_size $batch_size\
    --itr 1\
    --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_18_$current_time_"($batch_size)".out 

  nohup python -u $root_directory/baselines/Time-Series-Library/run_self.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
    --data_path Crimes-2001_2023.csv \
    --output_path $root_directory/HTSFB_output/\
    --checkpoints $root_directory/HTSFB_output/checkpoints/\
    --model_id crime_24_24 \
    --model $model_name \
    --data crime \
    --features M \
    --seq_len 24 \
    --label_len 6 \
    --pred_len 24 \
    --top_k 5 \
    --des 'Exp' \
    --dropout 0.6 \
    --batch_size $batch_size\
    --itr 1\
    --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_24_$current_time_"($batch_size)".out 
done