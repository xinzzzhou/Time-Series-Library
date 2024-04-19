export CUDA_VISIBLE_DEVICES=0

model_name=DLinear
# root_directory=$(dirname "$(dirname "$(pwd)")")
root_directory='/home/xinz/ar57_scratch/xinz/HTSFB_project/'
current_time=$(date +"%Y-%m-%d %H:%M:%S")

d_model=384 #d_models=(64 128 256 384)
learning_rate=0.0001 # learning_rates=(0.1 0.01 0.001 0.0001)
batch_size=128 # batch_sizes=(16 32 64 128)
horizons=(6 12 18 24)

for horizon in "${horizons[@]}"; do
  for d_model in "${d_models[@]}"; do
    # for learning_rate in "${learning_rates[@]}"; do
    #   for batch_size in "${batch_sizes[@]}"; do
        current_time=$(date +"%Y-%m-%d %H:%M:%S")
        nohup python -u $root_directory/baselines/Time-Series-Library/run_self.py \
          --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
          --task_name long_term_forecast \
          --is_training 1 \
          --data_path  Crimes-2001_2023.csv\
          --output_path $root_directory/HTSFB_output/\
          --checkpoints $root_directory/HTSFB_output/checkpoints/\
          --seasonal_patterns 'Monthly' \
          --model_id crime_24_$horizon \
          --model $model_name \
          --data crime-re-re \
          --features M \
          --seq_len 24 \
          --label_len 6 \
          --pred_len $horizon \
          --des 'Exp' \
          --dropout 0.6 \
          --k 0\
          --d_model $d_model\
          --top_k 5 \
          --itr 1 \
          --batch_size $batch_size\
          --learning_rate $learning_rate\
          --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_reindex_crime_24_"($horizon)"_"($current_time)"_"($d_model)"_"($learning_rate)"_"($batch_size)".out
    #   done
    # done
  done
done

# nohup python -u $root_directory/baselines/Time-Series-Library/run_self.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
#   --data_path Crimes-2001_2023.csv \
#   --output_path $root_directory/HTSFB_output/\
#   --checkpoints $root_directory/HTSFB_output/checkpoints/\
#   --seasonal_patterns 'Monthly' \
#   --model_id crime_24_6 \
#   --model $model_name \
#   --data crime \
#   --features M \
#   --seq_len 24 \
#   --label_len 6 \
#   --pred_len 6 \
#   --d_model 256 \
#   --top_k 5 \
#   --des 'Exp' \
#   --dropout 0.6 \
#   --batch_size 16\
#   --itr 1\
#   --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_6_$current_time.out 

# nohup python -u $root_directory/baselines/Time-Series-Library/run_self.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
#   --data_path Crimes-2001_2023.csv \
#   --output_path $root_directory/HTSFB_output/\
#   --checkpoints $root_directory/HTSFB_output/checkpoints/\
#   --seasonal_patterns 'Monthly' \
#   --model_id crime_24_12 \
#   --model $model_name \
#   --data crime \
#   --features M \
#   --seq_len 24 \
#   --label_len 6 \
#   --pred_len 12 \
#   --d_model 256 \
#   --top_k 5 \
#   --des 'Exp' \
#   --dropout 0.6 \
#   --batch_size 16\
#   --itr 1\
#   --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_12_$current_time.out 

# nohup python -u $root_directory/baselines/Time-Series-Library/run_self.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
#   --data_path Crimes-2001_2023.csv \
#   --output_path $root_directory/HTSFB_output/\
#   --checkpoints $root_directory/HTSFB_output/checkpoints/\
#   --seasonal_patterns 'Monthly' \
#   --model_id crime_24_18 \
#   --model $model_name \
#   --data crime \
#   --features M \
#   --seq_len 24 \
#   --label_len 6 \
#   --pred_len 18 \
#   --d_model 256 \
#   --top_k 5 \
#   --des 'Exp' \
#   --dropout 0.6 \
#   --batch_size 16\
#   --itr 1\
#   --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_18_$current_time.out 

# nohup python -u $root_directory/baselines/Time-Series-Library/run_self.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
#   --data_path Crimes-2001_2023.csv \
#   --output_path $root_directory/HTSFB_output/\
#   --checkpoints $root_directory/HTSFB_output/checkpoints/\
#   --seasonal_patterns 'Monthly' \
#   --model_id crime_24_24 \
#   --model $model_name \
#   --data crime \
#   --features M \
#   --seq_len 24 \
#   --label_len 6 \
#   --pred_len 24 \
#   --d_model 256 \
#   --top_k 5 \
#   --des 'Exp' \
#   --dropout 0.6 \
#   --batch_size 16\
#   --itr 1\
#   --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_24_$current_time.out 
 


# nohup python -u $root_directory/baselines/Time-Series-Library/run_self.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
#   --data_path Crimes-2001_2023.csv \
#   --output_path $root_directory/HTSFB_output/\
#   --checkpoints $root_directory/HTSFB_output/checkpoints/\
#   --seasonal_patterns 'Monthly' \
#   --model_id crime_24_6 \
#   --model $model_name \
#   --data crime \
#   --features M \
#   --seq_len 24 \
#   --label_len 6 \
#   --pred_len 6 \
#   --d_model 256 \
#   --top_k 5 \
#   --des 'Exp' \
#   --dropout 0.6 \
#   --batch_size 32\
#   --itr 1\
#   --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_6_$current_time.out 

# nohup python -u $root_directory/baselines/Time-Series-Library/run_self.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
#   --data_path Crimes-2001_2023.csv \
#   --output_path $root_directory/HTSFB_output/\
#   --checkpoints $root_directory/HTSFB_output/checkpoints/\
#   --seasonal_patterns 'Monthly' \
#   --model_id crime_24_12 \
#   --model $model_name \
#   --data crime \
#   --features M \
#   --seq_len 24 \
#   --label_len 6 \
#   --pred_len 12 \
#   --d_model 256 \
#   --top_k 5 \
#   --des 'Exp' \
#   --dropout 0.6 \
#   --batch_size 32\
#   --itr 1\
#   --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_12_$current_time.out 

# nohup python -u $root_directory/baselines/Time-Series-Library/run_self.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
#   --data_path Crimes-2001_2023.csv \
#   --output_path $root_directory/HTSFB_output/\
#   --checkpoints $root_directory/HTSFB_output/checkpoints/\
#   --seasonal_patterns 'Monthly' \
#   --model_id crime_24_18 \
#   --model $model_name \
#   --data crime \
#   --features M \
#   --seq_len 24 \
#   --label_len 6 \
#   --pred_len 18 \
#   --d_model 256 \
#   --top_k 5 \
#   --des 'Exp' \
#   --dropout 0.6 \
#   --batch_size 32\
#   --itr 1\
#   --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_18_$current_time.out 

# nohup python -u $root_directory/baselines/Time-Series-Library/run_self.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
#   --data_path Crimes-2001_2023.csv \
#   --output_path $root_directory/HTSFB_output/\
#   --checkpoints $root_directory/HTSFB_output/checkpoints/\
#   --seasonal_patterns 'Monthly' \
#   --model_id crime_24_24 \
#   --model $model_name \
#   --data crime \
#   --features M \
#   --seq_len 24 \
#   --label_len 6 \
#   --pred_len 24 \
#   --d_model 256 \
#   --top_k 5 \
#   --des 'Exp' \
#   --dropout 0.6 \
#   --batch_size 32\
#   --itr 1\
#   --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_24_$current_time.out 
 
