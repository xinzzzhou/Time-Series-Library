export CUDA_VISIBLE_DEVICES=1

model_name=Random
# root_directory=$(dirname "$(dirname "$(pwd)")")
root_directory='/home/xinz/ar57_scratch/xinz/HTSFB_project/'
current_time=$(date +"%Y-%m-%d %H:%M:%S")

nohup python -u $root_directory/baselines/Time-Series-Library/run-forTraditional.py \
  --task_name statistic \
  --is_training 0 \
  --distribution Gaussian \
  --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
  --data_path Crimes-2001_2023-1130+25_random.csv \
  --output_path $root_directory/HTSFB_output/\
  --checkpoints $root_directory/HTSFB_output/checkpoints/\
  --model_id crime_24_6 \
  --model $model_name \
  --data crime \
  --features M \
  --seq_len 24 \
  --label_len 6 \
  --pred_len 6 \
  --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_6_Gaussian_$current_time.out 

nohup python -u $root_directory/baselines/Time-Series-Library/run-forTraditional.py \
  --task_name statistic \
  --is_training 0 \
  --distribution Gaussian \
  --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
  --data_path Crimes-2001_2023-1130+25_random.csv \
  --output_path $root_directory/HTSFB_output/\
  --checkpoints $root_directory/HTSFB_output/checkpoints/\
  --model_id crime_24_12 \
  --model $model_name \
  --data crime \
  --features M \
  --seq_len 24 \
  --label_len 6 \
  --pred_len 12 \
  --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_12_Gaussian_$current_time.out 

nohup python -u $root_directory/baselines/Time-Series-Library/run-forTraditional.py \
  --task_name statistic \
  --is_training 0 \
  --distribution Gaussian \
  --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
  --data_path Crimes-2001_2023-1130+25_random.csv \
  --output_path $root_directory/HTSFB_output/\
  --checkpoints $root_directory/HTSFB_output/checkpoints/\
  --model_id crime_24_18 \
  --model $model_name \
  --data crime \
  --features M \
  --seq_len 24 \
  --label_len 6 \
  --pred_len 18 \
  --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_18_Gaussian_$current_time.out 

nohup python -u $root_directory/baselines/Time-Series-Library/run-forTraditional.py \
  --task_name statistic \
  --is_training 0 \
  --distribution Gaussian \
  --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
  --data_path Crimes-2001_2023-1130+25_random.csv \
  --output_path $root_directory/HTSFB_output/\
  --checkpoints $root_directory/HTSFB_output/checkpoints/\
  --model_id crime_24_24 \
  --model $model_name \
  --data crime \
  --features M \
  --seq_len 24 \
  --label_len 6 \
  --pred_len 24 \
  --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_24_Gaussian_$current_time.out 
