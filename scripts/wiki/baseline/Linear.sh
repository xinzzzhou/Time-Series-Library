export CUDA_VISIBLE_DEVICES=0

model_name=Linear
# root_directory=$(dirname "$(dirname "$(pwd)")")
root_directory='/home/xinz/ar57_scratch/xinz/HTSFB_project/'
current_time=$(date +"%Y-%m-%d %H:%M:%S")

nohup python -u $root_directory/baselines/Time-Series-Library/run_self_wiki.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_directory/HTSFB_datasets/self/Wiki/ \
  --data_path train_1_people.csv \
  --output_path $root_directory/HTSFB_output/\
  --checkpoints $root_directory/HTSFB_output/checkpoints/\
  --model_id wiki_people_28_7 \
  --model $model_name \
  --data wiki \
  --features M \
  --seq_len 28 \
  --label_len 7 \
  --pred_len 7 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6107 \
  --dec_in 6107 \
  --c_out 6107 \
  --d_model 256 \
  --d_ff 128 \
  --batch_size 32\
  --top_k 5 \
  --des 'Exp' \
  --dropout 0.6 \
  --itr 1 \
  --freq 'd' > $root_directory/HTSFB_output/"($model_name)"_wiki_people_28_7_$current_time.out 

nohup python -u $root_directory/baselines/Time-Series-Library/run_self_wiki.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_directory/HTSFB_datasets/self/Wiki/ \
  --data_path train_1_people.csv \
  --output_path $root_directory/HTSFB_output/\
  --checkpoints $root_directory/HTSFB_output/checkpoints/\
  --model_id wiki_people_28_14 \
  --model $model_name \
  --data wiki \
  --features M \
  --seq_len 28 \
  --label_len 7 \
  --pred_len 14 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6107 \
  --dec_in 6107 \
  --c_out 6107 \
  --d_model 256 \
  --d_ff 128 \
  --batch_size 32\
  --top_k 5 \
  --des 'Exp' \
  --dropout 0.6 \
  --itr 1 \
  --freq 'd' > $root_directory/HTSFB_output/"($model_name)"_wiki_people_28_14_$current_time.out 

nohup python -u $root_directory/baselines/Time-Series-Library/run_self_wiki.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_directory/HTSFB_datasets/self/Wiki/ \
  --data_path train_1_people.csv \
  --output_path $root_directory/HTSFB_output/\
  --checkpoints $root_directory/HTSFB_output/checkpoints/\
  --model_id wiki_people_28_21 \
  --model $model_name \
  --data wiki \
  --features M \
  --seq_len 28 \
  --label_len 7 \
  --pred_len 21 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6107 \
  --dec_in 6107 \
  --c_out 6107 \
  --d_model 256 \
  --d_ff 128 \
  --batch_size 32\
  --top_k 5 \
  --des 'Exp' \
  --dropout 0.6 \
  --itr 1 \
  --freq 'd' > $root_directory/HTSFB_output/"($model_name)"_wiki_people_28_21_$current_time.out 

nohup python -u $root_directory/baselines/Time-Series-Library/run_self_wiki.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_directory/HTSFB_datasets/self/Wiki/ \
  --data_path train_1_people.csv \
  --output_path $root_directory/HTSFB_output/\
  --checkpoints $root_directory/HTSFB_output/checkpoints/\
  --model_id wiki_people_28_28 \
  --model $model_name \
  --data wiki \
  --features M \
  --seq_len 28 \
  --label_len 7 \
  --pred_len 28 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6107 \
  --dec_in 6107 \
  --c_out 6107 \
  --d_model 256 \
  --d_ff 128 \
  --batch_size 32\
  --top_k 5 \
  --des 'Exp' \
  --dropout 0.6 \
  --itr 1 \
  --freq 'd' > $root_directory/HTSFB_output/"($model_name)"_wiki_people_28_28_$current_time.out 
