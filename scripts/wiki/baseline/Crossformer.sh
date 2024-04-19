export CUDA_VISIBLE_DEVICES=0

model_name=Crossformer
# root_directory=$(dirname "$(dirname "$(pwd)")")
root_directory='/home/xinz/ar57_scratch/xinz/HTSFB_project/'
current_time=$(date +"%Y-%m-%d %H:%M:%S")

#bz128 dff128 dm128 lr0.001 head4 layer2
d_ff=128 #s=(64 128 256 384)
d_model=128 # d_models=(64 256 384)
layers=(2)
learning_rate=0.001 # learning_rates=(0.1 0.01 0.0001)
batch_size=32 # batch_sizes=(16 32 64 128)
n_heads=4 # n_headss=(1 2 3 5 6 8 16)
horizons=(28 35 42 49)

for horizon in "${horizons[@]}"; do
  # for d_ff in "${d_ffs[@]}"; do
    # for d_model in "${d_models[@]}"; do
      for layer in "${layers[@]}"; do
    #     for learning_rate in "${learning_rates[@]}"; do
    #       for batch_size in "${batch_sizes[@]}"; do
    #         for n_heads in "${n_headss[@]}"; do
              current_time=$(date +"%Y-%m-%d %H:%M:%S")
              nohup python -u $root_directory/baselines/Time-Series-Library/run_self_wiki.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path $root_directory/HTSFB_datasets/self/Wiki/ \
                --data_path train_1_people.csv \
                --output_path $root_directory/HTSFB_output/\
                --checkpoints $root_directory/HTSFB_output/checkpoints/\
                --model_id wiki_people_28_$horizon \
                --model $model_name \
                --data wiki \
                --features M \
                --seq_len 42 \
                --label_len 7 \
                --pred_len $horizon \
                --patch_len 14 \
                --patch_stride 7\
                --e_layers $layer \
                --d_layers 1 \
                --factor 3 \
                --enc_in 6107 \
                --d_model $d_model \
                --d_ff $d_ff \
                --batch_size $batch_size\
                --top_k 5 \
                --des 'Exp' \
                --dropout 0.6 \
                --seasonality 7\
                --itr 1 \
                --learning_rate $learning_rate\
                --n_heads $n_heads \
                --freq 'd' > $root_directory/HTSFB_output/"($model_name)"_wiki_people_28_"($horizon)"_"($current_time)"_"($d_ff)"_"($d_model)"_"($layer)"_"($learning_rate)"_"($batch_size)"_"($n_heads)".out
    #           done
    #         done
    #       done
    #     done
    #   done
    # done
  done
done
# nohup python -u $root_directory/baselines/Time-Series-Library/run_self_wiki.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_directory/HTSFB_datasets/self/Wiki/ \
#   --data_path train_1_people.csv \
#   --output_path $root_directory/HTSFB_output/\
#   --checkpoints $root_directory/HTSFB_output/checkpoints/\
#   --model_id wiki_people_28_7 \
#   --model $model_name \
#   --data wiki \
#   --features M \
#   --seq_len 28 \
#   --label_len 7 \
#   --pred_len 7 \
#   --patch_len 14 \
#   --patch_stride 7\
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 6107 \
#   --d_model 256 \
#   --d_ff 128 \
#   --batch_size 32\
#   --top_k 5 \
#   --des 'Exp' \
#   --dropout 0.6 \
#   --seasonality 7\
#   --itr 1 \
#   --freq 'd' > $root_directory/HTSFB_output/"($model_name)"_wiki_people_28_7_$current_time.out 

# nohup python -u $root_directory/baselines/Time-Series-Library/run_self_wiki.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_directory/HTSFB_datasets/self/Wiki/ \
#   --data_path train_1_people.csv \
#   --output_path $root_directory/HTSFB_output/\
#   --checkpoints $root_directory/HTSFB_output/checkpoints/\
#   --model_id wiki_people_28_14 \
#   --model $model_name \
#   --data wiki \
#   --features M \
#   --seq_len 28 \
#   --label_len 7 \
#   --pred_len 14 \
#   --patch_len 14 \
#   --patch_stride 7\
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 6107 \
#   --d_model 256 \
#   --d_ff 128 \
#   --batch_size 32\
#   --top_k 5 \
#   --des 'Exp' \
#   --dropout 0.6 \
#   --seasonality 7\
#   --itr 1 \
#   --freq 'd' > $root_directory/HTSFB_output/"($model_name)"_wiki_people_28_14_$current_time.out 

# nohup python -u $root_directory/baselines/Time-Series-Library/run_self_wiki.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_directory/HTSFB_datasets/self/Wiki/ \
#   --data_path train_1_people.csv \
#   --output_path $root_directory/HTSFB_output/\
#   --checkpoints $root_directory/HTSFB_output/checkpoints/\
#   --model_id wiki_people_28_21 \
#   --model $model_name \
#   --data wiki \
#   --features M \
#   --seq_len 28 \
#   --label_len 7 \
#   --pred_len 21 \
#   --patch_len 14 \
#   --patch_stride 7\
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --d_model 6107 \
#   --d_model 256 \
#   --d_ff 128 \
#   --batch_size 32\
#   --top_k 5 \
#   --des 'Exp' \
#   --dropout 0.6 \
#   --seasonality 7\
#   --itr 1 \
#   --freq 'd' > $root_directory/HTSFB_output/"($model_name)"_wiki_people_28_21_$current_time.out 

#   nohup python -u $root_directory/baselines/Time-Series-Library/run_self_wiki.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path $root_directory/HTSFB_datasets/self/Wiki/ \
#   --data_path train_1_people.csv \
#   --output_path $root_directory/HTSFB_output/\
#   --checkpoints $root_directory/HTSFB_output/checkpoints/\
#   --model_id wiki_people_28_28 \
#   --model $model_name \
#   --data wiki \
#   --features M \
#   --seq_len 28 \
#   --label_len 7 \
#   --pred_len 28 \
#   --patch_len 14 \
#   --patch_stride 7 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --d_model 6107 \
#   --d_model 256 \
#   --d_ff 128 \
#   --batch_size 32\
#   --top_k 5 \
#   --des 'Exp' \
#   --dropout 0.6 \
#   --seasonality 7\
#   --itr 1 \
#   --freq 'd' > $root_directory/HTSFB_output/"($model_name)"_wiki_people_28_28_$current_time.out 
  