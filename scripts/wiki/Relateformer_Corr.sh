export CUDA_VISIBLE_DEVICES=0

model_name=Relateformer
root_directory=$(dirname "$(dirname "$(pwd)")")
# root_directory='/home/xinz/ar57_scratch/xinz/HTSFB_project/'
current_time=$(date +"%Y-%m-%d %H:%M:%S")

corrs=(0) 
d_ff=128 #d_ffs=(64 128 256 384)
d_model=128 # d_models=(64 128 256 384)
layer=2 # layers=(1 2 3 4 5 6)
learning_rate=0.001 # learning_rates=(0.1 0.01 0.001 0.0001)
batch_size=128 # batch_sizes=(16 32 64 128)
n_heads=4 # n_headss=(1 2 3 4 5 6 8 16)
horizons=(28 35 42 49)

for corr in "${corrs[@]}"; do
  for horizon in "${horizons[@]}"; do
    # for d_ff in "${d_ffs[@]}"; do
      # for d_model in "${d_models[@]}"; do
      #   for layer in "${layers[@]}"; do
      #     for learning_rate in "${learning_rates[@]}"; do
      #       for batch_size in "${batch_sizes[@]}"; do
      #         for n_heads in "${n_headss[@]}"; do
                current_time=$(date +"%Y-%m-%d %H:%M:%S")
                nohup python -u $root_directory/baselines/Time-Series-Library/run_self_wiki.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path $root_directory/HTSFB_datasets/self/Wiki/ \
                --data_path train_1_people.csv \
                --data_topk_path $root_directory/HTSFB_datasets/self/Wiki/top-k/matrix_rank_wiki1.npy\
                --output_path $root_directory/HTSFB_output/\
                --checkpoints $root_directory/HTSFB_output/checkpoints/\
                --model_id wiki_people_28_"($horizon)"_Corr$corr\
                --model $model_name \
                --data wiki-reindex\
                --features M \
                --seq_len 28 \
                --label_len 7 \
                --pred_len $horizon \
                --patch_len 14\
                --patch_stride 7\
                --e_layers $layer \
                --d_layers 1 \
                --factor 3 \
                --enc_in 6107 \
                --n_heads $n_heads \
                --d_model $d_model \
                --d_ff $d_ff \
                --top_k 5 \
                --des 'Exp' \
                --dropout 0.6 \
                --k $corr\
                --seasonality 7\
                --itr 1 \
                --learning_rate $learning_rate\
                --batch_size $batch_size\
                --freq 'd' > $root_directory/HTSFB_output/"($model_name)"_wiki_people_28_"($horizon)"_"($corr)"_"($current_time)"_"dff($d_ff)"_"dm($d_model)"_"l($layer)"_"lr($learning_rate)"_"bz($batch_size)"_"head($n_heads)".out
    #           done
    #         done
    #       done
    #     done
    #   done
    # done
  done
done


#   nohup python -u $root_directory/baselines/Time-Series-Library/run_self_wiki.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path $root_directory/HTSFB_datasets/self/Wiki/ \
#     --data_path train_1_people.csv \
#     --data_topk_path /home/xinz/ar57_scratch/xinz/HTSFB_project/HTSFB_datasets/self/Wiki/top-k/matrix_rank_wiki.npy\
#     --output_path $root_directory/HTSFB_output/\
#     --checkpoints $root_directory/HTSFB_output/checkpoints/\
#     --model_id wiki_people_28_14_Corr$corr \
#     --model $model_name \
#     --data wiki-re-re\
#     --features M \
#     --seq_len 28 \
#     --label_len 7 \
#     --pred_len 14 \
#     --patch_len 14\
#     --patch_stride 7\
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 6107 \
#     --d_model 128 \
#     --d_ff 128 \
#     --top_k 5 \
#     --des 'Exp' \
#     --dropout 0.6 \
#     --k $corr\
#     --seasonality 7\
#     --itr 1 \
#     --batch_size 128\
#     --freq 'd' > $root_directory/HTSFB_output/"($model_name)"_wiki_people_28_14_"($corr)"_$current_time.out 

#   nohup python -u $root_directory/baselines/Time-Series-Library/run_self_wiki.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path $root_directory/HTSFB_datasets/self/Wiki/ \
#     --data_path train_1_people.csv \
#     --data_topk_path /home/xinz/ar57_scratch/xinz/HTSFB_project/HTSFB_datasets/self/Wiki/top-k/matrix_rank_wiki.npy\
#     --output_path $root_directory/HTSFB_output/\
#     --checkpoints $root_directory/HTSFB_output/checkpoints/\
#     --model_id wiki_people_28_21_Corr$corr \
#     --model $model_name \
#     --data wiki-re-re\
#     --features M \
#     --seq_len 28 \
#     --label_len 7 \
#     --pred_len 21 \
#     --patch_len 14\
#     --patch_stride 7\
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 6107 \
#     --d_model 128 \
#     --d_ff 128 \
#     --top_k 5 \
#     --des 'Exp' \
#     --dropout 0.6 \
#     --k $corr\
#     --seasonality 7\
#     --itr 1 \
#     --batch_size 128\
#     --freq 'd' > $root_directory/HTSFB_output/"($model_name)"_wiki_people_28_21_"($corr)"_$current_time.out 

#   nohup python -u $root_directory/baselines/Time-Series-Library/run_self_wiki.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path $root_directory/HTSFB_datasets/self/Wiki/ \
#     --data_path train_1_people.csv \
#     --data_topk_path /home/xinz/ar57_scratch/xinz/HTSFB_project/HTSFB_datasets/self/Wiki/top-k/matrix_rank_wiki.npy\
#     --output_path $root_directory/HTSFB_output/\
#     --checkpoints $root_directory/HTSFB_output/checkpoints/\
#     --model_id wiki_people_28_28_Corr$corr \
#     --model $model_name \
#     --data wiki-re-re\
#     --features M \
#     --seq_len 28 \
#     --label_len 7 \
#     --pred_len 28 \
#     --patch_len 14\
#     --patch_stride 7\
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 6107 \
#     --d_model 128 \
#     --d_ff 128 \
#     --top_k 5 \
#     --des 'Exp' \
#     --dropout 0.6 \
#     --k $corr\
#     --seasonality 7\
#     --itr 1 \
#     --batch_size 128\
#     --freq 'd' > $root_directory/HTSFB_output/"($model_name)"_wiki_people_28_28_"($corr)"_$current_time.out 
# done