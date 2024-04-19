export CUDA_VISIBLE_DEVICES=0

model_name=Relateformer
root_directory=$(dirname "$(dirname "$(pwd)")")
# root_directory='/home/xinz/ar57_scratch/xinz/HTSFB_project/'
current_time=$(date +"%Y-%m-%d %H:%M:%S")
corrs=(13) 

d_ff=384 #d_ffs=(64 128 256 384)
d_model=256 # d_models=(64 128 256 384)
layer=1 # layers=(1 2 3 4 5 6)
learning_rate=0.001 # learning_rates=(0.1 0.01 0.001 0.0001)
batch_size=128 # batch_sizes=(16 32 64 128)
n_heads=4 # n_headss=(1 2 3 4 5 6 8 16)
horizons=(6)
for corr in "${corrs[@]}"; do
  for horizon in "${horizons[@]}"; do
    # for d_ff in "${d_ffs[@]}"; do
      # for d_model in "${d_models[@]}"; do
      #   for layer in "${layers[@]}"; do
      #     for learning_rate in "${learning_rates[@]}"; do
      #       for batch_size in "${batch_sizes[@]}"; do
      #         for n_heads in "${n_headss[@]}"; do
              current_time=$(date +"%Y-%m-%d %H:%M:%S")
              nohup python -u $root_directory/baselines/Time-Series-Library/run.py \
                --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
                --task_name long_term_forecast \
                --is_training 1 \
                --data_path  Crimes-2001_2023.csv\
                --data_topk_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/top-k-train/matrix_rank_train_crime1.npy\
                --output_path $root_directory/HTSFB_output/\
                --checkpoints $root_directory/HTSFB_output/checkpoints/\
                --model_id crime_24_"($horizon)" \
                --model $model_name \
                --data crime-reindex-unrelated \
                --features M \
                --seq_len 24 \
                --label_len 6 \
                --pred_len $horizon \
                --patch_len 12 \
                --patch_stride 6\
                --e_layers $layer \
                --d_layers 1 \
                --factor 3 \
                --enc_in 1155 \
                --des 'Exp' \
                --n_heads $n_heads \
                --d_model $d_model\
                --d_ff $d_ff\
                --top_k 5 \
                --dropout 0.6 \
                --k $corr\
                --seasonality 6\
                --itr 1 \
                --batch_size $batch_size\
                --learning_rate $learning_rate\
                --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_"($horizon)"_"($corr)"_"($current_time)"_"dff($d_ff)"_"dm($d_model)"_"l($layer)"_"lr($learning_rate)"_"bz($batch_size)"_"head($n_heads)".out
    #           done
    #         done
    #       done
    #     done
    #   done
    # done
  done
done