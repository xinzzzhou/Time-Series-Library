export CUDA_VISIBLE_DEVICES=0

model_name=Relateformer
root_directory=$(dirname "$(dirname "$(pwd)")")
# root_directory='/home/xinz/ar57_scratch/xinz/HTSFB_project/'
current_time=$(date +"%Y-%m-%d %H:%M:%S")
corrs=(4) 
#dm_256, el_2, nh_4, bz_32, df_384, lr0.001
d_ff=256 #d_ffs=(64 128 256 384)
d_model=128 # d_models=(64 256 384)
layer=3
learning_rate=0.0001 # learning_rates=(0.1 0.01 0.0001)
batch_size=32 # batch_sizes=(16 32 64 128)
n_heads=16 # n_headss=(1 2 3 5 6 8 16)
horizons=(720)
for corr in "${corrs[@]}"; do
  for horizon in "${horizons[@]}"; do
    # for d_ff in "${d_ffs[@]}"; do
      # for d_model in "${d_models[@]}"; do
      #   for layer in "${layers[@]}"; do
      #     for learning_rate in "${learning_rates[@]}"; do
      #       for batch_size in "${batch_sizes[@]}"; do
      #         for n_heads in "${n_headss[@]}"; do
                current_time=$(date +"%Y-%m-%d %H:%M:%S")
                nohup python -u $root_directory/baselines/Time-Series-Library/run_self_traffic.py \
                  --root_path $root_directory/HTSFB_datasets/self/traffic/ \
                  --task_name long_term_forecast \
                  --is_training 1 \
                  --data_path  traffic.csv\
                  --data_topk_path $root_directory/HTSFB_datasets/self/traffic/top-k-train/matrix_rank_train_traffic1.npy\
                  --output_path $root_directory/HTSFB_output/\
                  --checkpoints $root_directory/HTSFB_output/checkpoints/\
                  --model_id traffic_336_"($horizon)" \
                  --model $model_name \
                  --data traffic-reindex \
                  --features M \
                  --seq_len 336 \
                  --label_len 36 \
                  --pred_len $horizon \
                  --patch_len 16 \
                  --patch_stride 8\
                  --e_layers $layer \
                  --d_layers 1 \
                  --factor 3 \
                  --enc_in 862 \
                  --des 'Exp' \
                  --n_heads $n_heads \
                  --d_model $d_model\
                  --d_ff $d_ff\
                  --top_k 5 \
                  --itr 1 \
                  --k $corr\
                  --seasonality 24\
                  --batch_size $batch_size\
                  --learning_rate $learning_rate\
                  --dropout 0.2\
                  --freq 'h' > $root_directory/HTSFB_output/"($model_name)"_traffic336_"($horizon)"_corr"($corr)"_"($current_time)"_"($d_ff)"_"($d_model)"_"($layer)"_"($learning_rate)"_"($batch_size)"_"($n_heads)".out
      #         done
      #       done
      #     done
      #   done
      # done
  #   done
  done
done
