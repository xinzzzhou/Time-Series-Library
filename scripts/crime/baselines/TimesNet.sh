export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet
# root_directory=$(dirname "$(dirname "$(pwd)")")
root_directory='/home/xinz/ar57_scratch/xinz/HTSFB_project/'
current_time=$(date +"%Y-%m-%d %H:%M:%S")

#dm_256, el_2, nh_4, bz_32, df_384, lr0.001
d_ff=384
d_model=256 # d_models=(64 256 384)laoshi
layers=(1 2)
learning_rate=0.001 # learning_rates=(0.1 0.01 0.0001)
batch_size=32 # batch_sizes=(16 32 64 128)
n_heads=4 # n_headss=(1 2 3 5 6 8 16)
horizons=(6 12 18 24)
for horizon in "${horizons[@]}"; do
  # for d_ff in "${d_ffs[@]}"; do
  #   for d_model in "${d_models[@]}"; do
      for layer in "${layers[@]}"; do
        # for learning_rate in "${learning_rates[@]}"; do
        #   for batch_size in "${batch_sizes[@]}"; do
        #     for n_heads in "${n_headss[@]}"; do
              current_time=$(date +"%Y-%m-%d %H:%M:%S")
              nohup python -u $root_directory/baselines/Time-Series-Library/run_self.py \
                --root_path $root_directory/HTSFB_datasets/self/CrimeChicago/region/ \
                --task_name long_term_forecast \
                --is_training 1 \
                --data_path  Crimes-2001_2023.csv\
                --output_path $root_directory/HTSFB_output/\
                --checkpoints $root_directory/HTSFB_output/checkpoints/\
                --model_id crime_24_"($horizon)" \
                --model $model_name \
                --data crime \
                --features M \
                --seq_len 24 \
                --label_len 6 \
                --pred_len $horizon \
                --e_layers $layer \
                --d_layers 1 \
                --factor 3 \
                --enc_in 1155 \
                --c_out 1155 \
                --des 'Exp' \
                --n_heads $n_heads \
                --d_model $d_model\
                --d_ff $d_ff\
                --top_k 5 \
                --itr 1 \
                --batch_size $batch_size\
                --learning_rate $learning_rate\
                --freq 'm' > $root_directory/HTSFB_output/"($model_name)"_crime_24_"($horizon)"_"($current_time)"_"($d_ff)"_"($d_model)"_"($layer)"_"($learning_rate)"_"($batch_size)"_"($n_heads)".out
    #         done
    #       done
    #     done
    #   done
    # done
  done
done