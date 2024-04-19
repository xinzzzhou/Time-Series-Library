export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
# root_directory=$(dirname "$(dirname "$(pwd)")")
root_directory='/home/xinz/ar57_scratch/xinz/HTSFB_project/'
current_time=$(date +"%Y-%m-%d %H:%M:%S")


d_ff=512 
d_model=512 
layer=4
learning_rate=0.001 
batch_size=8
n_heads=16 
horizons=(96 192 336 720)
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
                --output_path $root_directory/HTSFB_output/\
                --checkpoints $root_directory/HTSFB_output/checkpoints/\
                --model_id traffic_96_"($horizon)" \
                --model $model_name \
                --data custom \
                --features M \
                --seq_len 96 \
                --label_len 36 \
                --pred_len $horizon \
                --e_layers $layer \
                --enc_in 862 \
                --dec_in 862 \
                --c_out 862 \
                --des 'Exp' \
                --n_heads $n_heads \
                --d_model $d_model\
                --d_ff $d_ff\
                --itr 1 \
                --batch_size $batch_size\
                --learning_rate $learning_rate\
                --dropout 0.1\
                --freq 'h' > $root_directory/HTSFB_output/"($model_name)"_traffic96_"($horizon)"_"($current_time)"_"($d_ff)"_"($d_model)"_"($layer)"_"($learning_rate)"_"($batch_size)"_"($n_heads)".out
    #         done
    #       done
    #     done
    #   done
    # done
#   done
done
