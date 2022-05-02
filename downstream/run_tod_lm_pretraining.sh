gpu=$1
model_type=$2
bert_dir=$3
output_dir=$4
add1=$5
add2=$6
add3=$7
add4=$8
add5=$9
# For XLMR
CUDA_VISIBLE_DEVICES=$gpu python tod_xlmr_pretraining.py \
    --task=usdl \
    --model_type=${model_type} \
    --model_name_or_path=${bert_dir} \
    --output_dir=${output_dir} \
    --cache_dir="./save/transformers" \
    --do_train \
    --do_eval \
    --mlm \
    --evaluate_during_training \
    --save_steps=2500 --logging_steps=800 --gradient_accumulation_steps=4\
    --per_gpu_train_batch_size=2 --per_gpu_eval_batch_size=2 --max_seq_length=512 \
    ${add1} ${add2} ${add3} ${add4} ${add5}
