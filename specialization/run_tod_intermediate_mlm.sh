gpu=$1
bert_dir=$2
output_dir=$3
train_file=$4
val_file=$5
add1=$6
add2=$7
add3=$8
add4=$9

CUDA_VISIBLE_DEVICES=$gpu python run_mlm.py \
    --model_name_or_path=${bert_dir} \
    --output_dir=${output_dir} \
    --train_file=${train_file} \
    --validation_file=${val_file} \
    --cache_dir="./save/transformers" \
    --line_by_line \
    --do_train \
    --do_eval \
    --fp16 \
    --load_best_model_at_end \
    --max_train_samples=100000 \
    --max_val_samples=10000\
    --learning_rate=1e-5 --evaluation_strategy="epoch"\
    --save_steps=2500 --logging_steps=100 --gradient_accumulation_steps=4\
    --num_train_epochs=30 --save_total_limit 2\
    --per_device_train_batch_size=4 --max_seq_length=256\
    ${add1} ${add2} ${add3} ${add4}