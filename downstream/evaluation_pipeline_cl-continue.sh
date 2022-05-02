gpu=$1
model=$2
bert_dir=$3
output_dir=$4
continue_model_path=$5
add1=$6
add2=$7
add3=$8

# ## Dialog State Tracking
# CUDA_VISIBLE_DEVICES=$gpu python main_cl-continue.py \
#     --my_model=BeliefTracker \
#     --model_type=${model} \
#     --dataset='["multiwoz"]' \
#     --task_name="dst" \
#     --earlystop="joint_acc" \
#     --output_dir=${output_dir}/DST-fs/MWOZ \
#     --cache_dir="./save/transformers" \
#     --do_train \
#     --task=dst \
#     --example_type=turn \
#     --model_name_or_path=${bert_dir} \
#     --batch_size=6 --eval_batch_size=6 \
#     --usr_token=[USR] --sys_token=[SYS] \
#     --eval_by_step=4000 --epoch=15\
#     --continue_model_path=${continue_model_path}\
#     --continue_ft \
#     $add1 $add2 $add3
    
## Response Retrieval
CUDA_VISIBLE_DEVICES=$gpu python main_cl-continue.py \
    --my_model=dual_encoder_ranking \
    --do_train \
    --task=nlg \
    --task_name=rs \
    --example_type=turn \
    --model_type=${model} \
    --model_name_or_path=${bert_dir} \
    --output_dir=${output_dir}/RR-fs/MWOZ/ \
    --cache_dir="./save/transformers" \
    --batch_size=24 --eval_batch_size=100 \
    --usr_token=[USR] --sys_token=[SYS] \
    --fix_rand_seed \
    --eval_by_step=1000 --epoch=20 \
    --continue_model_path=${continue_model_path}\
    --continue_ft \
    --max_seq_length=256 \
    $add1 $add2 $add3
