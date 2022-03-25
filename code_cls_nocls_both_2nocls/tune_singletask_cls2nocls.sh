
TASKS=("multi_news" "superglue-copa" "quail" "search_qa" "squad-with_context"
"blimp-anaphor_gender_agreement" "blimp-ellipsis_n_bar_1" "common_gen" "acronym_identification" "aeslc")

CHECKPOINT="None"
size="large"
IDENTIFIER="T5-"$size"-ft-cls2nocls"

for TASK in ${TASKS[@]}
do
  echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
  python -m torch.distributed.launch --nproc_per_node 2 --master_port 24569 tune_singletask_cls2nocls.py \
        --task_dir data/${TASK}/ \
        --task_name ${TASK} \
        --identifier $IDENTIFIER \
        --checkpoint $CHECKPOINT \
        --do_train \
        --do_predict \
        --learning_rate_list 5e-4 3e-4 2e-4 1e-4 \
        --bsz_list 8 \
        --predict_batch_size 32 \
        --total_steps 1000 \
        --eval_period 50 \
        --warmup_steps 50 \
        --num_train_epochs 300.0 \
        --gradient_accumulation_steps 1 \
        --output_dir models/${IDENTIFIER}/singletask-${TASK} \
        --cuda 0,1 \
        --model google/t5-v1_1-$size
  echo "++++++++++++++++++++++++++++++"
	ps aux | grep tune_singletask_cls2nocls.py | awk '{print $2}' | xargs kill -9
done
