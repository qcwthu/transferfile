
TASKS=("ai2_arc" "codah" "cosmos_qa" "dream" "hellaswag" "openbookqa" "qasc" "quail" "quarel" "quartz-no_knowledge" "quartz-with_knowledge" "sciq" "superglue-copa" "swag" "wino_grande" "wiqa")


CHECKPOINT="None"
size="large"
IDENTIFIER="T5-"$size"-ft-qa2qa"

for TASK in ${TASKS[@]}
do
  echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
  python -m torch.distributed.launch --nproc_per_node 2 --master_port 27667 tune_singletask_qa2qa.py \
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
        --cuda 4,5 \
        --model google/t5-v1_1-$size
  echo "++++++++++++++++++++++++++++++"
	ps aux | grep tune_singletask_qa2qa.py | awk '{print $2}' | xargs kill -9
done
