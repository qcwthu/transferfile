
TASKS=( "glue-mrpc" "glue-qqp" "medical_questions_pairs" "paws")

CHECKPOINT="None"
size="base"
IDENTIFIER="T5-"$size"-ft-nopara2para"

for TASK in ${TASKS[@]}
do
  echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
  python -m torch.distributed.launch --nproc_per_node 2 --master_port 21389 tune_singletask_nopara2para.py \
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
        --cuda 6,7 \
        --model google/t5-v1_1-$size
  echo "++++++++++++++++++++++++++++++"
	ps aux | grep tune_singletask_nopara2para.py | awk '{print $2}' | xargs kill -9
done
