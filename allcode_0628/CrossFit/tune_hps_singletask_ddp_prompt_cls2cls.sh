


#TASKS=("superglue-cb" "dbpedia_14" "wiki_qa" "emo" "yelp_polarity" "ethos-religion" "amazon_polarity" "tab_fact" "anli" "ethos-race")

TASKS=("anli")

CHECKPOINT="None"
#CHECKPOINT="models/upstream-multitask/last-model.pt"

#size="base"
size="large"

IDENTIFIER="T5-"$size"-cls2cls"

for TASK in ${TASKS[@]}
do
  echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
  python -m torch.distributed.launch --nproc_per_node 2 --master_port 29111 tune_hps_singletask_ddp_prompt_cls2cls.py \
        --task_dir data/${TASK}/ \
        --task_name ${TASK} \
        --identifier $IDENTIFIER \
        --checkpoint $CHECKPOINT \
        --do_train \
        --do_predict \
        --learning_rate_list 5e-1 4e-1 3e-1 2e-1 \
        --bsz_list 8 \
        --predict_batch_size 16 \
        --total_steps 3000 \
        --eval_period 50 \
        --warmup_steps 50 \
        --num_train_epochs 1000.0 \
        --gradient_accumulation_steps 1 \
        --output_dir models/${IDENTIFIER}/singletask-${TASK} \
        --cuda 0,1 \
        --lm_adapted_path /export/share/sjoty/continual-learning/lm_adapted_model/torch_ckpt/$size/pytorch_model.bin \
        --model google/t5-v1_1-$size \
        --prompt_number 100
      echo "++++++++++++++++++++++++++++++"
      ps aux | grep tune_hps_singletask_ddp_prompt_cls2cls.py | awk '{print $2}' | xargs kill -9
done









TASKS=("yelp_polarity")

CHECKPOINT="None"

#size="base"
size="large"

IDENTIFIER="T5-"$size"-cls2cls"

for TASK in ${TASKS[@]}
do
  echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
  python -m torch.distributed.launch --nproc_per_node 2 --master_port 29113 tune_hps_singletask_ddp_prompt_cls2cls.py \
        --task_dir data/${TASK}/ \
        --task_name ${TASK} \
        --identifier $IDENTIFIER \
        --checkpoint $CHECKPOINT \
        --do_train \
        --do_predict \
        --learning_rate_list 5e-1 4e-1 3e-1 2e-1 \
        --bsz_list 4 8 \
        --predict_batch_size 16 \
        --total_steps 3000 \
        --eval_period 50 \
        --warmup_steps 50 \
        --num_train_epochs 1000.0 \
        --gradient_accumulation_steps 1 \
        --output_dir models/${IDENTIFIER}/singletask-${TASK} \
        --cuda 0,1 \
        --lm_adapted_path /export/share/sjoty/continual-learning/lm_adapted_model/torch_ckpt/$size/pytorch_model.bin \
        --model google/t5-v1_1-$size \
        --prompt_number 100
      echo "++++++++++++++++++++++++++++++"
      ps aux | grep tune_hps_singletask_ddp_prompt_cls2cls.py | awk '{print $2}' | xargs kill -9
done
