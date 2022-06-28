
#TASKS=("ag_news" "ethos-disability" "superglue-rte" "glue-cola" "ethos-sexual_orientation" "ai2_arc" "crawl_domain" "samsum" "blimp-sentential_negation_npi_scope" "quoref" "amazon_polarity" "blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony")


#TASKS=("crawl_domain" "samsum" "blimp-sentential_negation_npi_scope" "quoref" "amazon_polarity" "blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony")

#TASKS=("blimp-sentential_negation_npi_licensor_present" "crawl_domain" "amazon_polarity" "tweet_eval-irony")


TASKS=("race-high" "glue-qnli" "freebase_qa" "break-QDMR" "hatexplain" "circa" "wiki_split" "yelp_polarity")


CHECKPOINT="None"
size="large"
IDENTIFIER="T5-"$size"-ft-random"

for TASK in ${TASKS[@]}
do
  echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
  python -m torch.distributed.launch --nproc_per_node 2 --master_port 24566 tune_singletask_random.py \
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
	ps aux | grep tune_singletask_random.py | awk '{print $2}' | xargs kill -9
done
