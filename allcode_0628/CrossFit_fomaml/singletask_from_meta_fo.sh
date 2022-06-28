TASKS=("ag_news" "quoref" "wiki_split" "ethos-disability" "yelp_polarity" "superglue-rte" "glue-cola" "ethos-sexual_orientation" "blimp-sentential_negation_npi_scope" "ai2_arc" "amazon_polarity" "race-high" "blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony" "break-QDMR" "crawl_domain" "freebase_qa" "glue-qnli" "hatexplain" "circa")

####跑完ag_news" "wiki_split" "ethos-disability" "superglue-rte"了
#TASKS=("glue-cola" "ethos-sexual_orientation" "quoref" "yelp_polarity" "blimp-sentential_negation_npi_scope"
#"ai2_arc" "amazon_polarity" "race-high" "blimp-sentential_negation_npi_licensor_present"
#"tweet_eval-irony" "break-QDMR" "crawl_domain" "freebase_qa" "glue-qnli" "hatexplain" "circa")

####跑完"ag_news" "wiki_split" "ethos-disability" "superglue-rte" "glue-cola" "ethos-sexual_orientation" "ai2_arc"
#TASKS=("ag_news" "wiki_split" "ethos-disability" "superglue-rte" "glue-cola" "ethos-sexual_orientation" "ai2_arc" "crawl_domain" "samsum" "blimp-sentential_negation_npi_scope" "quoref" "yelp_polarity" "amazon_polarity" "race-high" "blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony" "break-QDMR" "freebase_qa" "glue-qnli" "hatexplain" "circa")

#TASKS=("superglue-rte" "glue-cola" "ethos-sexual_orientation" "ai2_arc" "crawl_domain" "samsum" "blimp-sentential_negation_npi_scope" "quoref" "yelp_polarity" "amazon_polarity" "race-high" "blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony" "break-QDMR" "freebase_qa" "glue-qnli" "hatexplain" "circa")
#TASKS=("tweet_eval-irony" "break-QDMR" "freebase_qa" "glue-qnli" "hatexplain" "circa")
#TASKS=("break-QDMR")
#TASKS=("superglue-cb" "dbpedia_14" "wiki_qa" "emo" "yelp_polarity" "ethos-religion" "financial_phrasebank" "tab_fact" "anli" "ethos-race")
TASKS=("superglue-cb" "yelp_polarity" "ethos-race")

#TASKS=("ag_news" "superglue-rte" "glue-cola" "ethos-sexual_orientation" "ai2_arc" "crawl_domain" "samsum"
#"quoref" "yelp_polarity" "blimp-sentential_negation_npi_scope" "amazon_polarity" "race-high"
#"blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony" "break-QDMR" "freebase_qa"
#"glue-qnli" "hatexplain" "circa")


#TASKS=("ethos-disability" "superglue-rte" "glue-cola" "ethos-sexual_orientation" "ai2_arc" "samsum" "crawl_domain"
#"quoref" "yelp_polarity" "blimp-sentential_negation_npi_scope" "amazon_polarity" "race-high"
#"blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony" "break-QDMR" "freebase_qa"
#"glue-qnli" "hatexplain" "circa")
#TASKS=("ethos-disability" "superglue-rte" "glue-cola" "ethos-sexual_orientation" "ai2_arc" "crawl_domain" "samsum"
#"quoref" "yelp_polarity" "blimp-sentential_negation_npi_scope" "amazon_polarity" "race-high"
#"blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony" "break-QDMR" "freebase_qa"
#"glue-qnli" "hatexplain" "circa")
#TASKS=("ag_news" "quoref" "wiki_split" "ethos-disability" "yelp_polarity" "superglue-rte" "glue-cola" "ethos-sexual_orientation"
#"blimp-sentential_negation_npi_scope" "ai2_arc" "amazon_polarity" "race-high" "blimp-sentential_negation_npi_licensor_present"
#"tweet_eval-irony" "break-QDMR" "crawl_domain" "freebase_qa" "glue-qnli" "hatexplain" "circa")

#CHECKPOINT="None"
#CHECKPOINT="models/upstream-maml/last-model.pt"
#CHECKPOINT="models/upstream-fomaml/last-model.pt"
#CHECKPOINT="models/upstream-fomaml-cls2cls/last-model.pt"
CHECKPOINT="models/upstream-fomaml-noncls2cls/last-model.pt"


#size="base"
size="large"
#IDENTIFIER="T5-"$size-"maml"
IDENTIFIER="T5-"$size-"fomaml"

for TASK in ${TASKS[@]}
do
  echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
  python -m torch.distributed.launch --nproc_per_node 2 --master_port 29589 singletask_from_meta_fo.py \
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
      ps aux | grep singletask_from_meta_fo.py | awk '{print $2}' | xargs kill -9
done
