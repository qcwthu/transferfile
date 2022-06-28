#TASKS=("tweet_eval-emotion" "financial_phrasebank" "tweet_qa" "empathetic_dialogues" "cos_e")

#TASKS=("ag_news" "ethos-disability" "superglue-rte" "glue-cola" "ethos-sexual_orientation" "ai2_arc" "crawl_domain" "samsum" "blimp-sentential_negation_npi_scope" "quoref" "amazon_polarity" "blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony")

#TASKS=("ai2_arc" "crawl_domain" "samsum" "blimp-sentential_negation_npi_scope" "quoref" "amazon_polarity" "blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony")


#TASKS=("quoref" "amazon_polarity" "blimp-sentential_negation_npi_licensor_present")

#TASKS=("glue-qnli" "freebase_qa" "race-high" "break-QDMR" "hatexplain" "circa" "wiki_split" "yelp_polarity")

TASKS=("wiki_split" "yelp_polarity")

size="large"
allepochs=(20)
allbatchsize=(4)
alllr=(5e-1)


for onelr in ${alllr[@]}
do
        for onebs in ${allbatchsize[@]}
        do
                for oneepoch in ${allepochs[@]}
                do
			CHECKPOINT="models/upstream-multitask-random-"$onelr"-"$onebs"-"$oneepoch"/last-model.pt"
                        IDENTIFIER="T5-"$size"-multitaski-random-"$onelr"-"$onebs"-"$oneepoch
			for TASK in ${TASKS[@]}
			do
  				echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
  				python -m torch.distributed.launch --nproc_per_node 2 --master_port 26125 singletask_from_multi_random.py \
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
        			--cuda 2,3 \
       				--lm_adapted_path /export/share/sjoty/continual-learning/lm_adapted_model/torch_ckpt/$size/pytorch_model.bin \
        			--model google/t5-v1_1-$size \
 			        --prompt_number 100
     				echo "++++++++++++++++++++++++++++++"
      			ps aux | grep singletask_from_multi_random.py | awk '{print $2}' | xargs kill -9
			done
		done
        done
done

