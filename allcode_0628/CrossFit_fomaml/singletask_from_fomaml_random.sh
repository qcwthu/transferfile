#TASKS=("ag_news" "wiki_split" "ethos-disability" "superglue-rte" "glue-cola" "ethos-sexual_orientation" "ai2_arc" "crawl_domain" "samsum" "blimp-sentential_negation_npi_scope" "quoref" "yelp_polarity" "amazon_polarity" "race-high" "blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony" "break-QDMR" "freebase_qa" "glue-qnli" "hatexplain" "circa")
#TASKS=("ag_news" "ethos-disability" "superglue-rte" "glue-cola" "ethos-sexual_orientation" "ai2_arc" "crawl_domain" "samsum" "blimp-sentential_negation_npi_scope" "quoref" "amazon_polarity" "blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony")

#TASKS=("ai2_arc" "crawl_domain" "samsum" "blimp-sentential_negation_npi_scope" "quoref" "amazon_polarity" "blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony")


#TASKS=("quoref" "amazon_polarity" "blimp-sentential_negation_npi_licensor_present" "tweet_eval-irony")


TASKS=("freebase_qa" "race-high" "glue-qnli" "break-QDMR" "hatexplain" "circa" "wiki_split" "yelp_polarity")

allinnerlr=(3e-5)
allgradient=(2)
allstep=(5000)
alloutlr=(5e-1)

size="large"
for onelr in ${allinnerlr[@]}
do
        for oneg in ${allgradient[@]}
        do
                for onestep in ${allstep[@]}
                do
                        for outerlr in ${alloutlr[@]}
                        do
				CHECKPOINT="models/upstream-fomaml-random-"$onelr"-"$oneg"-"$onestep"-"$outerlr"/last-model.pt"
				IDENTIFIER="T5-"$size"-fomaml-random-"$onelr"-"$oneg"-"$onestep"-"$outerlr
				for TASK in ${TASKS[@]}
				do
  				echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
 		 		python -m torch.distributed.launch --nproc_per_node 2 --master_port 24568 singletask_from_fomaml_random.py \
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
        			--cuda 4,5 \
       				--lm_adapted_path /export/share/sjoty/continual-learning/lm_adapted_model/torch_ckpt/$size/pytorch_model.bin \
        			--model google/t5-v1_1-$size \
        			--prompt_number 100
      				echo "++++++++++++++++++++++++++++++"
      				ps aux | grep singletask_from_fomaml_random.py | awk '{print $2}' | xargs kill -9
				done
			 done
                done
        done
done















allgradient=(2)
for onelr in ${allinnerlr[@]}
do
        for oneg in ${allgradient[@]}
        do
                for onestep in ${allstep[@]}
                do
                        for outerlr in ${alloutlr[@]}
                        do
                                CHECKPOINT="models/upstream-maml-random-"$onelr"-"$oneg"-"$onestep"-"$outerlr"/last-model.pt"
                                IDENTIFIER="T5-"$size"-maml-"$onelr"-"$oneg"-"$onestep"-"$outerlr
                                for TASK in ${TASKS[@]}
                                do
                                echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
                                python -m torch.distributed.launch --nproc_per_node 2 --master_port 29588 singletask_from_meta_1.py \
                                --task_dir data/${TASK}/ \
                                --task_name ${TASK} \
                                --identifier $IDENTIFIER \
                                --checkpoint $CHECKPOINT \
                                --do_train \
                                --do_predict \
                                --learning_rate_list 5e-1 2e-1 \
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
                                ps aux | grep singletask_from_meta_1.py | awk '{print $2}' | xargs kill -9
                                done
                         done
                done
        done
done















allgradient=(4)
for onelr in ${allinnerlr[@]}
do
        for oneg in ${allgradient[@]}
        do
                for onestep in ${allstep[@]}
                do
                        for outerlr in ${alloutlr[@]}
                        do
                                CHECKPOINT="models/upstream-maml-random-"$onelr"-"$oneg"-"$onestep"-"$outerlr"/last-model.pt"
                                IDENTIFIER="T5-"$size"-maml-"$onelr"-"$oneg"-"$onestep"-"$outerlr
                                for TASK in ${TASKS[@]}
                                do
                                echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
                                python -m torch.distributed.launch --nproc_per_node 2 --master_port 29588 singletask_from_meta_1.py \
                                --task_dir data/${TASK}/ \
                                --task_name ${TASK} \
                                --identifier $IDENTIFIER \
                                --checkpoint $CHECKPOINT \
                                --do_train \
                                --do_predict \
                                --learning_rate_list 5e-1 2e-1 \
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
                                ps aux | grep singletask_from_meta_1.py | awk '{print $2}' | xargs kill -9
                                done
                         done
                done
        done
done
