#TASKS=("superglue-cb" "dbpedia_14" "wiki_qa" "emo" "yelp_polarity" "ethos-religion" "financial_phrasebank" "tab_fact" "anli" "ethos-race")
TASKS=("tweet_eval-emotion" "financial_phrasebank" "tweet_qa" "empathetic_dialogues" "cos_e")
#allinnerlr=(2e-5 3e-5 5e-5)
allinnerlr=(5e-5)
#allgradient=(1 2 4)
allgradient=(1)
allstep=(2500 5000 10000)
#alloutlr=(2e-1 3e-1 5e-1)
alloutlr=(2e-1 3e-1)

#CHECKPOINT="models/upstream-maml-noncls2cls/last-model.pt"

#size="base"
size="large"
#IDENTIFIER="T5-"$size-"maml"
#IDENTIFIER="T5-"$size-"fomaml"
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
 		 		python -m torch.distributed.launch --nproc_per_node 2 --master_port 29810 singletask_from_meta_3.py \
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
        			--cuda 4,5 \
       				--lm_adapted_path /export/share/sjoty/continual-learning/lm_adapted_model/torch_ckpt/$size/pytorch_model.bin \
        			--model google/t5-v1_1-$size \
        			--prompt_number 100
      				echo "++++++++++++++++++++++++++++++"
      				ps aux | grep singletask_from_meta_3.py | awk '{print $2}' | xargs kill -9
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
                                python -m torch.distributed.launch --nproc_per_node 2 --master_port 29810 singletask_from_meta_3.py \
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
                                --cuda 4,5 \
                                --lm_adapted_path /export/share/sjoty/continual-learning/lm_adapted_model/torch_ckpt/$size/pytorch_model.bin \
                                --model google/t5-v1_1-$size \
                                --prompt_number 100
                                echo "++++++++++++++++++++++++++++++"
                                ps aux | grep singletask_from_meta_3.py | awk '{print $2}' | xargs kill -9
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
                                python -m torch.distributed.launch --nproc_per_node 2 --master_port 29810 singletask_from_meta_3.py \
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
                                --cuda 4,5 \
                                --lm_adapted_path /export/share/sjoty/continual-learning/lm_adapted_model/torch_ckpt/$size/pytorch_model.bin \
                                --model google/t5-v1_1-$size \
                                --prompt_number 100
                                echo "++++++++++++++++++++++++++++++"
                                ps aux | grep singletask_from_meta_3.py | awk '{print $2}' | xargs kill -9
                                done
                         done
                done
        done
done
