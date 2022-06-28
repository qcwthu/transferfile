
#TASKS=("superglue-cb" "dbpedia_14" "wiki_qa" "emo" "yelp_polarity" "ethos-religion" "amazon_polarity" "tab_fact" "anli" "ethos-race")
TASKS=("dbpedia_14" "wiki_qa" "emo" "yelp_polarity" "ethos-religion" "amazon_polarity" "tab_fact" "anli" "ethos-race")

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
                      CHECKPOINT="models/upstream-multitask-nocls2cls-"$onelr"-"$onebs"-"$oneepoch"/last-model.pt"
                      IDENTIFIER="T5-"$size"-multitask-nocls2cls-"$onelr"-"$onebs"-"$oneepoch
                      for TASK in ${TASKS[@]}
                      do
                          echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
                          python -m torch.distributed.launch --nproc_per_node 2 --master_port 26127 singletask_from_multi_nocls2cls.py \
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
                              --cuda 6,7 \
                              --lm_adapted_path /export/share/sjoty/continual-learning/lm_adapted_model/torch_ckpt/$size/pytorch_model.bin \
                              --model google/t5-v1_1-$size \
                              --prompt_number 100
                            echo "++++++++++++++++++++++++++++++"
                            ps aux | grep singletask_from_multi_nocls2cls.py | awk '{print $2}' | xargs kill -9
			                done
		            done
        done
done

