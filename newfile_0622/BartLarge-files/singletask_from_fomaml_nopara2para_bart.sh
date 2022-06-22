

TASKS=( "glue-mrpc" "glue-qqp" "medical_questions_pairs" "paws")

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
                            CHECKPOINT="models/upstream-fomaml-nopara2para-"$onelr"-"$oneg"-"$onestep"-"$outerlr"-bart/last-model.pt"
                            IDENTIFIER="Bart-"$size"-fomaml-nopara2para-"$onelr"-"$oneg"-"$onestep"-"$outerlr
                            for TASK in ${TASKS[@]}
                            do
                              echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
                              python -m torch.distributed.launch --nproc_per_node 2 --master_port 28423 singletask_from_fomaml_nopara2para_bart.py \
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
                                  --model facebook/bart-$size \
                                  --prompt_number 100
                                  echo "++++++++++++++++++++++++++++++"
                                  ps aux | grep singletask_from_fomaml_nopara2para_bart.py | awk '{print $2}' | xargs kill -9
				                    done
			                done
                done
        done
done










