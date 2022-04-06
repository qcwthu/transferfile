
TASKS=("anli" "glue-mnli" "glue-qnli" "glue-rte" "glue-wnli" "scitail" "sick" "superglue-cb" "superglue-rte")

allinnerlr=(3e-5)
allgradient=(2)
allstep=(5000)
alloutlr=(5e-1)
allreptilestep=(10)


size="large"
for onelr in ${allinnerlr[@]}
do
        for oneg in ${allgradient[@]}
        do
                for onestep in ${allstep[@]}
                do
                        for outerlr in ${alloutlr[@]}
                        do
                            for onerep in ${allreptilestep[@]}
                            do
                            CHECKPOINT="models/upstream-reptile-nonli2nli-"$onelr"-"$oneg"-"$onestep"-"$outerlr"-"$onerep"/last-model.pt"
                            IDENTIFIER="T5-"$size"-reptile-nonli2nli-"$onelr"-"$oneg"-"$onestep"-"$outerlr"-"$onerep
                            for TASK in ${TASKS[@]}
                            do
                              echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
                              python -m torch.distributed.launch --nproc_per_node 2 --master_port 28416 singletask_from_reptile_nonli2nli.py \
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
                                  ps aux | grep singletask_from_reptile_nonli2nli.py | awk '{print $2}' | xargs kill -9
                            done
			                  done
			              done
                done
        done
done


