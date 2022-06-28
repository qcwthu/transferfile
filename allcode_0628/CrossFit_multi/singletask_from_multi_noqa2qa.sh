
#TASKS=("ai2_arc" "codah" "cosmos_qa" "dream" "hellaswag" "openbookqa" "qasc" "quail" "quarel" "quartz-no_knowledge" "quartz-with_knowledge" "sciq" "superglue-copa" "swag" "wino_grande" "wiqa")

TASKS=("superglue-copa" "swag" "wino_grande" "wiqa")

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
                      CHECKPOINT="models/upstream-multitask-noqa2qa-"$onelr"-"$onebs"-"$oneepoch"/last-model.pt"
                      IDENTIFIER="T5-"$size"-multitask-noqa2qa-"$onelr"-"$onebs"-"$oneepoch
                      for TASK in ${TASKS[@]}
                      do
                          echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
                          python -m torch.distributed.launch --nproc_per_node 2 --master_port 26238 singletask_from_multi_noqa2qa.py \
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
                            ps aux | grep singletask_from_multi_noqa2qa.py | awk '{print $2}' | xargs kill -9
			                done
		            done
        done
done

