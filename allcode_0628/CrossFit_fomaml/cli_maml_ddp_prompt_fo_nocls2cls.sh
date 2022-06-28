size="large"
allinnerlr=(3e-5)
allgradient=(2)
allstep=(5000)
alloutlr=(5e-1)
for onelr in ${allinnerlr[@]}
do
        for oneg in ${allgradient[@]}
        do
                for onestep in ${allstep[@]}
                do
                        for outerlr in ${alloutlr[@]}
                        do
                              python cli_maml_ddp_prompt_fo_nocls2cls.py \
                                --output_dir models/upstream-fomaml-nocls2cls-$onelr-$oneg-$onestep-$outerlr \
                                --custom_tasks_splits ./dataloader/custom_tasks_splits/train_nonclassification_test_classification.json \
                                --do_train \
                                --prompt_number 100 \
                                --cuda 3 \
                                --inner_bsz 4 \
                                --inner_lr $onelr \
                                --gradient_accumulation_steps $oneg \
                                --num_train_epochs 120.0 \
                                --total_steps $onestep \
                                --learning_rate $outerlr \
                                --eval_period 10 \
                                --identifier $size \
                                --model google/t5-v1_1-$size \
                                --lm_adapted_path /export/share/sjoty/continual-learning/lm_adapted_model/torch_ckpt/$size/pytorch_model.bin
			                  done
                done
        done
done

