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
                              python cli_maml_ddp_prompt_fo_nopara2para_bart.py \
                                --output_dir models/upstream-fomaml-nopara2para-$onelr-$oneg-$onestep-$outerlr-bart \
                                --custom_tasks_splits ./dataloader/custom_tasks_splits/train_nonparaphrase_classification_test_paraphrase.json \
                                --do_train \
                                --prompt_number 100 \
                                --cuda 2 \
                                --inner_bsz 4 \
                                --inner_lr $onelr \
                                --gradient_accumulation_steps $oneg \
                                --num_train_epochs 120.0 \
                                --total_steps $onestep \
                                --learning_rate $outerlr \
                                --eval_period 10 \
                                --identifier $size \
                                --model facebook/bart-$size
			                  done
                done
        done
done

