size="base"
allinnerlr=(3e-5)
allgradient=(2)
allstep=(5000)
alloutlr=(5e-1)
allreptilestep=(10)
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
					    python cli_reptile_ddp_prompt_nopara2para.py \
					      --output_dir models/upstream-reptile-nopara2para-$onelr-$oneg-$onestep-$outerlr-$onerep-t5base \
					      --custom_tasks_splits ./dataloader/custom_tasks_splits/train_nonparaphrase_classification_test_paraphrase.json \
					      --do_train \
        				--prompt_number 100 \
        				--cuda 6 \
        				--inner_bsz 4 \
					      --reptile_step $onerep \
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
done
