size="large"
#allinnerlr=(2e-5 3e-5 5e-5)
allinnerlr=(2e-5)
allgradient=(1 2 4)
allstep=(2500 5000 10000)
#alloutlr=(2e-1 3e-1 5e-1)
alloutlr=(2e-1)
#python -m torch.distributed.launch --nproc_per_node 1 --master_port 29533 cli_maml_ddp_prompt.py \
for onelr in ${allinnerlr[@]}
do
	for oneg in ${allgradient[@]}
	do
		for onestep in ${allstep[@]}
		do
			for outerlr in ${alloutlr[@]}
			do
			python cli_maml_ddp_prompt_tune1.py \
			--output_dir models/upstream-maml-random-$onelr-$oneg-$onestep-$outerlr \
			--custom_tasks_splits ./dataloader/custom_tasks_splits/random.json \
			--do_train \
        		--prompt_number 100 \
        		--cuda 0 \
        		--inner_bsz 2 \
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
