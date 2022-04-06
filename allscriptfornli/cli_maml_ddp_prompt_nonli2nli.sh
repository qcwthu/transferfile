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
            python cli_maml_ddp_prompt_nonli2nli.py \
              --output_dir models/upstream-maml-nonli2nli-$onelr-$oneg-$onestep-$outerlr \
              --custom_tasks_splits ./dataloader/custom_tasks_splits/train_non_nli_test_nli.json \
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
