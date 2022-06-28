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
			python -m torch.distributed.launch --nproc_per_node 1 --master_port 32531 cli_multitask_ddp_prompt_tune1.py \
        		--output_dir models/upstream-50prompt-multitask-cls2cls-$onelr-$onebs-$oneepoch  \
        		--custom_tasks_splits ./dataloader/custom_tasks_splits/train_classification_test_classification.json \
			--identifier $size \
        		--do_train \
        		--prompt_number 50 \
        		--train_batch_size $onebs \
			--num_train_epochs $oneepoch \
			--learning_rate $onelr \
        		--gradient_accumulation_steps 1 \
        		--cuda 4 \
        		--model google/t5-v1_1-$size \
        		--lm_adapted_path /export/share/sjoty/continual-learning/lm_adapted_model/torch_ckpt/$size/pytorch_model.bin
			ps aux | grep cli_multitask_ddp_prompt_tune1.py | awk '{print $2}' |xargs kill -9
		done
	done
done
