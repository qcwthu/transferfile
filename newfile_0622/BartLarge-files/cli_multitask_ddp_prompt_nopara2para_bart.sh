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
                    python -m torch.distributed.launch --nproc_per_node 1 --master_port 29561 cli_multitask_ddp_prompt_nopara2para_bart.py \
                          --output_dir models/upstream-multitask-nopara2para-$onelr-$onebs-$oneepoch-bart  \
                          --custom_tasks_splits ./dataloader/custom_tasks_splits/train_nonparaphrase_classification_test_paraphrase.json \
                          --identifier $size \
                          --do_train \
                          --prompt_number 100 \
                          --train_batch_size $onebs \
                          --num_train_epochs $oneepoch \
                          --learning_rate $onelr \
                          --gradient_accumulation_steps 1 \
                          --cuda 6 \
                          --model facebook/bart-$size
                    ps aux | grep cli_multitask_ddp_prompt_nopara2para_bart.py | awk '{print $2}' |xargs kill -9
		            done
	      done
done
