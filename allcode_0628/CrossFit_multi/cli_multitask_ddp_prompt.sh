size="large"
python -m torch.distributed.launch --nproc_per_node 1 --master_port 29530 cli_multitask_ddp_prompt.py \
        --output_dir models/upstream-multitask-noncls2cls \
        --custom_tasks_splits ./dataloader/custom_tasks_splits/train_nonclassification_test_classification.json \
	--identifier $size \
        --do_train \
        --prompt_number 100 \
        --train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --cuda 2 \
        --model google/t5-v1_1-$size \
        --lm_adapted_path /export/share/sjoty/continual-learning/lm_adapted_model/torch_ckpt/$size/pytorch_model.bin
