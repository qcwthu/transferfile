size="large"
#python -m torch.distributed.launch --nproc_per_node 1 --master_port 29533 cli_maml_ddp_prompt.py \
python cli_maml_ddp_prompt.py \
	--output_dir models/upstream-maml-noncls2cls \
	--custom_tasks_splits ./dataloader/custom_tasks_splits/train_nonclassification_test_classification.json \
	--do_train \
        --prompt_number 100 \
        --cuda 2 \
        --inner_bsz 2 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 120.0 \
        --total_steps 4500 \
        --eval_period 10 \
	--identifier $size \
        --model google/t5-v1_1-$size \
        --lm_adapted_path /export/share/sjoty/continual-learning/lm_adapted_model/torch_ckpt/$size/pytorch_model.bin
