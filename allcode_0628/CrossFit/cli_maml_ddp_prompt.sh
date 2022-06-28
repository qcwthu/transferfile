size="large"
python -m torch.distributed.launch --nproc_per_node 1 --master_port 29533 cli_maml_ddp_prompt.py \
        --output_dir models/upstream-maml \
        --do_train \
        --prompt_number 100 \
        --cuda 0 \
        --inner_bsz 2 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 200.0 \
        --total_steps 10000 \
        --eval_period 10 \
        --identifier $size \
        --model google/t5-v1_1-$size \
        --lm_adapted_path /export/share/sjoty/continual-learning/lm_adapted_model/torch_ckpt/$size/pytorch_model.bin
