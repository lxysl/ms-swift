CUDA_VISIBLE_DEVICES=2 \
swift pt \
    --model_type qwen3-optimus \
    --model Qwen/Qwen3-0.6B \
    --custom_dataset_info wsi-scripts/wsi_pt_dataset_info.json \
    --dataset wsi_pt \
    --freeze_llm true \
    --freeze_vit true \
    --trainable_parameters vision_projector \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --num_train_epochs 1 \
    --dataloader_num_workers 1 \
    --max_length 2048 \
    --output_dir ./outputs/optimus-qwen3-pt \
    --report_to wandb \
    # > ./outputs/optimus-qwen3-pt/train.log 2>&1

    # --per_device_train_batch_size  1 \
    # --save_steps 50 \
    # --save_total_limit 2 \
    # --logging_steps 5 \