CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift pt \
    --model_type qwen3-optimus \
    --model Qwen/Qwen3-0.6B \
    --custom_dataset_info wsi_pt_dataset_info.json \
    --freeze_llm true \
    --freeze_vit true \
    --trainable_parameters vision_projector \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --output_dir outputs/optimus-qwen3-pt
