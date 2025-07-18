
# -*- coding: utf-8 -*-

from typing import Any, Dict

import torch.nn as nn

from swift.utils import get_env_args
from ..utils import ModelInfo
from ..register import (Model, ModelGroup, ModelMeta, register_model)
from ..constant import MLLMModelType
from .qwen import get_model_tokenizer_qwen
from swift.llm import TemplateType


def get_model_tokenizer_qwen3_optimus(model_dir: str,
                                      model_info: ModelInfo,
                                      model_kwargs: Dict[str, Any],
                                      load_model: bool = True,
                                      **kwargs):
    # 1. Load Qwen3 base model
    model, tokenizer = get_model_tokenizer_qwen(model_dir, model_info, model_kwargs, load_model, **kwargs)

    # 2. Add special tokens
    special_tokens = {
        'additional_special_tokens': [
            '<|image_start|>', '<|image_end|>',
            '<|patch_start|>', '<|patch_end|>',
            '<|image_pad|>', '<|patch_pad|>',
            '<|image_idx_start|>', '<|image_idx_end|>',
            '<|patch_idx_start|>', '<|patch_idx_end|>'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # 3. Load H-optimus-1 vision encoder
    import timm
    vision_tower = timm.create_model(
        "hf-hub:bioptimus/H-optimus-1", 
        pretrained=True, 
        init_values=1e-5, 
        dynamic_img_size=False
    )
    vision_tower.eval()

    # 4. Initialize Adapter (Projector)
    hidden_size = model.config.hidden_size  # Qwen3 hidden size
    vision_hidden_size = 1536  # H-optimus-1 output size
    downsample_ratio = get_env_args('downsample_ratio', float, 0.5)

    model.vision_tower = vision_tower
    model.vision_projector = nn.Sequential(
        nn.LayerNorm(vision_hidden_size * int(1 / downsample_ratio) ** 2),
        nn.Linear(vision_hidden_size * int(1 / downsample_ratio) ** 2, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size)
    )

    return model, tokenizer


# 注册 Optimus Qwen3 模型
register_model(
    ModelMeta(
        MLLMModelType.optimus_qwen3,
        [ModelGroup([
            Model('Qwen/Qwen3-8B', 'Qwen/Qwen3-8B'),
            Model('Qwen/Qwen3-0.6B', 'Qwen/Qwen3-0.6B'),
        ])],
        template=TemplateType.optimus_qwen3,
        get_function=get_model_tokenizer_qwen3_optimus,
        architectures=['Qwen3ForCausalLM'],
    ))
