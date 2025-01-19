# Titans_project/modules/vit_eagle.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor

class PretrainedViTEagle(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224", freeze=True):
        super().__init__()
        # Fast image processor 사용
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)  # do_rescale 제거
        self.model = AutoModel.from_pretrained(model_name)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, images):
        """
        images: List of PIL Images
        """
        device = next(self.model.parameters()).device
        # images는 PIL 이미지의 리스트입니다.
        inputs = self.processor(images=images, return_tensors="pt")  # do_rescale=True 제거
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output  # [B, hidden_size]
        else:
            return outputs.last_hidden_state.mean(dim=1)  # [B, hidden_size]
