# Titans_project/modules/vit_eagle.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor

class PretrainedViTEagle(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224", freeze=True):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor):
        device = images.device
        np_imgs = images.cpu().numpy()
        np_list = [np_imgs[i] for i in range(np_imgs.shape[0])]
        inputs = self.processor(images=np_list, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(device)

        outputs = self.model(**inputs)
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output
        else:
            return outputs.last_hidden_state.mean(dim=1)
