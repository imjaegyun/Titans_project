# Titans_project/modules/base_llm.py

import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from .lora_utils import apply_lora_if_needed

class BaseLLM(nn.Module):
    """
    Pre-trained LLM with real weights, LoRA possible
    """
    def __init__(self, model_name, tokenizer_name=None, freeze=True, use_lora=True):
        super().__init__()
        if tokenizer_name is None:
            tokenizer_name = model_name

        # => tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=False,
            trust_remote_code=True
        )

        # => 모델 설정 로드 및 수정
        custom_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        custom_config.output_hidden_states = True  # 필요한 설정만 수정

        # => 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=custom_config,
            trust_remote_code=True
        )

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        if use_lora and not freeze:
            self.model = apply_lora_if_needed(self.model, ["q_proj","k_proj"])

    def forward(self, inputs_embeds, attention_mask=None):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs
