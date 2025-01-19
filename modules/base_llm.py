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

        # Tokenizer 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=False,
            trust_remote_code=True
        )

        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # 모델 설정 로드 및 수정
        custom_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        custom_config.output_hidden_states = True  # hidden states 출력 설정

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=custom_config,
            trust_remote_code=True
        )

        # 패딩 토큰 추가 후 모델의 임베딩을 업데이트
        if self.tokenizer.pad_token is not None:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # 모델의 hidden_size 확인
        self.hidden_size = self.model.config.hidden_size  # 일반적으로 768

        print(f"BaseLLM initialized with hidden_size: {self.hidden_size}")  # 디버깅용 출력

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
