# Titans_project/modules/lightweight_llm.py

import pytorch_lightning as pl
from transformers import AutoTokenizer

class LightWeightLLM(pl.LightningModule):
    """
    Light LLM:
    - Just loads the same model_name as Base LLM
    - Only uses AutoTokenizer for scene_text tokenization (embedding not required here).
    """
    def __init__(self, model_name, lr=1e-5, use_lora=False):
        super().__init__()
        self.save_hyperparameters()

        # => tokenizer 만 로딩
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True
        )

        # pad_token이 없으면 eos_token을 패딩 토큰으로 설정하거나 새로운 pad_token 추가
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # 모델에도 pad_token을 추가했는지 확인 필요
                # BaseLLM에서도 추가해야 할 수 있음

        self.lr = lr

    def tokenize_text(self, text):
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

    def forward(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        # no real training
        return []
