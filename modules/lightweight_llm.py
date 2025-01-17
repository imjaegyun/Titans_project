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
            use_fast=True,
            trust_remote_code=True
        )
        # (모델 가중치 없음)

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
