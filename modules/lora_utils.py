# Titans_project/modules/lora_utils.py
from peft import LoraConfig, get_peft_model

def apply_lora_if_needed(model, target_modules):
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, peft_config)
