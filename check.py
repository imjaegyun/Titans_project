from transformers import AutoModelForCausalLM
import torch
model_name = "meta-llama/Llama-3.2-1B"

# 우선 모델을 로딩해 보고, config 정보를 출력
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,
    torch_dtype=torch.float32,
    # 아래 옵션들: "기본값"이므로 꼭 필요는 없지만 참고용
    output_attentions=False,
    output_hidden_states=False,
    # ignore_mismatched_sizes=False -> 여기선 mismatch 에러 발생 가능
)

# 1) config 전체를 json 형태로 출력
print(model.config)

# 2) 부분만 골라서 출력 (예: hidden_size, n_heads, n_layers 등)
print("hidden_size =", model.config.hidden_size)
print("num_attention_heads =", model.config.num_attention_heads)
print("num_hidden_layers =", model.config.num_hidden_layers)
