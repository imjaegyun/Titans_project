# Titans_project/modules/titans_core.py

import torch
import torch.nn as nn

class TitansMemory(nn.Module):
    def __init__(self, d_model=2048, memory_depth=2,
                 surprise_decay=0.9, momentum=0.9, forget_alpha=0.1):
        super().__init__()
        layers = []
        for _ in range(memory_depth):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
        self.memory_mlp = nn.Sequential(*layers)

        self.surprise_decay = surprise_decay
        self.momentum = momentum
        self.forget_alpha = forget_alpha
        self.register_buffer("memory_buffer", torch.zeros(d_model))

    def forward(self, x):
        surprise_val = x.mean(dim=1)
        self.memory_buffer = (1 - self.forget_alpha) * self.memory_buffer \
            + self.momentum * surprise_val.mean().unsqueeze(0)
        return self.memory_mlp(x)

class TitansCore(nn.Module):
    def __init__(self, d_model=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, x):
        return self.net(x)

class TitansDecoder(nn.Module):
    def __init__(self, d_model=2048,
                 memory_depth=2,
                 decoder_layers=2,
                 surprise_decay=0.9,
                 momentum=0.9,
                 forget_alpha=0.1):
        super().__init__()
        self.d_model = d_model
        self.decoder_layers = decoder_layers
        self.T_out = 2  # 출력 시간 단위

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mem": TitansMemory(d_model, memory_depth,
                                    surprise_decay, momentum, forget_alpha),
                "core": TitansCore(d_model)
            })
            for _ in range(decoder_layers)
        ])

        # 최종 출력 레이어: [B,2,2048] -> [B,2,2]
        self.final_linear = nn.Linear(d_model, self.T_out * 2)

    def forward(self, x):
        # 입력 차원 확인 및 변환
        if len(x.shape) == 2:
            # [B, D] -> [B, 1, D]
            x = x.unsqueeze(1)
            print(f"TitansDecoder received input with shape: {x.shape} (reshaped from [B, D] to [B, 1, D])")
        elif len(x.shape) == 3:
            # [B, T, D]
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        B, T, D = x.shape

        # T가 T_out보다 작거나 클 경우 조정
        if T < self.T_out:
            # 마지막 타임스텝 반복하여 T_out 맞추기
            repeat_times = self.T_out - T
            last_step = x[:, -1:, :].repeat(1, repeat_times, 1)
            x = torch.cat([x, last_step], dim=1)
            print(f"TitansDecoder input T was less than T_out. Repeated last timestep to make T={self.T_out}")
        elif T > self.T_out:
            # T_out으로 잘라내기
            x = x[:, :self.T_out, :]
            print(f"TitansDecoder input T was greater than T_out. Truncated to T={self.T_out}")

        B, T, D = x.shape  # 이제 T == T_out

        out = x
        for idx, layerdict in enumerate(self.layers):
            mem = layerdict["mem"]
            cor = layerdict["core"]
            out_tokens = []
            for t in range(T):
                mo = mem(out[:, t, :])  # [B, D]
                co = cor(mo)             # [B, D]
                out_tokens.append(co)
            out = torch.stack(out_tokens, dim=1)  # [B, T, D]
            print(f"TitansDecoder layer {idx+1}/{self.decoder_layers} processed. Current shape: {out.shape}")

        out_avg = out.mean(dim=1)  # [B, D]
        out_final = self.final_linear(out_avg)  # [B, T_out*2]
        out_final = out_final.view(B, self.T_out, 2)  # [B,2,2]
        print(f"TitansDecoder final output shape: {out_final.shape}")
        return out_final
