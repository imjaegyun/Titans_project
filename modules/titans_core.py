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
        B, T, D = x.shape
        out = x
        for layerdict in self.layers:
            mem = layerdict["mem"]
            cor = layerdict["core"]
            out_tokens = []
            for t in range(T):
                mo = mem(out[:, t, :])  # [B,2048]
                co = cor(mo)             # [B,2048]
                out_tokens.append(co)
            out = torch.stack(out_tokens, dim=1)  # [B,T,2048]

        out_avg = out.mean(dim=1)  # [B,2048]
        out_final = self.final_linear(out_avg)  # [B,4]
        out_final = out_final.view(B, self.T_out, 2)  # [B,2,2]
        return out_final
