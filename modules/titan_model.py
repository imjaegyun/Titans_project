# Titans_project/modules/titan_model.py

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .lightweight_llm import LightWeightLLM
from .vit_eagle import PretrainedViTEagle
from .trajectory_encoder import TrajectoryEncoder
from .base_llm import BaseLLM
from .titans_core import TitansDecoder

class TitanModel(pl.LightningModule):
    def __init__(
        self,
        lightweight_llm_cfg,
        vit_eagle_cfg,
        trajectory_encoder_cfg,
        base_llm_cfg,
        titans_cfg,
        optimizer_cfg
    ):
        super().__init__()
        self.save_hyperparameters()

        # (1) Base LLM => 실제 모델
        self.base_llm = BaseLLM(
            model_name=base_llm_cfg["model_name"],
            tokenizer_name=base_llm_cfg.get("tokenizer_name"),
            freeze=base_llm_cfg["freeze"],
            use_lora=base_llm_cfg.get("use_lora", True)
        )

        # (2) Light LLM => 자기 tokenizer도 똑같이 from_pretrained(...)
        #                 (모델 가중치 없이 tokenizer만? or 작은 모델도 가능)
        self.light_llm = LightWeightLLM(
            model_name=lightweight_llm_cfg["model_name"],
            lr=lightweight_llm_cfg["lr"],
            use_lora=lightweight_llm_cfg.get("use_lora", False)
        )

        # (3) ViT
        self.vit = PretrainedViTEagle(
            model_name=vit_eagle_cfg["model_name"],
            freeze=vit_eagle_cfg["freeze"]
        )

        # (4) Trajectory
        self.traj_enc = TrajectoryEncoder(
            input_dim=trajectory_encoder_cfg["input_dim"],
            hidden_dim=trajectory_encoder_cfg["hidden_dim"]
        )

        # (5) Titans
        self.titans = TitansDecoder(
            d_model = titans_cfg["d_model"],
            memory_depth = titans_cfg["memory_depth"],
            decoder_layers = titans_cfg["decoder_layers"],
            surprise_decay = titans_cfg["surprise_decay"],
            momentum = titans_cfg["momentum"],
            forget_alpha = titans_cfg["forget_alpha"]
        )

        self.opt_cfg = optimizer_cfg

    def forward(self, batch):
        # A) Light LLM => tokenizer
        #    scene_text -> tokenize
        tokens = self.light_llm.tokenize_text(batch["scene_text"])
        input_ids = tokens["input_ids"].to(self.device)
        attn_mask = tokens["attention_mask"].to(self.device)

        # B) -> Base LLM: 지금 base_llm는 'inputs_embeds'를 받도록 구성되어 있음
        #    but we only have input_ids => need to embed ourselves or change base_llm's forward
        # => Let's do a quick hack: create an embedding ourselves is tricky
        #    We'll just do a dummy user_emb for demonstration
        Bsz = input_ids.size(0)
        user_emb = torch.zeros(Bsz, 768, device=self.device)

        # C) image -> vit
        drone_img = batch["drone_image"].to(self.device)
        if drone_img.dim() == 3:
            drone_img = drone_img.unsqueeze(0)
        img_emb = self.vit(drone_img)  # (B,768)

        # D) trajectory -> traj_enc
        traj = batch["trajectory"].to(self.device)
        B,T,D = traj.shape
        if B==1 and T==1:
            traj = traj.view(B,D)
        if traj.shape[1] < 6:
            pad = torch.zeros((B,6 - traj.shape[1]), device=traj.device)
            traj = torch.cat([traj, pad], dim=1)
        traj_emb = self.traj_enc(traj.unsqueeze(1))  # (B,768)

        # E) concat => (B,3,768)
        combined = torch.stack([user_emb, img_emb, traj_emb], dim=1)  # (B,3,768)

        # F) base_llm => forward => (B,3, hidden_dim?)
        base_out = self.base_llm(
            inputs_embeds=combined,
            attention_mask=None
        )
        hidden = base_out.hidden_states[-1]  # (B,3, hidden_dim)

        # G) Titans => (B,2,768)
        dec_out = self.titans(hidden)
        return dec_out

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        traj_pred = out[:,1,:]  # (B,768)
        traj_label = batch["traj_label"].to(self.device)
        loss = F.mse_loss(traj_pred, traj_label)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        traj_pred = out[:,1,:]
        traj_label = batch["traj_label"].to(self.device)
        val_loss = F.mse_loss(traj_pred, traj_label)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.opt_cfg["lr"],
            weight_decay=self.opt_cfg["weight_decay"]
        )
