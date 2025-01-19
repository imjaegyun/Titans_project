# Titans_project/modules/titan_model.py

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from .lightweight_llm import LightWeightLLM
from .vit_eagle import PretrainedViTEagle
from .trajectory_encoder import TrajectoryEncoder
from .base_llm import BaseLLM
from .titans_core import TitansDecoder

import pandas as pd

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
        print(f"BaseLLM initialized with hidden_size: {self.base_llm.hidden_size}")  # 확인

        # (2) Light LLM => tokenizer만 로딩
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

        # (4) Trajectory Encoder
        self.traj_enc = TrajectoryEncoder(
            input_dim=trajectory_encoder_cfg["input_dim"],  # 2
            hidden_dim=trajectory_encoder_cfg["hidden_dim"]
        )

        # (5) 임베딩 크기 조정을 위한 선형 레이어 수정 (768 -> hidden_size)
        self.embedding_projection = nn.Linear(768, self.base_llm.hidden_size)  # 768 -> 2048

        # (6) Titans Decoder
        self.titans = TitansDecoder(
            d_model = titans_cfg["d_model"],  # 2048
            memory_depth = titans_cfg["memory_depth"],
            decoder_layers = titans_cfg["decoder_layers"],
            surprise_decay = titans_cfg["surprise_decay"],
            momentum = titans_cfg["momentum"],
            forget_alpha = titans_cfg["forget_alpha"]
        )

        self.opt_cfg = optimizer_cfg

        # 테스트 결과 저장을 위한 리스트 초기화
        self.test_outputs = []

    def forward(self, batch):
        # A) Light LLM => tokenizer
        tokens = self.light_llm.tokenize_text(batch["scene_text"])
        input_ids = tokens["input_ids"].to(self.device)
        attn_mask = tokens["attention_mask"].to(self.device)

        # B) Base LLM에 'inputs_embeds' 전달
        Bsz = input_ids.size(0)
        user_emb = torch.zeros(Bsz, 768, device=self.device)  # 더미 유저 임베딩

        # C) 이미지 처리
        drone_img = batch["drone_image"]  # PIL Images 리스트
        img_emb = self.vit(drone_img)  # [B, hidden_size=768]

        # D) 트래젝토리 처리
        traj = batch["trajectory"].to(self.device)  # [B,1,2]
        traj = traj.squeeze(1)  # [B,2]
        traj_emb = self.traj_enc(traj)  # [B, 768]

        # E) 임베딩 결합 => [B,3,768]
        combined = torch.stack([user_emb, img_emb, traj_emb], dim=1)  # [B,3,768]

        # F) 임베딩 크기 조정
        combined_projected = self.embedding_projection(combined)  # [B,3,2048]

        # G) Base LLM 통해 전달 => [B,3, hidden_size=2048]
        base_out = self.base_llm(
            inputs_embeds=combined_projected,
            attention_mask=None
        )
        hidden = base_out.hidden_states[-1]  # [B,3, hidden_size] = [B,3,2048]

        # H) Titans Decoder 통해 궤적 예측 => [B,2,2]
        traj_out = self.titans(hidden)  # [B,2,2]

        return traj_out

    def training_step(self, batch, batch_idx):
        traj_pred = self.forward(batch)  # [B, future_steps, 2]
        traj_label = batch["traj_label"].to(self.device)  # [B, future_steps, 2]
        
        # MSE Loss 계산
        mse_loss = F.mse_loss(traj_pred, traj_label)
        
        # 손실 함수의 상세 정보 로깅
        self.log("train_loss_mse", mse_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return mse_loss

    def validation_step(self, batch, batch_idx):
        traj_pred = self.forward(batch)  # [B, future_steps, 2]
        traj_label = batch["traj_label"].to(self.device)  # [B, future_steps, 2]
        
        # MSE Loss 계산
        mse_loss = F.mse_loss(traj_pred, traj_label)
        
        # ADE (Average Displacement Error) 계산
        ade = torch.mean(torch.norm(traj_pred - traj_label, dim=2))  # [B, future_steps] -> 평균
        
        # FDE (Final Displacement Error) 계산
        fde = torch.mean(torch.norm(traj_pred[:, -1, :] - traj_label[:, -1, :], dim=1))  # [B]
        
        # 손실 및 지표 로깅
        self.log("val_loss_mse", mse_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ADE", ade, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_FDE", fde, on_step=False, on_epoch=True, prog_bar=True)
        
        # **샘플 텍스트 로그 추가**
        if batch_idx == 0:
            sample_text = batch["scene_text"][0]  # 배치의 첫 번째 텍스트
            self.logger.experiment.add_text("Sample Scene Text", sample_text, self.current_epoch)
        
        return mse_loss

    def test_step(self, batch, batch_idx):
        traj_pred = self.forward(batch)  # [B, future_steps, 2]
        traj_label = batch["traj_label"].to(self.device)  # [B, future_steps, 2]
        
        # ADE (Average Displacement Error) 계산
        ade = torch.mean(torch.norm(traj_pred - traj_label, dim=2))  # [B, future_steps] -> 평균
        
        # FDE (Final Displacement Error) 계산
        fde = torch.mean(torch.norm(traj_pred[:, -1, :] - traj_label[:, -1, :], dim=1))  # [B]
        
        # 지표 로깅
        self.log("test_ADE", ade, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_FDE", fde, on_step=False, on_epoch=True, prog_bar=True)
        
        # **예측된 궤적과 원본 텍스트 반환**
        output = {
            "pred_traj": traj_pred.detach().cpu(),
            "true_traj": traj_label.detach().cpu(),
            "scene_text": batch["scene_text"]
        }
        self.test_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        # 모든 배치의 결과를 하나의 리스트로 합칩니다.
        all_pred_traj = []
        all_true_traj = []
        all_scene_text = []
        
        for output in self.test_outputs:
            all_pred_traj.append(output["pred_traj"])
            all_true_traj.append(output["true_traj"])
            all_scene_text.extend(output["scene_text"])
        
        # 텐서를 하나로 합칩니다.
        all_pred_traj = torch.cat(all_pred_traj, dim=0).numpy()  # [N, future_steps, 2]
        all_true_traj = torch.cat(all_true_traj, dim=0).numpy()  # [N, future_steps, 2]
        
        # scene_text 리스트는 이미 리스트 형태이므로 별도의 변환이 필요 없습니다.
        
        # 예측 결과를 CSV 파일로 저장하거나, 다른 형식으로 저장할 수 있습니다.
        data = {
            "scene_text": all_scene_text,
            "pred_traj": [traj.tolist() for traj in all_pred_traj],
            "true_traj": [traj.tolist() for traj in all_true_traj]
        }

        df = pd.DataFrame(data)
        # 결과를 CSV 파일로 저장
        df.to_csv("test_results.csv", index=False)
        print("Test results saved to test_results.csv")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.opt_cfg["lr"],
            weight_decay=self.opt_cfg["weight_decay"]
        )
