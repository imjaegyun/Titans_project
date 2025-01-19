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

        # (7) 차선 변경 여부 분류기
        self.lane_change_classifier = nn.Linear(self.base_llm.hidden_size, 1)  # 이진 분류

        # (8) Optimizer 설정
        self.opt_cfg = optimizer_cfg

        # 테스트 결과 저장을 위한 리스트 초기화
        self.test_outputs = []

        # 손실 함수 정의
        self.criterion_traj = nn.MSELoss()
        self.criterion_lane_change = nn.BCEWithLogitsLoss()

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
            attention_mask=attn_mask  # attention_mask 설정
        )
        hidden_states = base_out.hidden_states[-1]  # [B,3,2048]

        # Separate hidden for decoder and classification
        # For decoder: [B, T_out, D] where T_out=2
        # For classification: [B,D] (mean of hidden_states)
        T_out = self.titans.T_out  # 2
        if hidden_states.size(1) >= T_out:
            hidden_for_decoder = hidden_states[:, :T_out, :]  # [B,2,2048]
        else:
            # T_out=2, if less, repeat last step
            repeat_times = T_out - hidden_states.size(1)
            last_step = hidden_states[:, -1:, :].repeat(1, repeat_times, 1)  # [B,1,2048]
            hidden_for_decoder = torch.cat([hidden_states, last_step], dim=1)  # [B,2,2048]
            print(f"TitanModel: Hidden size less than T_out. Repeated last timestep to make T={T_out}")

        # For classification
        hidden_for_classification = hidden_states.mean(dim=1)  # [B,2048]

        # I) Titans Decoder 통해 궤적 예측 => [B,2,2]
        traj_out = self.titans(hidden_for_decoder)  # [B,2,2]

        # J) 차선 변경 여부 예측
        lane_change_logit = self.lane_change_classifier(hidden_for_classification).squeeze(1)  # [B]

        # K) 차선 변경 의도 설명 생성
        lane_change_pred = torch.sigmoid(lane_change_logit) > 0.5  # [B]
        lane_change_pred = lane_change_pred.int().cpu().tolist()  # 리스트 형태로 변환

        explanations = self.generate_explanation(batch["scene_text"], lane_change_pred)

        return {
            "traj_out": traj_out,
            "lane_change_logit": lane_change_logit,
            "lane_change_explainer_text": explanations  # 동적 텍스트 생성
        }

    def generate_explanation(self, scene_texts, lane_change_preds):
        """
        BaseLLM을 사용하여 차선 변경 의도 설명을 생성합니다.
        """
        explanations = []
        for text, pred in zip(scene_texts, lane_change_preds):
            if pred == 1:
                prompt = f"{text}\n\nThe vehicle is changing lanes. Provide an explanation for this action:"
            else:
                prompt = f"{text}\n\nThe vehicle is maintaining its current lane. Provide an explanation for this decision:"

            # BaseLLM의 generate 메서드를 사용하여 텍스트 생성
            generated_ids = self.base_llm.model.generate(
                input_ids=self.base_llm.tokenizer.encode(prompt, return_tensors='pt').to(self.device),
                max_new_tokens=150,  # max_length 대신 max_new_tokens 사용
                num_return_sequences=1,
                do_sample=True,
                top_p=0.95,
                top_k=60,
                pad_token_id=self.base_llm.tokenizer.eos_token_id  # pad_token_id 설정
            )
            explanation = self.base_llm.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            explanations.append(explanation)

        return explanations

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        traj_pred = outputs["traj_out"]  # [B, 2, 2]
        lane_change_logit = outputs["lane_change_logit"]  # [B]
        lane_change_label = batch["lane_change_label"].float().to(self.device)  # [B]

        traj_label = batch["traj_label"].to(self.device)  # [B, 2, 2]

        # MSE Loss 계산
        mse_loss = self.criterion_traj(traj_pred, traj_label)

        # 차선 변경 여부 손실 계산
        lane_change_loss = self.criterion_lane_change(lane_change_logit, lane_change_label)

        # 총 손실
        total_loss = mse_loss + lane_change_loss

        # 손실 함수의 상세 정보 로깅
        self.log("train_loss_mse", mse_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_lane_change", lane_change_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        traj_pred = outputs["traj_out"]  # [B, 2, 2]
        lane_change_logit = outputs["lane_change_logit"]  # [B]
        lane_change_label = batch["lane_change_label"].float().to(self.device)  # [B]

        traj_label = batch["traj_label"].to(self.device)  # [B, 2, 2]

        # MSE Loss 계산
        mse_loss = self.criterion_traj(traj_pred, traj_label)

        # 차선 변경 여부 손실 계산
        lane_change_loss = self.criterion_lane_change(lane_change_logit, lane_change_label)

        # ADE (Average Displacement Error) 계산
        ade = torch.mean(torch.norm(traj_pred - traj_label, dim=2))  # [B, 2] -> 평균

        # FDE (Final Displacement Error) 계산
        fde = torch.mean(torch.norm(traj_pred[:, -1, :] - traj_label[:, -1, :], dim=1))  # [B]

        # 차선 변경 정확도 계산
        lane_change_pred = torch.sigmoid(lane_change_logit) > 0.5
        lane_change_acc = (lane_change_pred.float() == lane_change_label).float().mean()

        # 손실 및 지표 로깅
        self.log("val_loss_mse", mse_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss_lane_change", lane_change_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ADE", ade, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_FDE", fde, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_lane_change_acc", lane_change_acc, on_step=False, on_epoch=True, prog_bar=True)

        return mse_loss

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        traj_pred = outputs["traj_out"]  # [B, 2, 2]
        lane_change_logit = outputs["lane_change_logit"]  # [B]
        lane_change_label = batch["lane_change_label"].float().to(self.device)  # [B]

        traj_label = batch["traj_label"].to(self.device)  # [B, 2, 2]

        # ADE (Average Displacement Error) 계산
        ade = torch.mean(torch.norm(traj_pred - traj_label, dim=2))  # [B, 2] -> 평균

        # FDE (Final Displacement Error) 계산
        fde = torch.mean(torch.norm(traj_pred[:, -1, :] - traj_label[:, -1, :], dim=1))  # [B]

        # 차선 변경 정확도 계산
        lane_change_pred = torch.sigmoid(lane_change_logit) > 0.5
        lane_change_acc = (lane_change_pred.float() == lane_change_label).float().mean()

        # 지표 로깅
        self.log("test_ADE", ade, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_FDE", fde, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_lane_change_acc", lane_change_acc, on_step=False, on_epoch=True, prog_bar=True)

        # 예측된 궤적과 차선 변경 예측 및 설명 반환
        lane_change_pred_label = lane_change_pred.int().cpu().tolist()
        lane_change_explainer = outputs["lane_change_explainer_text"]  # 리스트 형태

        output = {
            "lane_change_pred": lane_change_pred_label,
            "lane_change_explainer": lane_change_explainer,
            "pred_traj": traj_pred.detach().cpu(),
            "true_traj": traj_label.detach().cpu()
        }
        self.test_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        # 모든 배치의 결과를 하나의 리스트로 합칩니다.
        all_pred_traj = []
        all_true_traj = []
        all_lane_change_pred = []
        all_lane_change_explainer = []

        for output in self.test_outputs:
            all_pred_traj.append(output["pred_traj"])
            all_true_traj.append(output["true_traj"])
            all_lane_change_pred.extend(output["lane_change_pred"])
            all_lane_change_explainer.extend(output["lane_change_explainer"])

        # 텐서를 하나로 합칩니다.
        all_pred_traj = torch.cat(all_pred_traj, dim=0).numpy()  # [N, 2, 2]
        all_true_traj = torch.cat(all_true_traj, dim=0).numpy()  # [N, 2, 2]

        # 예측 결과를 CSV 파일로 저장
        data = {
            "lane_change_pred": all_lane_change_pred,
            "lane_change_explainer": all_lane_change_explainer,
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
