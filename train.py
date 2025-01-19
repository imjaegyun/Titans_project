# Titans_project/train.py

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import os

from modules.data_module import DroneTrajectoryDataModule
from modules.titan_model import TitanModel
from hydra.utils import instantiate

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Tensor Core 활용을 위한 정밀도 설정
    torch.set_float32_matmul_precision('medium')  # 또는 'high'

    # TOKENIZERS_PARALLELISM 환경 변수 설정
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("csv_path:", cfg.data.csv_path)

    dm = DroneTrajectoryDataModule(
        csv_path=cfg.data.csv_path,
        video_path=cfg.data.video_path,
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,  # 추가된 부분
        train_batch_size=cfg.data.train_batch_size,
        val_batch_size=cfg.data.val_batch_size,
        test_batch_size=cfg.data.test_batch_size,  # 추가된 부분
        frames_per_second=cfg.data.frames_per_second,
        past_sec=cfg.data.past_sec,
        future_steps=cfg.data.future_steps
    )
    dm.setup()

    model = TitanModel(
        lightweight_llm_cfg=cfg.model.lightweight_llm,
        vit_eagle_cfg=cfg.model.vit_eagle,
        trajectory_encoder_cfg=cfg.model.trajectory_encoder,
        base_llm_cfg=cfg.model.base_llm,
        titans_cfg=cfg.model.titans,
        optimizer_cfg=cfg.model.optimizer
    )

    # Callbacks 인스턴스화
    callbacks = []
    if "callbacks" in cfg.trainer:
        for cb_cfg in cfg.trainer.callbacks:
            callback = instantiate(cb_cfg)
            callbacks.append(callback)

    # Logger 인스턴스화
    logger = None
    if "logger" in cfg.trainer:
        logger = instantiate(cfg.trainer.logger)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callbacks,
        # 기타 필요한 Trainer 설정 추가
    )
    trainer.fit(model, dm)
    
    # 테스트 수행
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()
