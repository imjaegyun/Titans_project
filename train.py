# Titans_project/train.py

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

from modules.data_module import DroneTrajectoryDataModule
from modules.titan_model import TitanModel

@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print("csv_path:", cfg.data.csv_path)

    dm = DroneTrajectoryDataModule(
        csv_path = cfg.data.csv_path,
        video_path = cfg.data.video_path,
        train_size = cfg.data.train_size,
        val_size   = cfg.data.val_size,
        train_batch_size = cfg.data.train_batch_size,
        val_batch_size   = cfg.data.val_batch_size,
        frames_per_second= cfg.data.frames_per_second,
        past_sec= cfg.data.past_sec
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

    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
