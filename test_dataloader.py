# Titans_project/test_dataloader.py

from modules.data_module import DroneTrajectoryDataModule
from omegaconf import OmegaConf

def test_dataloader():
    config = OmegaConf.load("configs/config.yaml")
    dm = DroneTrajectoryDataModule(
        csv_path=config.data.csv_path,
        video_path=config.data.video_path,
        train_size=config.data.train_size,
        val_size=config.data.val_size,
        train_batch_size=config.data.train_batch_size,
        val_batch_size=config.data.val_batch_size,
        frames_per_second=config.data.frames_per_second,
        past_sec=config.data.past_sec
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    print("Testing train DataLoader:")
    for batch in train_loader:
        print(batch)
        break  # 첫 번째 배치만 테스트

    print("Testing validation DataLoader:")
    for batch in val_loader:
        print(batch)
        break  # 첫 번째 배치만 테스트

if __name__ == "__main__":
    test_dataloader()
