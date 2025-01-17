# Titans_project/modules/data_module.py

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import torchvision.transforms as T
import torch
import random

from .data_prep import build_scene_description

class DroneTrajectoryDataset(Dataset):
    def __init__(self, csv_path, video_path, size=20, frames_per_second=10, past_sec=1):
        super().__init__()
        self.csv_path = csv_path
        self.video_path = video_path
        self.size = size
        self.fps = frames_per_second
        self.past_sec = past_sec

        df = pd.read_csv(csv_path)
        if len(df) > size:
            df = df.head(size).copy().reset_index(drop=True)
        self.df = df

        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224,224)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        scene_text = build_scene_description(row)

        frame_idx = random.randint(0, self.frame_count-1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            frame_bgr = 255 * torch.ones((224,224,3), dtype=torch.uint8).numpy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_tensor = self.transform(frame_rgb)

        cx = row.get('center_x', 0.0)
        cy = row.get('center_y', 0.0)
        traj = torch.tensor([[[cx, cy]]], dtype=torch.float)

        text_label = torch.zeros(768)
        traj_label = torch.zeros(768)

        return {
            "scene_text": scene_text,
            "drone_image": frame_tensor,
            "trajectory": traj,
            "text_label": text_label,
            "traj_label": traj_label
        }

class DroneTrajectoryDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, video_path,
                 train_size, val_size,
                 train_batch_size, val_batch_size,
                 frames_per_second=10, past_sec=1):
        super().__init__()
        self.csv_path = csv_path
        self.video_path = video_path
        self.train_size = train_size
        self.val_size = val_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.fps = frames_per_second
        self.past_sec = past_sec

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = DroneTrajectoryDataset(
                self.csv_path,
                self.video_path,
                size=self.train_size,
                frames_per_second=self.fps,
                past_sec=self.past_sec
            )
            self.val_dataset = DroneTrajectoryDataset(
                self.csv_path,
                self.video_path,
                size=self.val_size,
                frames_per_second=self.fps,
                past_sec=self.past_sec
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size,
                          shuffle=False,
                          num_workers=0)
