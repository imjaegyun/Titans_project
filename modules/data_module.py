# Titans_project/modules/data_module.py

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import torchvision.transforms as T
import torch
from PIL import Image
from .data_prep import build_scene_description
from sklearn.model_selection import train_test_split

def custom_collate(batch):
    """
    커스텀 collate_fn:
    - 'drone_image'는 PIL 이미지 리스트로 유지
    - 'scene_text'는 배치 리스트로 유지 (모델 입력 용도)
    - 나머지 필드는 기본 collate_fn을 사용하여 텐서로 변환
    """
    scene_texts = [item['scene_text'] for item in batch]
    drone_images = [item['drone_image'] for item in batch]
    trajectories = torch.stack([item['trajectory'] for item in batch])
    traj_labels = torch.stack([item['traj_label'] for item in batch])
    lane_change_labels = torch.stack([item['lane_change_label'] for item in batch])
    
    return {
        'scene_text': scene_texts,          # 모델 입력 용도
        'drone_image': drone_images,
        'trajectory': trajectories,
        'traj_label': traj_labels,
        'lane_change_label': lane_change_labels
    }

class DroneTrajectoryDataset(Dataset):
    def __init__(self, df, video_path, frames_per_second=10, past_sec=1, future_steps=2):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.video_path = video_path
        self.fps = frames_per_second
        self.past_sec = past_sec
        self.future_steps = future_steps  # 미래 프레임 수

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            # T.ToTensor()  # ToTensor() 제거
        ])

    def __len__(self):
        # 미래 프레임을 고려하여 데이터셋 크기 조정
        return len(self.df) - self.future_steps

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 이전 프레임 정보 (속도 계산을 위해 필요)
        if idx > 0:
            previous_row = self.df.iloc[idx - 1]
        else:
            previous_row = None

        scene_text = build_scene_description(row, previous_row)

        # 현재 프레임 인덱스
        frame_idx = row['frame']  # 실제 프레임 번호 사용

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret or frame_bgr is None:
            frame_bgr = 255 * torch.ones((224, 224, 3), dtype=torch.uint8).numpy()
        else:
            if frame_bgr.ndim == 2:  # grayscale
                frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
            elif frame_bgr.shape[2] != 3:  # 다른 채널 수
                frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = self.transform(frame_rgb)  # PIL Image

        # 현재 트래젝토리 정보 (현재 프레임의 center_x, center_y)
        cx = row.get('center_x', 0.0)
        cy = row.get('center_y', 0.0)
        traj_input = torch.tensor([[cx, cy]], dtype=torch.float)  # [1, 2]

        # 미래 트래젝토리 정보 (다음 2 프레임의 center_x, center_y)
        future_traj = []
        for step in range(1, self.future_steps + 1):
            future_idx = idx + step
            if future_idx < len(self.df):
                future_row = self.df.iloc[future_idx]
                future_cx = float(future_row.get('center_x', 0.0))
                future_cy = float(future_row.get('center_y', 0.0))
                future_traj.append([future_cx, future_cy])
            else:
                # 미래 프레임이 없을 경우, 현재 위치를 유지
                future_traj.append([cx, cy])
        traj_label = torch.tensor(future_traj, dtype=torch.float)  # [future_steps, 2]

        # 차선 변경 여부 레이블 추가
        # 'lane' 필드의 변화를 기반으로 판단
        current_lane = row.get('lane', None)
        if idx + self.future_steps < len(self.df):
            future_row = self.df.iloc[idx + self.future_steps]
            future_lane = future_row.get('lane', None)
            lane_change = 1 if future_lane != current_lane else 0
        else:
            lane_change = 0  # 미래 프레임이 없을 경우 변경 없음

        lane_change_label = torch.tensor(lane_change, dtype=torch.float)  # [1]

        return {
            "scene_text": scene_text,
            "drone_image": frame_pil,
            "trajectory": traj_input,
            "traj_label": traj_label,
            "lane_change_label": lane_change_label  # 추가된 부분
        }

class DroneTrajectoryDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, video_path,
                 train_size=70, val_size=20, test_size=10,
                 train_batch_size=4, val_batch_size=4, test_batch_size=4,
                 frames_per_second=10, past_sec=1, future_steps=2, seed=42):
        super().__init__()
        self.csv_path = csv_path
        self.video_path = video_path
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size  # test_size 저장
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size  # test_batch_size 저장
        self.fps = frames_per_second
        self.past_sec = past_sec
        self.future_steps = future_steps
        self.seed = seed
        self.num_workers = 4  # 초기에는 4으로 설정, 시스템에 맞게 조정

    def setup(self, stage=None):
        # 전체 데이터셋 로드
        df = pd.read_csv(self.csv_path)
        
        # track_id 기준으로 그룹화
        track_ids = df['track_id'].unique()
        
        # train: 70%, val: 20%, test: 10% 비율로 track_ids 분할
        train_ids, temp_ids = train_test_split(
            track_ids,
            test_size=(100 - self.train_size) / 100,
            random_state=self.seed
        )
        val_ratio = self.val_size / (self.val_size + self.test_size)  # 20 / (20 + 10) = 0.666...
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=1 - val_ratio,
            random_state=self.seed
        )
        
        # 각 split에 해당하는 데이터프레임 생성
        train_df = df[df['track_id'].isin(train_ids)].reset_index(drop=True)
        val_df = df[df['track_id'].isin(val_ids)].reset_index(drop=True)
        test_df = df[df['track_id'].isin(test_ids)].reset_index(drop=True)
        
        # 각 split에 대한 Dataset 인스턴스 생성
        self.train_dataset = DroneTrajectoryDataset(
            train_df,
            self.video_path,
            frames_per_second=self.fps,
            past_sec=self.past_sec,
            future_steps=self.future_steps
        )
        self.val_dataset = DroneTrajectoryDataset(
            val_df,
            self.video_path,
            frames_per_second=self.fps,
            past_sec=self.past_sec,
            future_steps=self.future_steps
        )
        self.test_dataset = DroneTrajectoryDataset(
            test_df,
            self.video_path,
            frames_per_second=self.fps,
            past_sec=self.past_sec,
            future_steps=self.future_steps
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate
        )
