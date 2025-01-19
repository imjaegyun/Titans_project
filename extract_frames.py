# Titans_project/extract_frames.py

import cv2
import os

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
    cap.release()
    print(f"Extracted {frame_idx} frames to {output_dir}")

if __name__ == "__main__":
    video_path = "/home/user/imjaegyun/Titans_model/Titans_project/datasets/DJI_0023_stab.mp4"
    output_dir = "/path/to/extracted_frames"
    extract_frames(video_path, output_dir)
