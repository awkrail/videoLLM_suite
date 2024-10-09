import argparse

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

from firefly.frame_extractor.frame_extractor import VideoFrameExtractor
from firefly.vision_text.clip import CLIPEncoder

def main(input_path: str):
    fps: float = 1.0
    video_frame_extractor = VideoFrameExtractor(fps)
    video_frames = video_frame_extractor.extract_frames(input_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_encoder = CLIPEncoder(model_path='ViT-B/32', device=device)
    frame_features = clip_encoder.encode_video(video_frames)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, required=True, help='input video path.')
    args = parser.parse_args()
    main(args.input_path)