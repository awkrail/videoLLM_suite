import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import random
import torch

from firefly.frame_extractor.frame import VideoFrame
from firefly.vision_text.clip import CLIPEncoder

def test_frame_length_openai_clip():
    length = random.randint(1, 300)
    mock_video_frames = VideoFrame(config=None, frames=torch.ones((length, 3, 256, 256)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision_encoder = CLIPEncoder(model_path='ViT-B/32', device=device)
    out = vision_encoder.encode_video(mock_video_frames)
    assert out.shape[0] == length, f"output vector should be same length, expected={length} but got {out.shape[0]}"