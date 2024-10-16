import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
import pytest
import random
import torch

from firefly.frame_extractor.frame import AudioFrame
from firefly.audio_text.clap import CLAPEncoder

@pytest.fixture(autouse=True)
def suppress_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.meshgrid.*")

def test_non_existing_models():
    model_path = 'Unavailable'
    with pytest.raises(ValueError):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        audio_encoder = CLAPEncoder(model_path=model_path, device=device)

def test_frame_length_microsoft_clap():
    models = ['2022', '2023'] # microsoft CLAP
    for model in models:
        length = random.randint(1, 300)
        sample_rate = 44100
        mock_audio_frames = AudioFrame(config=None, frames=torch.ones((length, sample_rate)))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        audio_encoder = CLAPEncoder(model_path=model, device=device)
        out = audio_encoder.encode_audio(mock_audio_frames)
        assert out.shape[0] == length, f"[{model}] output vector should be same length, expected={length} but got {out.shape[0]}"

def test_text_encoder_microsoft_clap_2023():
    models = ['2022', '2023']
    for model in models:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        audio_encoder = CLAPEncoder(model_path=model, device=device)
        text = ["I will become God of the new world"]
        text_features = audio_encoder.encode_text(text)
        assert text_features.shape == torch.Size([1, 1024])