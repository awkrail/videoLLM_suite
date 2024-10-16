import argparse

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

from firefly.frame_extractor.frame_extractor import AudioFrameExtractor
from firefly.audio_text.clap import CLAPEncoder

def main(input_path: str):
    audio_frame_extractor = AudioFrameExtractor(sample_rate=44100, win_sec=2, hop_sec=2)
    audio_frames = audio_frame_extractor.extract_frames(input_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    audio_encoder = CLAPEncoder(model_path='2023', device=device)
    audio_vectors = audio_encoder.encode_audio(audio_frames)
    text_vectors = audio_encoder.encode_text(['dog', 'a man of speaking'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, required=True, help='input video path.')
    args = parser.parse_args()
    main(args.input_path)