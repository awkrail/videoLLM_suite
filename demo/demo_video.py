import argparse

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

from videoLLM_suite.frame_extractor import VideoPreprocessor

def main(input_path: str):
    video_frame_extractor = VideoPreprocessor()
    video_frames = video_frame_extractor(input_path)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, required=True, help='input video path.')
    args = parser.parse_args()
    main(args.input_path)