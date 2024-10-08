import argparse

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from firefly.vision.frame_extractor.frame_extractor import VideoFrameExtractor
from firefly.vision.frame_extractor.video_frame import VideoFrame

def main(input_path: str):
    fps: float = 1.0
    video_frame_extractor: VideoFrameExtractor = VideoFrameExtractor(fps)
    video_frames: VideoFrame = video_frame_extractor.extract_frames(input_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, required=True, help='input video path.')
    args = parser.parse_args()
    main(args.input_path)