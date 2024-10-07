import argparse

from firefly.vision.frame_extractor import VideoFrameExtractor
from firefly.vision.video_frame import VideoFrame

def main(input_path: str):
    fps: float = 1.0
    width: int = 224
    height: int = 224

    video_frame_extractor: VideoFrameExtractor = VideoFrameExtractor(fps, width, height)
    video_frames: VideoFrame = video_frame_extractor.extract_frames(input_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, required=True, help='input video path.')
    args = parser.parse_args()
    main(args.input_path)