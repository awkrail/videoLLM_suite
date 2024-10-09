import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest

from firefly.frame_extractor.frame_extractor import VideoFrameExtractor

def test_fps_larger_than_original_one():
    fps: float = 1e5
    input_path = 'tests/videos/input.mp4'
    with pytest.raises(ValueError):
        video_frame_extractor = VideoFrameExtractor(fps)
        video_frame_extractor.extract_frames(input_path)

def test_fps_is_not_set_but_use_original_fps_false():
    with pytest.raises(TypeError):
        video_frame_extractor = VideoFrameExtractor()

def test_fps_is_set_but_use_original_fps():
    fps: float = -1.0
    with pytest.raises(ValueError):
        video_frame_extractor = VideoFrameExtractor(fps)

def test_fps_is_set_but_use_original_fps():
    fps: float = 1.0
    with pytest.raises(TypeError):
        video_frame_extractor = VideoFrameExtractor(fps, use_original_fps=True)

def test_input_path_does_not_exist():
    fps: float = 1.0
    input_path = 'tests/videos/does_not_exist.mp4'
    with pytest.raises(OSError):
        video_frame_extractor = VideoFrameExtractor(fps)
        video_frame_extractor.extract_frames(input_path)

def test_video_length():
    fps: float = 1.0
    video_length = 150
    input_path = 'tests/videos/input.mp4'
    video_frame_extractor = VideoFrameExtractor(fps)
    video_frames = video_frame_extractor.extract_frames(input_path)
    len(video_frames.frames) == video_length