import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest

from firefly.frame_extractor.frame_extractor import VideoFrameExtractor, AudioFrameExtractor

"""
AudioFrameExtractor
"""
def test_sliding_window():
    sample_rate = 44100
    win_sec = 2
    hop_sec = 2
    duration = 150

    input_path = 'tests/audio/input.wav'

    audio_frame_extractor = AudioFrameExtractor(sample_rate=sample_rate,
                                                win_sec=win_sec, hop_sec=hop_sec)
    audio_frames = audio_frame_extractor.extract_frames(input_path)
    assert audio_frames.frames.shape[0] == duration // hop_sec + 1, f'audio_frames.frames.shape[0] should be {duration // hop_sec + 1}, but {audio_frames.frames.shape[0]}.'
    assert audio_frames.frames.shape[1] == sample_rate * win_sec, f'audio_frames.frames.shape[1] should be {sample_rate * win_sec}, but {audio_frames.frames.shape[1]}.'


"""
VideoFrameExtractor
"""
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
    assert len(video_frames.frames) == video_length