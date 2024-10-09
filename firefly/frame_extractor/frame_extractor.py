import os
import ffmpeg
import torch
import numpy as np

from typing import Optional

from firefly.frame_extractor.video_frame import VideoFrame
from firefly.frame_extractor.config import VideoExtractorConfig


class VideoFrameExtractor:
    def __init__(
        self, 
        fps: Optional[float] = None,
        use_original_fps: bool = False):
        self.use_original_fps = use_original_fps
        if self.use_original_fps:
            if fps is not None:
                raise TypeError('When you set use_original_fps=True, fps should be None.')
        else:
            if fps is None:
                raise TypeError('When you set use_original_fps=False, fps should be float.')
            if fps <= 0:
                raise ValueError(f'fps should be more than 0, got {fps}.')
        self.fps = fps

    def _probe_video(
        self,
        input_path: str) -> VideoExtractorConfig:
        if not os.path.exists(input_path):
            raise OSError(f'{input_path} does not exist.')

        probe = ffmpeg.probe(input_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            raise RuntimeError('not found video stream from the input video.')
        
        fps = eval(video_stream['avg_frame_rate'])
        width = video_stream['width']
        height = video_stream['height']

        if self.use_original_fps:
            self.fps = fps

        return VideoExtractorConfig(fps=fps, width=width, height=height)

    def _run_ffmpeg(
        self,
        config: VideoExtractorConfig,
        input_path: str) -> torch.Tensor:

        if self.fps is None:
            raise TypeError(f'fps should be float value, but got {self.fps}. Did you use the original fps? Run _probe_video() first.')

        cmd = (
            ffmpeg
            .input(input_path)
            .filter('fps', fps=self.fps)
        )

        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )

        video = np.frombuffer(out, np.uint8).reshape([-1, config.height, config.width, 3])
        video_tensor = torch.from_numpy(video.astype('float32'))
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        return video_tensor

    def extract_frames(
        self,
        input_path: str) -> VideoFrame:
        if not os.path.exists(input_path):
            raise OSError(f'{input_path} does not exist.')

        config = self._probe_video(input_path)
        if self.fps is not None and self.fps > config.fps:
            raise ValueError(f'fps should be smaller than the input video. set fps={self.fps} video fps={config.fps}')

        frames = self._run_ffmpeg(config, input_path)
        return VideoFrame(config=config, frames=frames)