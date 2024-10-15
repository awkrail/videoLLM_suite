import torch

from firefly.frame_extractor.config import VideoExtractorConfig, AudioExtractorConfig

class VideoFrame:
    def __init__(
        self,
        config: VideoExtractorConfig,
        frames: torch.Tensor,
        ):
        self.config: VideoExtractorConfig = config
        self.frames: torch.Tensor = frames
    
    def __len__(self):
        return len(self.frames)


class AudioFrame:
    def __init__(
        self,
        config: AudioExtractorConfig,
        frames: torch.Tensor):
        self.config: AudioExtractorConfig = config
        self.frames: torch.Tensor = frames
    
    def __len__(self):
        return len(self.frames)