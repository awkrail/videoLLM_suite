import torch

from firefly.vision.frame_extractor.config import VideoExtractorConfig

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