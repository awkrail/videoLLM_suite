import clip
import torch
import math

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
from typing import Optional

from firefly.vision.frame_extractor.video_frame import VideoFrame

def _divide(x):
    return x / 255.0

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(n_px),
        _divide,
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class CLIPVisionEncoder:
    def __init__(
        self,
        device: str,
        model_path: str,
        transforms: Optional[Compose] = None):
        if model_path not in clip.available_models():
            raise ValueError(f'{model_path} are not in {clip.available_models()}.')
        self._device = device
        self._clip_extractor, preprocessor = clip.load(model_path, device=device, jit=False)
        self._transforms = _transform(self._clip_extractor.visual.input_resolution)
        if transforms is not None:
            self._transforms = Compose([transforms, self._transforms])

    @torch.no_grad()
    def extract_feature(
        self,
        video_frames: VideoFrame,
        bsz: int = 60) -> torch.Tensor:
        preprocessed_frames = self._transforms(video_frames.frames)
        n_frames = len(video_frames)
        n_batch = int(math.ceil(n_frames / bsz))
        video_features = []
        for i in range(n_batch):
            st_idx = i * bsz
            ed_idx = (i+1) * bsz
            _preprocessed_frames = preprocessed_frames[st_idx:ed_idx].to(self._device)
            _video_features = self._clip_extractor.encode_image(_preprocessed_frames)
            video_features.append(_video_features)
        video_feature_tensor = torch.cat(video_features, dim=0)
        return video_feature_tensor