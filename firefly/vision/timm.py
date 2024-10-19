import timm
import torch
import math

from torchvision.transforms import Compose
from typing import Optional

from firefly.base_encoder import BaseVisionEncoder
from firefly.frame_extractor.frame import VideoFrame
from firefly.model_config import _available_timm_models

class TimmEncoder(BaseVisionEncoder):
    def __init__(
        self,
        device: str,
        model_path: str,
        preprocess_transforms: Optional[Compose] = None):
        self._model_path: str = model_path
        self._device: str = device
        self._available_models = _available_timm_models()
        if model_path not in self._available_models:
            raise ValueError(f'{model_path} are not in {self._available_models}.')
        
        self._model = timm.create_model(model_path, pretrained=True, num_classes=0).eval().to(self._device)
        self._transforms = preprocess_transforms
    
    @torch.no_grad()
    def encode_video(
        self,
        video_frames: VideoFrame,
        batch_size: int = 60) -> torch.Tensor:
        preprocessed_frames = self._transforms(video_frames.frames) if self._transforms is not None else video_frames.frames
        n_frames = len(video_frames)
        n_batch = int(math.ceil(n_frames / batch_size))
        video_features = []
        for i in range(n_batch):
            st_idx = i * batch_size
            ed_idx = (i+1) * batch_size
            _frames = preprocessed_frames[st_idx:ed_idx].to(self._device)
            _video_features = self._model.forward_features(_frames)
            video_features.append(_video_features)
        video_feature_tensor = torch.cat(video_features, dim=0)
        return video_feature_tensor