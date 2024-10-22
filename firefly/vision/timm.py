import timm
import torch
import math

from torchvision.transforms import Compose
from typing import Optional

from firefly.base_encoder import BaseVisionEncoder
from firefly.frame_extractor.frame import VideoFrame
from firefly.model_config import _available_timm_models

import torchvision.models as models

class TimmEncoder(BaseVisionEncoder):
    def __init__(
        self,
        device: str,
        model_path: str,
        feature_map: bool = False,
        preprocess_transforms: Optional[Compose] = None):
        self._model_path: str = model_path
        self._device: str = device
        self._use_feature_map: bool = feature_map
        self._available_models = _available_timm_models()
        if model_path not in self._available_models:
            raise ValueError(f'{model_path} are not in {self._available_models}.')
    
        self._model = timm.create_model(model_path, pretrained=True).eval().to(self._device)
        if not self._use_feature_map:
            self._model = torch.nn.Sequential(*list(self._model.children())[:-1])
        
        self._transforms = preprocess_transforms
        self._model.eval()


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
            if self._use_feature_map:
                _video_features = self._model.forward_features(_frames)
            else:
                _video_features = self._model(_frames)
            import ipdb; ipdb.set_trace()
            video_features.append(_video_features)
        video_feature_tensor = torch.cat(video_features, dim=0)
        return video_feature_tensor