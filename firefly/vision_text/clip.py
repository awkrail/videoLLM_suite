import clip
import torch
import math

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
from transformers import AutoModel, AutoTokenizer
from typing import Optional, List

from firefly.frame_extractor.frame import VideoFrame
from firefly.model_config import ModelConfigDict, _available_models

def _divide(x):
    return x / 255.0

def _clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(n_px),
        _divide,
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _clyp_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(n_px),
        _divide,
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class CLIPEncoder:
    def __init__(
        self,
        device: str,
        model_path: str,
        preprocess_transforms: Optional[Compose] = None):
        self._model_path: str = model_path
        self._device: str = device
        self._available_models: ModelConfigDict = _available_models()
        if model_path not in self._available_models:
            raise ValueError(f'{model_path} are not in {self._available_models}.')

        clip_extractor, tokenizer, transforms = self._initialize_model()
        self._clip_extractor = clip_extractor
        self._tokenizer = tokenizer
        self._transforms = transforms(self._available_models[model_path].size)
        
        if preprocess_transforms is not None:
            self._transforms = Compose([preprocess_transforms, self._transforms])
    
    def _initialize_model(self):
        if self._available_models[self._model_path].owner == 'openai': # coz openai models have same APIs    
            clip_extractor, _ = clip.load(self._model_path, device=self._device, jit=False)
            tokenizer = clip.tokenize
            return clip_extractor, tokenizer, _clip_transform

        elif self._available_models[self._model_path].model_path == 'line-corporation/clip-japanese-base':
            tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(self._model_path, trust_remote_code=True).to(self._device)
            return model, tokenizer, _clyp_transform
        
        else:
            raise NotImplementedError(f'{self._model_path} is not implemented now. Choose model from {self._available_models}')

    @torch.no_grad()
    def _encode_frames(
        self,
        frames: torch.Tensor) -> torch.Tensor:
        if self._available_models[self._model_path].owner == 'openai':
            return self._clip_extractor.encode_image(frames)
        elif self._available_models[self._model_path].model_path == 'line-corporation/clip-japanese-base':
            return self._clip_extractor.get_image_features(frames)
        else:
            raise NotImplementedError(f'{self._model_path} is not implemented now. Choose model from {self._available_models}')

    @torch.no_grad()
    def encode_video(
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
            _video_features = self._encode_frames(_preprocessed_frames)
            video_features.append(_video_features)
        video_feature_tensor = torch.cat(video_features, dim=0)
        return video_feature_tensor
    
    @torch.no_grad()
    def encode_text(
        self,
        sentences: List[str]) -> torch.Tensor:
        text = self._tokenizer(sentences).to(self._device)
        if self._available_models[self._model_path].owner == 'openai':
            return self._clip_extractor.encode_text(text)
        elif self._available_models[self._model_path].model_path == 'line-corporation/clip-japanese-base':
            return self._clip_extractor.get_text_features(**text)
        else:
            raise NotImplementedError(f'{self._model_path} is not implemented now. Choose model from {self._available_models}')