import math
import torch

from typing import List

from firefly.base_encoder import BaseAudioTextEncoder
from firefly.frame_extractor.frame import AudioFrame
from firefly.model_config import ModelConfigDict, _available_clap_models

from msclap import CLAP

class CLAPEncoder(BaseAudioTextEncoder):
    def __init__(
        self,
        device: str,
        model_path: str):
        self._model_path: str = model_path
        self._device: str = device
        self._available_models: ModelConfigDict = _available_clap_models()
        if model_path not in self._available_models:
            raise ValueError(f'{model_path} are not in {self._available_models}.')

        self._clap = self._initialize_model()
    
    def _initialize_model(self):
        if self._available_models[self._model_path].owner == 'microsoft':
            clap_model = CLAP(version = self._model_path, use_cuda=True if 'cuda' in self._device else False)
            return clap_model
        
        else:
            raise NotImplementedError(f'{self._model_path} is not implemented now. Choose model from {self._available_models}')

    @torch.no_grad()
    def encode_audio(
        self,
        audio_frames: AudioFrame,
        batch_size: int = 60) -> torch.Tensor:
        n_frames = len(audio_frames)
        n_batch = int(math.ceil(n_frames / batch_size))
        audio_features = []
        for i in range(n_batch):
            st_idx = i * batch_size
            ed_idx = (i+1) * batch_size
            _input_audio = audio_frames.frames[st_idx:ed_idx].to(self._device)
            _audio_features = self._clap.clap.audio_encoder.base(_input_audio)["embedding"]
            _audio_features = self._clap.clap.audio_encoder.projection(_audio_features)            
            audio_features.append(_audio_features)
        audio_feature_tensor = torch.cat(audio_features, dim=0)
        return audio_feature_tensor

    @torch.no_grad()
    def encode_text(
        self,
        sentences: List[str]) -> torch.Tensor:
        text_embeddings = self._clap.get_text_embeddings(sentences)
        return text_embeddings