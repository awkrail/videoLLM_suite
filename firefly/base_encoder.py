import abc
import torch

from typing import List

from firefly.frame_extractor.frame import VideoFrame, AudioFrame

class BaseVisionEncoder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def encode_video(
        self,
        video_frames: VideoFrame,
        batch_size: int = 60) -> torch.Tensor:
        return torch.Tensor([])

class BaseVisionTextEncoder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def encode_video(
        self,
        video_frames: VideoFrame,
        batch_size: int = 60) -> torch.Tensor:
        return torch.Tensor([])

    @abc.abstractmethod
    def encode_text(
        self,
        sentences: List[str]) -> torch.Tensor:
        return torch.Tensor([])

class BaseAudioTextEncoder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def encode_audio(
        self,
        audio_frames: AudioFrame,
        batch_size: int = 60) -> torch.Tensor:
        return torch.Tensor([])

    @abc.abstractmethod
    def encode_text(
        self,
        sentences: List[str]) -> torch.Tensor:
        return torch.Tensor([])