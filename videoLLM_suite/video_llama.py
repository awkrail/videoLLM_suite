import torch
import torch.nn as nn

from typing import Union, Tuple

from ._builder import build_model_with_cfg
from ._register import register_model
from preprocessor import VideoPreprocessor

__all__ = ['VideoLLaMA']

class VideoLLaMA(nn.Module):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
    ):
        super().__init__()
        self._image_size = image_size

def _create_video_llama(variant, **kwargs):
    model = build_model_with_cfg(
        VideoLLaMA, variant, **kwargs,
    )
    return model

@register_model
def video_llama_vicuna_7b(**kwargs) -> VideoLLaMA:
    model_args = dict(image_size=256)
    model = _create_video_llama('video_llama_vicuna_7b', **dict(model_args, **kwargs))
    frame_extractor = VideoPreprocessor()
    return model