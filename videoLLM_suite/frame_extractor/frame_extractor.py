import decord
import torch

import numpy as np
import torch.nn.functional as F

from torchvision.transforms import Compose, ToTensor, Normalize
from typing import Tuple
from functools import partial

__all__ = ['VideoPreprocessor']

def to_tensor_video(clip: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    if clip.ndimension() != 4:
        raise ValueError(f"clip should be 4 dimensional, but got {clip.dim()}.")
    if not clip.dtype == torch.uint8:
        raise TypeError(
            "clip tensor should have data type uint8. Got %s" % str(clip.dtype)
        )
    return clip.float().permute(3, 0, 1, 2) / 255.0

def normalize(mean: Tuple[float], std: Tuple[float], clip: torch.Tensor) -> torch.Tensor:
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (C, T, H, W)
    """
    if not torch.is_tensor(clip):
        raise TypeError(f"clip should be Tensor, but got {type(clip)}.")
    if clip.ndimension() != 4:
        raise ValueError(f"clip should be 4 dimensional, but got {clip.dim()}.")
    
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip

class VideoPreprocessor:
    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        mean: Tuple[float] = (0.48145466, 0.4578275, 0.40821073),
        std: Tuple[float] = (0.26862954, 0.26130258, 0.27577711),
        num_frames: int = 8):
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height should be greater than 0, but got width={width} and height={height}.")
        if num_frames <= 0:
            raise ValueError(f"num_frames should be greater than 0, but got {num_frames}.")
        self._width = width
        self._height = height
        self._normalizer = None
        self._num_frames = num_frames

        self._transform = Compose([
            to_tensor_video,
            partial(normalize, mean, std)
        ])
    
    def __call__(
        self,
        video_path: str,
    ):
        clip = self._load_video(video_path)
        return self._transform(clip)
    
    def _load_video(
        self,
        video_path: str,
    ):
        video_reader = decord.VideoReader(video_path, height=self._height, width=self._width)
        video_len = len(video_reader)

        sampling_frame_num = min(self._num_frames, video_len)
        indices = np.arange(0, video_len, video_len / sampling_frame_num).astype(int)

        sampled_frames = video_reader.get_batch(indices).asnumpy()
        return torch.Tensor(sampled_frames).to(torch.uint8)