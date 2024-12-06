import torch
import torch.nn as nn

from typing import Union, Tuple

from ._builder import build_model_with_cfg
from ._register import register_model
from .vision_transformer import create_eva_vit_g

from preprocessor import VideoPreprocessor
from transformers import BertTokenizer

__all__ = ['video_llama_vicuna_7b']


class VideoLLaMA(nn.Module):
    """
        Video LLaMA
    """
    def __init__(
        self,
        vit_model: str = 'eva_clip_g',
        q_former_model: str = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth',
        img_size: int = 224,
        drop_path_rate: int = 0,
        use_grad_checkpoint: bool = False,
        vit_precision: str = 'fp16',
        freeze_vit: bool = True,
        freeze_qformer: bool = True,
        num_query_token: int = 32,
        llama_model: str = '',
        prompt_path: str = '',
        prompt_template: str = '',
        max_txt_len: int = 32,
        end_sym: str = '\n',
        low_resource: bool = False,
        device_8bit: int = 0,
        frozen_llama_proj: bool = True,
        frozen_video_Qformer: bool = True,
        frozen_audio_Qformer: bool = True,
        llama_proj_model: str = '',
        fusion_header_type: str = 'seqTransf',
        max_frame_pos: int = 32,
        fusion_head_layers: int = 2,
        num_video_query_token: int = 32,
        num_audio_query_token: int = 8,
        imagebind_ckpt_path: str = '',
        equip_audio_branch: bool = True,
    ):
        super().__init__()
        self._tokenizer = self._init_tokenizer()
        self._low_resource = low_resource
        self._img_size = img_size
        
        visual_encoder, ln_vision = self._init_vision_encoder(
            vit_model, img_size, drop_path_rate, 
            use_grad_checkpoint, vit_precision
        )
        self.visual_encoder = visual_encoder
        self.ln_vision = ln_vision
        if freeze_vit:
            self._freeze_vit()        

    def _init_vision_encoder(self):
        vit_encoder = create_eva_vit_g(
            img_size=self._img_size,
            patch_size=self._patch_size,
            use_mean_pooling=False,
            embed_dim=self._embed_dim,
            depth=

        )


        pass

    def _init_tokenizer(self):
        tokenizer = BertTokenizer('bert-base-uncased')
        tokenizer.add_special_tokens({ 'bos_token': '[DEC]' })
        return tokenizer

def _cfg():
    cfg = {
        'img_size': 224,
        'drop_path_rate': 0,
        'use_grad_checkpoint': False,
        'vit_precision': 'fp16',
        'freeze_vit': True,
        'freeze_qformer': True,
        'num_query_token': 32,
        'llama_model_path': None, # TODO **kwargs,
        'prompt': '',
        'model_type': 'pretrain_llama_v2',
        'max_txt_len': 512,
        'end_sym': '###',
        'low_resource': False,
        'frozen_llama_proj': False,
        'imagebind_ckpt_path': None, # TODO
        'ckpt': None, # TODO
        'ckpt_2': None, # TODO
        'equip_audio_branch': True,
        'fusion_head_layers': 2,
        'max_frame_pos': 32,
        'fusion_header_type': 'seqTransf',
        'device_8bit': 0,
    }
    return cfg

def _create_video_llama(variant, **kwargs):
    config = _cfg()
    model = build_model_with_cfg(
        VideoLLaMA, variant, config, **kwargs,
    )
    return model

@register_model
def video_llama_vicuna_7b(**kwargs) -> VideoLLaMA:
    model_args = dict(image_size=256)
    model = _create_video_llama('video_llama_vicuna_7b', **dict(model_args, **kwargs))
    frame_extractor = VideoPreprocessor()
    return model