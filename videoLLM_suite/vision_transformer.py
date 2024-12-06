import torch
import torch.nn as nn
import timm.models.hub as timm_hub

from typing import Dict
from functools import partial

__all__ = ['create_eva_vit_g']

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super.__init__()
        self._num_heads = num_heads
        head_dim = int(dim / num_heads)
        all_head_dim = head_dim * self._num_heads
        self._scale = head_dim ** -0.5
        self._qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self._q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self._v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self._q_bias = None
            self._v_bias = None        
        self._attn_drop = nn.Dropout(attn_drop)
        self._proj = nn.Linear(all_head_dim, dim)
        self._proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super.__init__()
        self._norm1 = nn.LayerNorm(dim)
        self._norm2 = nn.LayerNorm(dim)
        self._attn = Attention(
            dim = dim,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            qkv_bias = qkv_bias,
            attn_drop = attn_drop,
            proj_drop = proj_drop,
        )
        self._drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features = dim,
            hidden_features = mlp_hidden_dim,
            drop = drop,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x = x + self._drop_path(self.attn(self.norm1(x)))
        x = x + self._drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        num_patches = int(img_size / patch_size)**2 
        patch_shape = (int(img_size / patch_size), int(img_size / patch_size))
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self._img_size = img_size
        self._patch_size = patch_size
        self._num_patches = num_patches
        self._patch_shape = patch_shape

        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim,
            kernel_size = patch_size,
            stride = patch_size)
    
    @property
    def num_pacthes(self):
        return self._num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    """
        Vision Transformer with supports for patch / hybrid CNN input stage.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.LayerNorm = nn.LayerNorm,
        init_values = None,
        use_abs_pos_emb: bool = True,
        use_rel_pos_bias: bool = False,
        use_shared_rel_pos_bias: bool = False,
        use_mean_pooling: bool = True,
        init_scale: float = 0.001,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self._image_size = img_size
        self._num_classes = num_classes
        self._num_features = self._embed_dim = embed_dim
        self._patch_embed = PatchEmbed(
            img_size = img_size,
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim
        )

        self._cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            num_patches = self.patch_embed.num_pacthes
            self._pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self._pos_embed = None
        self._pos_drop = nn.Dropout(drop_rate)
        self._use_checkpoint = use_checkpoint
        self._blocks = nn.ModuleList([
            Block(
                dim = embed_dim,
                num_heads = num_heads,
                mlp_ratio = mlp_ratio,
                qkv_bias = qkv_bias,
                qk_scale = qk_scale,
                drop = drop_rate,
                attn_drop = attn_drop_rate,
                drop_path_rate = drop_path_rate,
                norm_layer = norm_layer,
                init_values = init_values,
                window_size = None,
            )
            for i in range(depth)
        ])





def _cfg() -> Dict[str, Any]:
    cfg = {
        'img_size' : 224,
        'patch_size': 14,
        'use_mean_pooling': False,
        'embed_dim': 1408,
        'depth': 39,
        'num_heads': 16,
        'mlp_ratio': 4.3637,
        'qkv_bias': True,
        'drop_path_rate': 0.4,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'use_checkpoint': False,
    }
    return cfg

def download_cached_file(
    url: str,
    check_hash: bool = True,
    progress: bool = False,
    ):
    timm_hub.download_cached_file(url, check_hash, progress)
    parts = torch.hub.urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(timm_hub.get_cache_dir(), filename)
    return cached_file

def create_eva_vit_g(
    vit_url: str = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth",
    precision: str = 'fp16',
    ):
    vit_cfg = _cfg()
    model = VisionTransformer(**vit_cfg)
    cache_file = download_cached_file(vit_url, check_hash=False, progress=True)
    state_dict = torch.load(cache_file, map_location='cpu')
    interpolate_pos_embed(model, state_dict)

    model.load_state_dict(state_dict, strict=False)
    if precision == 'fp16':
        convert_weights_to_fp16(model)
    
    return model