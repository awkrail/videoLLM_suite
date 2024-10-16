import clip

from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_path: str = ''
    owner: str = ''
    size: int = 0

class ModelConfigDict:
    def __init__(
        self,
        model_configs: Dict[str, ModelConfig]):
        self.model_configs = model_configs
    
    def __contains__(self, model_path: str):
        return model_path in self.model_configs.keys()
    
    def __repr__(self):
        model_keys = list(self.model_configs.keys())
        return f'{model_keys}'
    
    def __getitem__(self, key):
        return self.model_configs.get(key, ModelConfig())


def _available_clip_models() -> ModelConfigDict:
    model_configs: Dict[str, ModelConfig] = {}
    
    for openai_model in clip.available_models():
        size = 336 if openai_model == 'ViT-L/14@336px' else 224
        model_configs[openai_model] = ModelConfig(model_path=openai_model, owner='openai', size=size)

    other_models: List[Tuple[str, str, int]] = [
        ('line-corporation/clip-japanese-base', 'line-corporation', 224),
    ]

    for m in other_models:
        model_path, owner, size = m
        model_configs[model_path] = ModelConfig(model_path=model_path, owner=owner, size=size)

    return ModelConfigDict(model_configs)


def _available_clap_models() -> ModelConfigDict:
    model_configs: Dict[str, ModelConfig] = {}
    
    other_models: List[Tuple[str, str]] = [
        ('2022', 'microsoft'),
        ('2023', 'microsoft'),
    ]

    for m in other_models:
        model_path, owner = m
        model_configs[model_path] = ModelConfig(model_path=model_path, owner=owner)

    return ModelConfigDict(model_configs)