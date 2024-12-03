import warnings

from typing import Callable, Dict

__all__ = ['is_model', 'register_model']

_model_entrypoints: Dict[str, Callable[..., Any]] = {}  # mapping of model names to architecture entrypoint fns

def is_model(model_name: str) -> bool:
    return model_name in _model_checkpoints

def register_model(fn: Callable[..., Any]) -> Callable[..., Any]:
    model_name = fn.__name__
    if model_name in _model_entrypoints:
        warnings.warn(
            f'Overwriting {model_name} in registry with {fn.__module__}.{model_name}. This is because the name being '
            'registered conflicts with an existing name. Please check if this is not expected.',
            stacklevel=2,
        )
    _model_entrypoints[model_name] = fn
    return fn

@register_model
def video_llama_visual_audio(**kwargs) -> None: # TODO
    model_args = dict(
        img_size=256
    )
    #model = _create_video_llama
