import warnings

from typing import Callable, Dict, Any

__all__ = ['is_model', 'register_model', 'model_entrypoint']

_model_entrypoints: Dict[str, Callable[..., Any]] = {}  # mapping of model names to architecture entrypoint fns

def model_entrypoint(model_name: str) -> Callable[..., Any]:
    if not is_model(model_name):
        raise ValueError(f"{model_name} does not exist in _model_entrypoints. Double check the model name.")
    return _model_entrypoints[model_name]

def is_model(model_name: str) -> bool:
    return model_name in _model_entrypoints

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
