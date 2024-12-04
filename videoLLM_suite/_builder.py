from typing import Callable

def build_model_with_cfg(
    model_cls: Callable,
    variant: str,
    **kwargs,
):
    """
    Build model with specified default_cfg. and optional model_cfg
    """
    model = model_cls(**kwargs)
    return model