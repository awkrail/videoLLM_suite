from typing import Callable, Dict, Any

def build_model_with_cfg(
    model_cls: Callable,
    variant: str,
    config: Dict[str, Any],
    **kwargs,
):
    """
    Build model with specified default_cfg. and optional model_cfg
    """
    model = model_cls(**dict(config, **kwargs))
    return model