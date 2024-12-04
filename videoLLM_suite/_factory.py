from ._register import is_model, model_entrypoint

__all__ = ['create_pipeline']

def create_pipeline(model_name: str, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if not is_model(model_name):
        raise RuntimeError(f'Unknown model ({model_name})')

    create_fn = model_entrypoint(model_name)
    model  = create_fn(**kwargs)
    
    return model