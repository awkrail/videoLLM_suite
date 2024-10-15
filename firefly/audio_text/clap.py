class CLAPEncoder:
    
    _available_models = ['2022', '2023']
    
    def __init__(
        self,
        device: str,
        model_path: str):
        self._model_path: str = model_path
        self._device: str = device
        if model_path not in self._available_models:
            raise ValueError(f'{model_path} are not in {self._available_models}.')