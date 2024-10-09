from dataclasses import dataclass

@dataclass
class VideoExtractorConfig:
    fps: float = 0.0
    width: int = 0
    height: int = 0