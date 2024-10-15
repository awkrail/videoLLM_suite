from dataclasses import dataclass

@dataclass
class VideoExtractorConfig:
    fps: float = 0.0
    width: int = 0
    height: int = 0

@dataclass
class AudioExtractorConfig:
    win_sec: float = 0.0
    hop_sec: float = 0.0
    sample_rate: float = 0.0