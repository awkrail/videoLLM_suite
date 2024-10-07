
from typing import Optional
from firefly.vision.frame_extractor.video_frame import VideoFrame

class VideoFrameExtractor:
    def __init__(
        self, 
        fps: float,
        width: int,
        height: int):
        self.fps: float = fps
        self.width: int = width
        self.height: int = height

    def extract_frames(
        self,
        input_path: str) -> Optional[VideoFrame]:
        return None