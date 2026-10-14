"""实时视频流物体检测示例包。"""

from .core import DetectionResult, FramePacket, LetterboxMeta
from .detectors import ColorBlobDetector, TensorRTDetector
from .pipeline import PipelineConfig, run_pipeline
from .preprocess import preprocess_rgb, restore_boxes
from .sources import LatestFrameReader, OpenCVVideoSource

__all__ = [
    "ColorBlobDetector",
    "DetectionResult",
    "FramePacket",
    "LatestFrameReader",
    "LetterboxMeta",
    "OpenCVVideoSource",
    "PipelineConfig",
    "TensorRTDetector",
    "preprocess_rgb",
    "restore_boxes",
    "run_pipeline",
]

