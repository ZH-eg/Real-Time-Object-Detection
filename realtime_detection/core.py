from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """一帧图像及其时序元数据。"""

    frame_id: int
    image_bgr: np.ndarray
    received_at: float
    source_position_ms: float | None = None


@dataclass(frozen=True)
class LetterboxMeta:
    """Letterbox 坐标还原所需的数据。尺寸统一使用 (width, height)。"""

    original_size: tuple[int, int]
    input_size: tuple[int, int]
    scale: float
    pad_left: int
    pad_top: int


@dataclass(frozen=True)
class DetectionResult:
    frame_id: int
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    finished_at: float

