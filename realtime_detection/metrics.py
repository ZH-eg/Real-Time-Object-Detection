from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .core import DetectionResult, FramePacket


def summarize_ms(values) -> dict[str, float]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return {}
    return {
        "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "max": float(data.max()),
    }


@dataclass(frozen=True)
class MetricsSnapshot:
    window_fps: float
    frame_age_ms: float
    inference_ms: float


class RollingMetrics:
    def __init__(self, window_size: int = 120):
        self.finished_times = deque(maxlen=window_size)
        self.frame_ages_ms = deque(maxlen=window_size)
        self.preprocess_ms = deque(maxlen=window_size)
        self.inference_ms = deque(maxlen=window_size)
        self.postprocess_ms = deque(maxlen=window_size)

    def update(self, packet: FramePacket, result: DetectionResult) -> None:
        self.finished_times.append(result.finished_at)
        self.frame_ages_ms.append((result.finished_at - packet.received_at) * 1000)
        self.preprocess_ms.append(result.preprocess_ms)
        self.inference_ms.append(result.inference_ms)
        self.postprocess_ms.append(result.postprocess_ms)

    def window_fps(self) -> float:
        if len(self.finished_times) < 2:
            return 0.0
        duration = self.finished_times[-1] - self.finished_times[0]
        return (len(self.finished_times) - 1) / duration if duration > 0 else 0.0

    def snapshot(self) -> MetricsSnapshot:
        return MetricsSnapshot(
            window_fps=self.window_fps(),
            frame_age_ms=float(self.frame_ages_ms[-1]) if self.frame_ages_ms else 0.0,
            inference_ms=float(self.inference_ms[-1]) if self.inference_ms else 0.0,
        )

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            "frame_age_ms": summarize_ms(self.frame_ages_ms),
            "preprocess_ms": summarize_ms(self.preprocess_ms),
            "inference_ms": summarize_ms(self.inference_ms),
            "postprocess_ms": summarize_ms(self.postprocess_ms),
        }

