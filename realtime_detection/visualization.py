from __future__ import annotations

import cv2
import numpy as np

from .core import DetectionResult
from .metrics import MetricsSnapshot


_COLORS = [
    (80, 80, 255),
    (80, 220, 80),
    (255, 140, 60),
    (220, 100, 220),
    (60, 220, 220),
]


def draw_result(
    image_bgr: np.ndarray,
    result: DetectionResult,
    class_names: list[str],
    metrics: MetricsSnapshot,
    dropped_frames: int,
) -> np.ndarray:
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]

    for box, score, class_id_value in zip(
        result.boxes_xyxy, result.scores, result.class_ids
    ):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = np.clip([x1, x2], 0, max(0, width - 1))
        y1, y2 = np.clip([y1, y2], 0, max(0, height - 1))
        class_id = int(class_id_value)
        name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        color = _COLORS[class_id % len(_COLORS)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {float(score):.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    status = (
        f"frame={result.frame_id} objects={len(result.boxes_xyxy)} "
        f"FPS={metrics.window_fps:.1f} infer={metrics.inference_ms:.1f}ms "
        f"age={metrics.frame_age_ms:.1f}ms dropped={dropped_frames}"
    )
    cv2.rectangle(canvas, (0, 0), (width, 42), (25, 25, 25), -1)
    cv2.putText(
        canvas,
        status,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 235, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas

