from __future__ import annotations

import cv2
import numpy as np

from .core import LetterboxMeta


def preprocess_rgb(
    image_rgb: np.ndarray,
    input_size: tuple[int, int] = (640, 640),
    fill_value: int = 114,
) -> tuple[np.ndarray, LetterboxMeta]:
    """将 RGB HWC uint8 图像 Letterbox 为 RGB NCHW float32。

    Args:
        image_rgb: 形状为 ``(H, W, 3)`` 的 RGB 图像。
        input_size: 模型输入的 ``(width, height)``。
        fill_value: Letterbox 填充值。
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"期望 RGB HWC 三通道图像，实际形状为 {image_rgb.shape}")
    if image_rgb.dtype != np.uint8:
        raise ValueError(f"期望 uint8 图像，实际 dtype 为 {image_rgb.dtype}")

    src_h, src_w = image_rgb.shape[:2]
    dst_w, dst_h = map(int, input_size)
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        raise ValueError("图像尺寸和模型输入尺寸必须为正整数")

    scale = min(dst_w / src_w, dst_h / src_h)
    resized_w = max(1, min(dst_w, int(round(src_w * scale))))
    resized_h = max(1, min(dst_h, int(round(src_h * scale))))
    pad_left = (dst_w - resized_w) // 2
    pad_top = (dst_h - resized_h) // 2

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=interpolation)
    canvas = np.full((dst_h, dst_w, 3), fill_value, dtype=np.uint8)
    canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized

    tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(tensor)
    meta = LetterboxMeta(
        original_size=(src_w, src_h),
        input_size=(dst_w, dst_h),
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )
    return tensor, meta


def restore_boxes(boxes_xyxy: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """把模型输入坐标中的 xyxy 框还原到原图坐标。"""
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes = boxes.reshape(-1, 4).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta.pad_left) / meta.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta.pad_top) / meta.scale

    width, height = meta.original_size
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    return boxes

