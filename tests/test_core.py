from __future__ import annotations

import time

import cv2
import numpy as np

from realtime_detection.core import FramePacket
from realtime_detection.detectors import ColorBlobDetector, decode_single_output, nms_xyxy
from realtime_detection.pipeline import detect_frame
from realtime_detection.preprocess import preprocess_rgb, restore_boxes


def test_letterbox_round_trip_boxes() -> None:
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    _, meta = preprocess_rgb(image, (640, 640))
    original = np.asarray([[20, 30, 420, 250]], dtype=np.float32)
    model = original.copy()
    model[:, [0, 2]] = model[:, [0, 2]] * meta.scale + meta.pad_left
    model[:, [1, 3]] = model[:, [1, 3]] * meta.scale + meta.pad_top
    restored = restore_boxes(model, meta)
    np.testing.assert_allclose(restored, original, atol=1e-4)


def test_color_detector_pipeline_restores_original_coordinates() -> None:
    image = np.full((240, 400, 3), 25, dtype=np.uint8)
    cv2.rectangle(image, (50, 60), (150, 160), (0, 0, 255), -1)
    packet = FramePacket(7, image, time.perf_counter())
    detector = ColorBlobDetector((320, 320))
    result = detect_frame(packet, detector)
    assert result.frame_id == 7
    assert result.class_ids.tolist() == [0]
    np.testing.assert_allclose(result.boxes_xyxy[0], [50, 60, 151, 161], atol=3.0)


def test_nms_removes_overlapping_box() -> None:
    boxes = np.asarray([[0, 0, 100, 100], [5, 5, 98, 98], [200, 200, 250, 250]], dtype=np.float32)
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms_xyxy(boxes, scores, 0.5).tolist() == [0, 2]


def test_yolo8_decoder() -> None:
    # 两个类别；前两个候选框重叠，NMS 应保留高分框。
    output = np.asarray(
        [
            [50, 50, 40, 40, 0.90, 0.10],
            [52, 52, 40, 40, 0.80, 0.20],
            [200, 100, 20, 30, 0.05, 0.95],
        ],
        dtype=np.float32,
    )[None]
    boxes, scores, classes = decode_single_output(output, "yolo8", 0.25, 0.5)
    assert boxes.shape == (2, 4)
    assert classes.tolist() == [1, 0]
    np.testing.assert_allclose(scores, [0.95, 0.90], atol=1e-6)

