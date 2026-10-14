from __future__ import annotations

import cv2

from generate_demo_video import generate_demo_video
from realtime_detection.detectors import ColorBlobDetector
from realtime_detection.pipeline import PipelineConfig, run_pipeline
from realtime_detection.sources import OpenCVVideoSource


def test_video_pipeline_writes_reopenable_output(tmp_path) -> None:
    source_path = generate_demo_video(
        tmp_path / "input.avi", width=320, height=200, fps=10.0, seconds=0.6
    )
    output_path = tmp_path / "output.avi"
    source = OpenCVVideoSource(str(source_path), live=False)
    detector = ColorBlobDetector(input_size=(320, 320))
    summary = run_pipeline(
        source,
        detector,
        ColorBlobDetector.class_names,
        PipelineConfig(output_path=str(output_path), log_every=0),
    )

    assert summary["processed_frames"] == 6
    assert summary["overload_dropped_frames"] == 0
    capture = cv2.VideoCapture(str(output_path))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape == (200, 320, 3)
