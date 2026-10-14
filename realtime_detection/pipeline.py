from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from .core import DetectionResult, FramePacket
from .detectors import Detector
from .metrics import RollingMetrics
from .preprocess import preprocess_rgb, restore_boxes
from .sources import LatestFrameReader, OpenCVVideoSource
from .visualization import draw_result


@dataclass(frozen=True)
class PipelineConfig:
    mode: str = "sync"
    display: bool = False
    output_path: str | None = None
    output_fps: float | None = None
    max_frames: int = 0
    process_every: int = 1
    log_every: int = 30
    metrics_window: int = 120
    summary_json: str | None = None


def detect_frame(packet: FramePacket, detector: Detector) -> DetectionResult:
    t0 = time.perf_counter()
    rgb = cv2.cvtColor(packet.image_bgr, cv2.COLOR_BGR2RGB)
    input_tensor, letterbox_meta = preprocess_rgb(rgb, detector.input_size)
    t1 = time.perf_counter()
    boxes_model, scores, class_ids = detector.infer(input_tensor)
    t2 = time.perf_counter()
    boxes_original = restore_boxes(boxes_model, letterbox_meta)
    t3 = time.perf_counter()
    return DetectionResult(
        frame_id=packet.frame_id,
        boxes_xyxy=boxes_original,
        scores=scores,
        class_ids=class_ids,
        preprocess_ms=(t1 - t0) * 1000,
        inference_ms=(t2 - t1) * 1000,
        postprocess_ms=(t3 - t2) * 1000,
        finished_at=t3,
    )


def _create_writer(path: str, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    codec = "MJPG" if output.suffix.lower() == ".avi" else "mp4v"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*codec), fps, frame_size
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"无法创建输出视频 {output}；可尝试将扩展名改为 .avi 以使用 MJPG"
        )
    return writer


def _next_packet_sync(
    source: OpenCVVideoSource, process_every: int
) -> tuple[FramePacket | None, int]:
    skipped = 0
    while True:
        packet = source.read()
        if packet is None:
            return None, skipped
        if packet.frame_id % process_every == 0:
            return packet, skipped
        skipped += 1


def run_pipeline(
    source: OpenCVVideoSource,
    detector: Detector,
    class_names: list[str],
    config: PipelineConfig,
) -> dict:
    if config.mode not in {"sync", "latest"}:
        raise ValueError("mode 必须是 sync 或 latest")
    if config.process_every < 1:
        raise ValueError("process_every 必须大于等于 1")

    metrics = RollingMetrics(config.metrics_window)
    reader = LatestFrameReader(source).start() if config.mode == "latest" else None
    writer: cv2.VideoWriter | None = None
    processed = 0
    skipped = 0
    dropped = 0
    last_frame_id = -1
    started_at = time.perf_counter()

    try:
        while True:
            if reader is None:
                packet, newly_skipped = _next_packet_sync(source, config.process_every)
                skipped += newly_skipped
            else:
                packet = reader.get_newer_than(last_frame_id, timeout=1.0)
                if packet is None:
                    if reader.stopped:
                        break
                    continue
                gap = max(0, packet.frame_id - last_frame_id - 1)
                dropped += gap
                if packet.frame_id % config.process_every != 0:
                    skipped += 1
                    last_frame_id = packet.frame_id
                    continue

            if packet is None:
                break
            last_frame_id = packet.frame_id
            result = detect_frame(packet, detector)
            metrics.update(packet, result)
            snapshot = metrics.snapshot()
            output_frame = draw_result(packet.image_bgr, result, class_names, snapshot, dropped)

            if writer is None and config.output_path:
                height, width = output_frame.shape[:2]
                inferred_fps = source.reported_fps / config.process_every
                fps = config.output_fps or (inferred_fps if inferred_fps > 0 else 25.0)
                writer = _create_writer(config.output_path, fps, (width, height))
            if writer is not None:
                writer.write(output_frame)

            if config.display:
                cv2.imshow("Chapter 18-4 Video Detection", output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break

            processed += 1
            if config.log_every > 0 and processed % config.log_every == 0:
                print(
                    f"processed={processed} source_frame={packet.frame_id} "
                    f"window_fps={snapshot.window_fps:.2f} "
                    f"infer_ms={snapshot.inference_ms:.2f} age_ms={snapshot.frame_age_ms:.2f} "
                    f"skipped={skipped} dropped={dropped}"
                )
            if config.max_frames > 0 and processed >= config.max_frames:
                break
    finally:
        if reader is not None:
            reader.close()
        else:
            source.release()
        if writer is not None:
            writer.release()
        if config.display:
            cv2.destroyAllWindows()
        detector.close()

    elapsed = time.perf_counter() - started_at
    summary = {
        "mode": config.mode,
        "processed_frames": processed,
        "last_source_frame_id": last_frame_id,
        "strategy_skipped_frames": skipped,
        "overload_dropped_frames": dropped,
        "elapsed_seconds": elapsed,
        "average_fps": processed / elapsed if elapsed > 0 else 0.0,
        "metrics": metrics.summary(),
        "source": source.describe(),
        "config": asdict(config),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if config.summary_json:
        path = Path(config.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

