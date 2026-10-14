from __future__ import annotations

import math
import threading
import time

import cv2

from .core import FramePacket


def parse_video_source(value: str) -> str | int:
    """纯数字参数按摄像头索引处理，其他内容按路径/URL 处理。"""
    stripped = value.strip()
    if stripped.isdecimal():
        return int(stripped)
    return stripped


class OpenCVVideoSource:
    def __init__(
        self,
        source: str | int,
        live: bool,
        simulate_live: bool = False,
        requested_width: int | None = None,
        requested_height: int | None = None,
        requested_fps: float | None = None,
    ):
        self.source = source
        self.live = live
        self.simulate_live = simulate_live
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源：{source!r}")
        if requested_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
        if requested_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
        if requested_fps:
            self.cap.set(cv2.CAP_PROP_FPS, requested_fps)

        self.frame_id = 0
        self.reported_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.reported_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.reported_fps = fps if math.isfinite(fps) and fps > 0 else 0.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            self.backend_name = self.cap.getBackendName()
        except cv2.error:
            self.backend_name = "unknown"
        self._next_deadline: float | None = None
        self._release_lock = threading.Lock()

    def describe(self) -> str:
        return (
            f"source={self.source!r}, backend={self.backend_name}, "
            f"reported={self.reported_width}x{self.reported_height}@{self.reported_fps:.3f}, "
            f"frame_count={self.frame_count}, live={self.live}, simulate_live={self.simulate_live}"
        )

    def _pace_file(self) -> None:
        if not self.simulate_live or self.reported_fps <= 0:
            return
        interval = 1.0 / self.reported_fps
        now = time.perf_counter()
        if self._next_deadline is None:
            self._next_deadline = now
        delay = self._next_deadline - now
        if delay > 0:
            time.sleep(delay)
        self._next_deadline = max(self._next_deadline + interval, time.perf_counter())

    def read(self) -> FramePacket | None:
        if self.cap is None:
            return None
        self._pace_file()
        ok, frame = self.cap.read()
        received_at = time.perf_counter()
        if not ok or frame is None or frame.size == 0:
            return None
        source_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        if not math.isfinite(source_ms) or source_ms < 0:
            source_ms = None
        packet = FramePacket(
            frame_id=self.frame_id,
            image_bgr=frame,
            received_at=received_at,
            source_position_ms=source_ms,
        )
        self.frame_id += 1
        return packet

    def release(self) -> None:
        with self._release_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None


class LatestFrameReader:
    """容量为 1 的最新帧槽位；新帧覆盖尚未处理的旧帧。"""

    def __init__(self, source: OpenCVVideoSource):
        self.source = source
        self.condition = threading.Condition()
        self.latest: FramePacket | None = None
        self.stopped = False
        self.error: BaseException | None = None
        self.stop_requested = threading.Event()
        self.thread = threading.Thread(target=self._run, name="latest-frame-reader", daemon=True)

    def start(self) -> "LatestFrameReader":
        self.thread.start()
        return self

    def _run(self) -> None:
        try:
            while not self.stop_requested.is_set():
                packet = self.source.read()
                if packet is None:
                    break
                with self.condition:
                    self.latest = packet
                    self.condition.notify_all()
        except BaseException as exc:
            self.error = exc
        finally:
            with self.condition:
                self.stopped = True
                self.condition.notify_all()

    def get_newer_than(self, last_frame_id: int = -1, timeout: float = 1.0) -> FramePacket | None:
        with self.condition:
            ready = self.condition.wait_for(
                lambda: (
                    self.latest is not None and self.latest.frame_id > last_frame_id
                )
                or self.stopped,
                timeout=timeout,
            )
            if not ready:
                return None
            if self.error is not None:
                raise self.error
            if self.latest is None or self.latest.frame_id <= last_frame_id:
                return None
            return self.latest

    def close(self) -> None:
        self.stop_requested.set()
        self.source.release()
        self.thread.join(timeout=2.0)

