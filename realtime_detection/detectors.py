from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


DetectionTuple = tuple[np.ndarray, np.ndarray, np.ndarray]


class Detector(Protocol):
    input_size: tuple[int, int]

    def infer(self, input_tensor: np.ndarray) -> DetectionTuple: ...

    def close(self) -> None: ...


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """返回保留框的索引；实现与后端无关，便于单元测试。"""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        inter_x1 = np.maximum(x1[current], x1[rest])
        inter_y1 = np.maximum(y1[current], y1[rest])
        inter_x2 = np.minimum(x2[current], x2[rest])
        inter_y2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
        union = areas[current] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        local_keep = nms_xyxy(boxes[indices], scores[indices], iou_threshold)
        kept.extend(indices[local_keep].tolist())
    kept.sort(key=lambda index: float(scores[index]), reverse=True)
    return np.asarray(kept, dtype=np.int64)


def _empty_detections() -> DetectionTuple:
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
    )


def _xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes_xywh, dtype=np.float32).copy()
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    return boxes


def _filter_and_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    score_threshold: float,
    iou_threshold: float,
) -> DetectionTuple:
    valid = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & (scores >= score_threshold)
    boxes = np.asarray(boxes[valid], dtype=np.float32)
    scores = np.asarray(scores[valid], dtype=np.float32)
    class_ids = np.asarray(class_ids[valid], dtype=np.int64)
    if boxes.shape[0] == 0:
        return _empty_detections()
    keep = class_aware_nms(boxes, scores, class_ids, iou_threshold)
    return boxes[keep], scores[keep], class_ids[keep]


def decode_single_output(
    output: np.ndarray,
    decoder: str,
    score_threshold: float,
    iou_threshold: float,
) -> DetectionTuple:
    """解析常见单输出检测头。

    支持 ``end2end`` 的 ``[..., 6]``、YOLOv8 的 ``4+C`` 和 YOLOv5 的
    ``5+C``。YOLO 原始输出中的框被解释为模型输入坐标的 ``cxcywh``。
    """
    array = np.asarray(output)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"检测输出应能压缩为二维数组，实际形状为 {output.shape}")

    if decoder == "auto":
        if array.shape[-1] == 6:
            decoder = "end2end"
        elif array.shape[0] in {84, 85} and array.shape[1] > array.shape[0]:
            decoder = "yolo8" if array.shape[0] == 84 else "yolo5"
        elif array.shape[1] >= 6:
            decoder = "yolo8"
        else:
            raise ValueError(
                f"无法自动识别输出形状 {output.shape}；请显式设置 --decoder"
            )

    if decoder == "end2end":
        if array.shape[-1] != 6 and array.shape[0] == 6:
            array = array.T
        if array.shape[-1] != 6:
            raise ValueError("end2end 输出必须是 [N, 6]：xyxy, score, class_id")
        return _filter_and_nms(
            array[:, :4], array[:, 4], array[:, 5], score_threshold, iou_threshold
        )

    if decoder not in {"yolo8", "yolo5"}:
        raise ValueError(f"单输出不支持 decoder={decoder!r}")

    min_columns = 6 if decoder == "yolo5" else 5
    if array.shape[0] >= min_columns and array.shape[0] < array.shape[1]:
        array = array.T
    if array.shape[1] < min_columns:
        raise ValueError(f"{decoder} 输出列数不足，实际形状为 {array.shape}")

    boxes = _xywh_to_xyxy(array[:, :4])
    if decoder == "yolo5":
        class_scores = array[:, 5:]
        class_ids = class_scores.argmax(axis=1)
        scores = array[:, 4] * class_scores[np.arange(array.shape[0]), class_ids]
    else:
        class_scores = array[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(array.shape[0]), class_ids]
    return _filter_and_nms(boxes, scores, class_ids, score_threshold, iou_threshold)


def decode_efficient_nms(
    outputs: dict[str, np.ndarray], score_threshold: float
) -> DetectionTuple:
    """按名称和形状解析 TensorRT EfficientNMS 常见四输出。"""
    lowered = {name.lower(): value for name, value in outputs.items()}

    def by_keyword(*keywords: str) -> np.ndarray | None:
        for name, value in lowered.items():
            if any(keyword in name for keyword in keywords):
                return value
        return None

    boxes = by_keyword("box", "bbox")
    scores = by_keyword("score", "conf")
    classes = by_keyword("class", "label")
    count = by_keyword("num", "count")

    if boxes is None:
        boxes = next((value for value in outputs.values() if value.ndim >= 2 and value.shape[-1] == 4), None)
    if boxes is None or scores is None or classes is None:
        raise ValueError(
            "无法识别 EfficientNMS 输出，请确认输出名包含 boxes/scores/classes/num_dets"
        )

    boxes = np.asarray(boxes).reshape(-1, 4)
    scores = np.asarray(scores).reshape(-1)
    classes = np.asarray(classes).reshape(-1).astype(np.int64)
    limit = min(len(boxes), len(scores), len(classes))
    if count is not None and np.asarray(count).size:
        limit = min(limit, int(np.asarray(count).reshape(-1)[0]))
    boxes, scores, classes = boxes[:limit], scores[:limit], classes[:limit]
    valid = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & (scores >= score_threshold)
    return (
        boxes[valid].astype(np.float32),
        scores[valid].astype(np.float32),
        classes[valid].astype(np.int64),
    )


class ColorBlobDetector:
    """无需模型的教学后端，用来验证整条视频流水线。"""

    class_names = ["red", "green", "blue"]

    def __init__(
        self,
        input_size: tuple[int, int] = (640, 640),
        min_area: int = 120,
        simulate_inference_ms: float = 0.0,
    ):
        self.input_size = input_size
        self.min_area = min_area
        self.simulate_inference_ms = max(0.0, simulate_inference_ms)

    def infer(self, input_tensor: np.ndarray) -> DetectionTuple:
        tensor = np.asarray(input_tensor)
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise ValueError(f"ColorBlobDetector 期望 [1,3,H,W]，实际为 {tensor.shape}")
        rgb = np.clip(tensor[0].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        channels = [rgb[:, :, index] for index in range(3)]
        masks = [
            (channels[0] > 170) & (channels[1] < 120) & (channels[2] < 120),
            (channels[1] > 170) & (channels[0] < 120) & (channels[2] < 120),
            (channels[2] > 170) & (channels[0] < 120) & (channels[1] < 120),
        ]

        boxes: list[list[float]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        kernel = np.ones((3, 3), dtype=np.uint8)
        for class_id, boolean_mask in enumerate(masks):
            mask = (boolean_mask.astype(np.uint8) * 255)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                x, y, width, height = cv2.boundingRect(contour)
                boxes.append([x, y, x + width, y + height])
                scores.append(min(0.99, 0.75 + area / max(1.0, rgb.shape[0] * rgb.shape[1])))
                class_ids.append(class_id)

        if self.simulate_inference_ms:
            time.sleep(self.simulate_inference_ms / 1000.0)
        if not boxes:
            return _empty_detections()
        order = np.argsort(scores)[::-1]
        return (
            np.asarray(boxes, dtype=np.float32)[order],
            np.asarray(scores, dtype=np.float32)[order],
            np.asarray(class_ids, dtype=np.int64)[order],
        )

    def close(self) -> None:
        return None


@dataclass
class _CudaBuffer:
    shape: tuple[int, ...]
    dtype: np.dtype
    host: np.ndarray
    device: int


class TensorRTDetector:
    """TensorRT 11 命名张量 API 推理后端。

    该实现使用 ``cuda-python`` 分配与复制显存，固定/动态输入形状均可；
    数据相关的动态输出需要 IOutputAllocator，本课堂示例会明确拒绝该类 Engine。
    """

    def __init__(
        self,
        engine_path: str | Path,
        input_size: tuple[int, int] = (640, 640),
        decoder: str = "auto",
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_name: str | None = None,
    ):
        try:
            import tensorrt as trt
            from cuda.bindings import runtime as cudart
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT 后端需要 tensorrt 与 cuda-python；请安装 requirements-tensorrt.txt"
            ) from exc

        self.trt = trt
        self.cudart = cudart
        self.input_size = input_size
        self.decoder = decoder
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self._buffers: dict[str, _CudaBuffer] = {}
        self._closed = False

        engine_path = Path(engine_path)
        if not engine_path.is_file():
            raise FileNotFoundError(f"找不到 TensorRT Engine：{engine_path}")

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Engine 反序列化失败：{engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Execution Context 创建失败")

        self.io_names = [
            self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)
        ]
        input_names = [
            name
            for name in self.io_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name
            for name in self.io_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        if len(input_names) != 1:
            raise RuntimeError(f"课堂示例仅支持单输入 Engine，检测到输入：{input_names}")
        self.input_name = input_name or input_names[0]
        if self.input_name not in input_names:
            raise ValueError(f"输入名 {self.input_name!r} 不在 Engine 输入 {input_names} 中")

        error, stream = self.cudart.cudaStreamCreate()
        self._check_cuda(error, "cudaStreamCreate")
        self.stream = stream

    def _check_cuda(self, error, operation: str) -> None:
        if error != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{operation} 失败，CUDA error={error}")

    def describe(self) -> list[str]:
        lines: list[str] = []
        for name in self.io_names:
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            lines.append(f"{mode}: {name}, shape={shape}, dtype={dtype}")
        return lines

    def _ensure_buffer(self, name: str, shape: tuple[int, ...], dtype: np.dtype) -> _CudaBuffer:
        current = self._buffers.get(name)
        if current is not None and current.shape == shape and current.dtype == dtype:
            return current
        if current is not None:
            self._check_cuda(self.cudart.cudaFree(current.device)[0], f"cudaFree({name})")

        host = np.empty(shape, dtype=dtype)
        error, device = self.cudart.cudaMalloc(host.nbytes)
        self._check_cuda(error, f"cudaMalloc({name})")
        buffer = _CudaBuffer(shape=shape, dtype=dtype, host=host, device=int(device))
        self._buffers[name] = buffer
        return buffer

    def infer(self, input_tensor: np.ndarray) -> DetectionTuple:
        if self._closed:
            raise RuntimeError("TensorRTDetector 已关闭")
        input_array = np.ascontiguousarray(input_tensor)
        if not self.context.set_input_shape(self.input_name, tuple(input_array.shape)):
            raise RuntimeError(
                f"Engine 不接受输入形状 {input_array.shape}；请检查 optimization profile"
            )

        input_dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
        input_buffer = self._ensure_buffer(self.input_name, tuple(input_array.shape), input_dtype)
        np.copyto(input_buffer.host, input_array.astype(input_dtype, copy=False), casting="unsafe")

        output_buffers: dict[str, _CudaBuffer] = {}
        for name in self.output_names:
            shape = tuple(int(value) for value in self.context.get_tensor_shape(name))
            if not shape or any(value < 0 for value in shape):
                raise RuntimeError(
                    f"输出 {name!r} 具有数据相关动态形状 {shape}；课堂后端暂不支持 IOutputAllocator"
                )
            dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
            output_buffers[name] = self._ensure_buffer(name, shape, dtype)

        kind = self.cudart.cudaMemcpyKind
        self._check_cuda(
            self.cudart.cudaMemcpyAsync(
                input_buffer.device,
                input_buffer.host.ctypes.data,
                input_buffer.host.nbytes,
                kind.cudaMemcpyHostToDevice,
                self.stream,
            )[0],
            "cudaMemcpyAsync(H2D)",
        )
        self.context.set_tensor_address(self.input_name, input_buffer.device)
        for name, buffer in output_buffers.items():
            self.context.set_tensor_address(name, buffer.device)
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("execute_async_v3 提交失败")

        for name, buffer in output_buffers.items():
            self._check_cuda(
                self.cudart.cudaMemcpyAsync(
                    buffer.host.ctypes.data,
                    buffer.device,
                    buffer.host.nbytes,
                    kind.cudaMemcpyDeviceToHost,
                    self.stream,
                )[0],
                f"cudaMemcpyAsync(D2H:{name})",
            )
        self._check_cuda(self.cudart.cudaStreamSynchronize(self.stream)[0], "cudaStreamSynchronize")

        outputs = {name: buffer.host.copy() for name, buffer in output_buffers.items()}
        decoder = self.decoder
        if decoder == "efficientnms" or (decoder == "auto" and len(outputs) > 1):
            return decode_efficient_nms(outputs, self.score_threshold)
        if len(outputs) != 1:
            raise ValueError(f"decoder={decoder!r} 需要单输出 Engine，实际输出为 {list(outputs)}")
        return decode_single_output(
            next(iter(outputs.values())), decoder, self.score_threshold, self.iou_threshold
        )

    def close(self) -> None:
        if self._closed:
            return
        for name, buffer in list(self._buffers.items()):
            error = self.cudart.cudaFree(buffer.device)[0]
            self._check_cuda(error, f"cudaFree({name})")
        self._buffers.clear()
        if hasattr(self, "stream"):
            self._check_cuda(self.cudart.cudaStreamDestroy(self.stream)[0], "cudaStreamDestroy")
        self._closed = True

    def __enter__(self) -> "TensorRTDetector":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

