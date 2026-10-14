from __future__ import annotations

import argparse
from pathlib import Path

from realtime_detection.detectors import ColorBlobDetector, TensorRTDetector
from realtime_detection.pipeline import PipelineConfig, run_pipeline
from realtime_detection.sources import OpenCVVideoSource, parse_video_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="本地视频、摄像头或 RTSP 实时物体检测"
    )
    parser.add_argument("--source", default="assets/demo.mp4", help="视频路径、RTSP URL 或摄像头索引")
    parser.add_argument("--live", action="store_true", help="将输入标记为实时源")
    parser.add_argument(
        "--simulate-live",
        action="store_true",
        help="本地视频按标称 FPS 原速读取，便于演示最新帧策略",
    )
    parser.add_argument("--backend", choices=["color", "tensorrt"], default="color")
    parser.add_argument("--engine", default="models/detector.engine")
    parser.add_argument(
        "--decoder",
        choices=["auto", "end2end", "yolo8", "yolo5", "efficientnms"],
        default="auto",
    )
    parser.add_argument("--input-name", default=None, help="多版本 Engine 输入名不一致时显式指定")
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 640),
    )
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--classes", default=None, help="类别文件，每行一个名称")
    parser.add_argument("--mode", choices=["sync", "latest"], default="sync")
    parser.add_argument("--display", action="store_true", help="打开 OpenCV 显示窗口，按 q 或 Esc 退出")
    parser.add_argument("--output", default=None, help="保存检测视频，例如 outputs/detected.mp4")
    parser.add_argument("--output-fps", type=float, default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示处理到输入结束")
    parser.add_argument("--process-every", type=int, default=1, help="每 N 帧处理一帧")
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=float, default=None)
    parser.add_argument(
        "--simulate-inference-ms",
        type=float,
        default=0.0,
        help="仅 color 后端：人为增加推理耗时，演示过载丢帧",
    )
    return parser.parse_args()


def load_class_names(path: str | None, fallback: list[str]) -> list[str]:
    if path is None:
        return fallback
    class_file = Path(path)
    if not class_file.is_file():
        raise FileNotFoundError(f"找不到类别文件：{class_file}")
    names = [line.strip() for line in class_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"类别文件为空：{class_file}")
    return names


def main() -> None:
    args = parse_args()
    input_size = tuple(args.input_size)
    source_value = parse_video_source(args.source)
    live = args.live or isinstance(source_value, int) or str(source_value).lower().startswith(("rtsp://", "rtmp://"))

    source = OpenCVVideoSource(
        source_value,
        live=live,
        simulate_live=args.simulate_live,
        requested_width=args.camera_width,
        requested_height=args.camera_height,
        requested_fps=args.camera_fps,
    )
    print(source.describe())

    if args.backend == "color":
        detector = ColorBlobDetector(
            input_size=input_size,
            simulate_inference_ms=args.simulate_inference_ms,
        )
        fallback_names = ColorBlobDetector.class_names
    else:
        detector = TensorRTDetector(
            engine_path=args.engine,
            input_size=input_size,
            decoder=args.decoder,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            input_name=args.input_name,
        )
        print("TensorRT Engine I/O:")
        for line in detector.describe():
            print(f"  {line}")
        fallback_names = []

    class_names = load_class_names(args.classes, fallback_names)
    config = PipelineConfig(
        mode=args.mode,
        display=args.display,
        output_path=args.output,
        output_fps=args.output_fps,
        max_frames=args.max_frames,
        process_every=args.process_every,
        log_every=args.log_every,
        summary_json=args.summary_json,
    )
    run_pipeline(source, detector, class_names, config)


if __name__ == "__main__":
    main()
