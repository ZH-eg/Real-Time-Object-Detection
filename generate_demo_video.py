from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _bounce(frame_index: int, speed: float, limit: int) -> int:
    if limit <= 0:
        return 0
    value = (frame_index * speed) % (2 * limit)
    return int(value if value <= limit else 2 * limit - value)


def generate_demo_video(
    output_path: str | Path,
    width: int = 640,
    height: int = 360,
    fps: float = 30.0,
    seconds: float = 8.0,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    codec = "MJPG" if output.suffix.lower() == ".avi" else "mp4v"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"无法创建 {output}；可改用 assets/demo.avi 以使用 MJPG 编码"
        )

    frame_count = int(round(fps * seconds))
    try:
        for frame_id in range(frame_count):
            frame = np.full((height, width, 3), 28, dtype=np.uint8)
            for x in range(0, width, 80):
                cv2.line(frame, (x, 0), (x, height), (42, 42, 42), 1)
            for y in range(0, height, 60):
                cv2.line(frame, (0, y), (width, y), (42, 42, 42), 1)

            red_x = 30 + _bounce(frame_id, 3.2, max(1, width - 150))
            green_y = 65 + _bounce(frame_id, 1.7, max(1, height - 180))
            blue_x = 70 + _bounce(frame_id, 2.2, max(1, width - 210))
            blue_y = 210 + int(25 * np.sin(frame_id / 13.0))

            cv2.rectangle(frame, (red_x, 75), (red_x + 85, 145), (0, 0, 255), -1)
            cv2.rectangle(frame, (width - 190, green_y), (width - 105, green_y + 72), (0, 255, 0), -1)
            cv2.rectangle(frame, (blue_x, blue_y), (blue_x + 105, blue_y + 62), (255, 0, 0), -1)
            cv2.circle(frame, (width // 2, height // 2), 28, (110, 110, 110), -1)

            cv2.putText(
                frame,
                f"frame {frame_id:04d}  t={frame_id / fps:05.2f}s",
                (16, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="生成无需下载的彩色目标演示视频")
    parser.add_argument("--output", default="assets/demo.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()
    path = generate_demo_video(args.output, args.width, args.height, args.fps, args.seconds)
    print(f"已生成：{path.resolve()}")


if __name__ == "__main__":
    main()

