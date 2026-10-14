from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def module_version(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, str(getattr(module, "__version__", "已导入，未提供 __version__"))


def main() -> int:
    parser = argparse.ArgumentParser(description="检查第18章第4节运行环境")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--require-tensorrt", action="store_true")
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    states = {}
    for name in ("numpy", "cv2", "cuda.bindings", "tensorrt"):
        ok, detail = module_version(name)
        states[name] = ok
        print(f"{name}: {'OK' if ok else 'MISSING'} ({detail})")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"GPU: {result.stdout.strip() or result.stderr.strip()}")
    else:
        print("GPU: nvidia-smi 不在 PATH 中")

    if args.engine:
        engine = Path(args.engine)
        print(f"Engine: {'OK' if engine.is_file() else 'MISSING'} ({engine})")

    base_ok = states.get("numpy", False) and states.get("cv2", False)
    trt_ok = states.get("cuda.bindings", False) and states.get("tensorrt", False) and nvidia_smi is not None
    if not base_ok:
        return 2
    if args.require_tensorrt and not trt_ok:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

