# 实时视频流物体检测

## 一、运行流程

### 1. 创建环境

在进入视频检测代码之前，先为本节创建独立环境。下面以 Linux 终端为例，环境名称统一使用 `isaac`。首先进入代码目录：

```bash
cd /代码所在路径/isaac
```

然后创建 Python 3.11 环境并激活：

```bash
conda create -n isaac python=3.11 -y
conda activate isaac
```

激活完成后，终端提示符前通常会出现 `(isaac)`。接着确认当前使用的 Python 确实来自这个环境：

```bash
which python
python -V
```

`which python` 的结果中应包含 `envs/isaac/bin/python`，Python 版本应为 3.11。确认无误后安装基础视频检测依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

这里使用 `python -m pip`，可以让安装命令明确绑定到刚刚激活的 Python 环境，减少把依赖装进其他环境的情况。

如果本机已经准备好 NVIDIA 驱动、CUDA 和本课程统一的 TensorRT 运行条件，再安装 TensorRT 相关依赖：

```bash
python -m pip install -r requirements-tensorrt.txt
```

[版本提示] `requirements-tensorrt.txt` 默认安装 TensorRT 11.x 通用 Python 包。实际授课机器应根据统一的 CUDA 主版本选择对应的 TensorRT 包；例如 CUDA 12 环境使用 `tensorrt-cu12`，CUDA 13 环境使用 `tensorrt-cu13`。已经构建好的 Engine 还要与 TensorRT、CUDA、GPU 和构建配置保持兼容，换机器后优先从 ONNX 重新构建 Engine。

安装完成后执行环境检查：

```bash
python inspect_environment.py
```

只验证 OpenCV 视频流水线时，NumPy 和 OpenCV 显示为 `OK` 即可。准备运行 TensorRT Engine 时，执行更严格的检查：

```bash
nvidia-smi
python inspect_environment.py \
  --engine models/detector.engine \
  --require-tensorrt
```

如果终端提示 `conda activate` 尚未初始化，可执行：

```bash
conda init bash
source ~/.bashrc
conda activate isaac
```

[提示]`opencv-python` 与 `opencv-python-headless` 二选一，不要在同一环境中同时安装。需要 `--display` 窗口时必须使用前者，并确保系统存在图形桌面。

### 2. 生成测试视频

测试视频完全由脚本绘制，不需要联网下载：

```bash
python generate_demo_video.py
```

如果当前 OpenCV 无法写 MP4，可改用：

```bash
python generate_demo_video.py --output assets/demo.avi
```

### 3. 跑同步基线

```bash
python video_detection.py \
  --source assets/demo.mp4 \
  --backend color \
  --mode sync \
  --output outputs/sync_detected.mp4 \
  --summary-json outputs/sync_metrics.json
```

有图形桌面时追加 `--display`，按 `q` 或 `Esc` 退出。默认不打开窗口，因此脚本也能在 SSH、容器和 CI 环境中运行。

### 4. 运行 TensorRT

YOLOv8 原始单输出示例：

```bash
python video_detection.py \
  --source assets/demo.mp4 \
  --backend tensorrt \
  --engine models/detector.engine \
  --classes models/classes.txt \
  --decoder yolo8 \
  --input-size 640 640 \
  --score-threshold 0.25 \
  --iou-threshold 0.45 \
  --mode sync \
  --output outputs/trt_detected.mp4
```

若 Engine 已集成 NMS 且输出为 `num_dets / boxes / scores / classes`，将参数改为：

```bash
--decoder efficientnms
```

更多输出契约见 `models/README.md`。如果模型的预处理或输出结构不同，应修改对应函数，不能仅靠更换 Engine 文件解决。

## 二、文件结构

```text
isaac/
├── video_detection.py          # 主入口
├── generate_demo_video.py      # 生成自包含测试视频
├── inspect_environment.py      # 环境检查
├── realtime_detection/
│   ├── core.py                 # 帧包、Letterbox 与检测结果契约
│   ├── preprocess.py           # 预处理和坐标还原
│   ├── detectors.py            # 彩色目标与 TensorRT 后端
│   ├── sources.py              # OpenCV 视频源和最新帧槽位
│   ├── metrics.py              # FPS 与延迟分布
│   ├── visualization.py        # 画框和状态叠加
│   └── pipeline.py             # 同步/最新帧主循环
├── tests/                      # 核心功能测试
├── assets/                     # 演示视频
├── models/                     # Engine 与类别文件
└── outputs/                    # 输出视频和指标
```
