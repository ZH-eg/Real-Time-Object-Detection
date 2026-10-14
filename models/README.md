# 模型目录

把自己在前面生成并验证过的 TensorRT Engine 放在这里，例如：

```text
models/detector.engine
models/classes.txt
```

Engine 属于可执行二进制产物，只加载自己构建或来自可信渠道的文件。Engine 与 TensorRT、CUDA、GPU 及构建配置存在兼容约束，换机器后优先从 ONNX 重新构建。

代码支持以下输出契约：

- `end2end`：单输出 `[N, 6]` 或 `[1, N, 6]`，每行是 `x1,y1,x2,y2,score,class_id`；
- `yolo8`：单输出 `[1, 4+C, N]` 或 `[1, N, 4+C]`，框为 `cx,cy,w,h`；
- `yolo5`：单输出 `[1, N, 5+C]` 或转置形状，包含 objectness；
- `efficientnms`：常见四输出，名称应包含 `boxes`、`scores`、`classes`、`num_dets` 等关键词。

上述框坐标均应位于模型输入的像素坐标系中。若你的模型输出归一化坐标、不同排列或自定义插件结果，请修改 `realtime_detection/detectors.py` 中的解码器。