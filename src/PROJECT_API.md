# 项目结构与接口说明

> 适用范围：仓库 `invc-yolo-garbage`。本文按当前源码梳理生产代码中的模块、函数和命令行接口；测试代码用于验证行为，不作为产品接口列出。

## 1. 项目概览

这是一个基于 Ultralytics YOLO11 的垃圾识别项目，包含两条工作流：

1. **训练工作流**：检查数据集 →（可选）生成练习数据清单 → 载入 YOLO 权重并微调。
2. **Windows 实时演示工作流**：摄像头/视频 → YOLO 检测与 ByteTrack 跟踪 → 垃圾分类映射和速度估算 → 画面显示/录制 → JSONL 日志及可选串口输出。

当前没有 HTTP/REST 服务端。项目接口由 Python 模块函数和类、命令行脚本、JSONL 文件格式、串口 JSON 行协议组成。

## 2. 目录结构

| 路径 | 用途 |
|---|---|
| `src/invc_vision/` | 实时视觉应用及可复用模块。|
| `src/invc_vision/app.py` | 实时应用入口：参数解析、摄像头/视频读取、检测/跟踪、速度和遥测编排。|
| `src/invc_vision/categories.py` | YOLO 物品小类到四种垃圾大类的映射。|
| `src/invc_vision/motion.py` | 按跟踪 ID 和真实时间戳估算速度、维护轨迹。|
| `src/invc_vision/calibration.py` | 棋盘格平面像素到厘米的单应变换。|
| `src/invc_vision/telemetry.py` | JSONL 日志及紧凑串口协议编码、发送。|
| `scripts/run_live.py` | 将 `src` 加入导入路径并启动实时应用。|
| `scripts/calibrate_camera.py` | 打开摄像头/视频，交互拍摄棋盘格并保存平面标定。|
| `scripts/serial_monitor.py` | 串口接收 JSON 行并统计序号缺失。|
| `scripts/train_garbage40.py` | 检查 40 类数据集并运行 YOLO 微调。|
| `scripts/build_practice_split.py` | 去除训练/验证重复图，生成练习用清单及 YAML；不改原数据集。|
| `scripts/audit_dataset.py` | 只读审计图片、标签匹配关系和类别标注。|
| `scripts/smoke_test.py` | 检查 CUDA、载入权重并执行最小训练验证。|
| `configs/` | 数据集 YAML、类别映射 YAML 和示例配置。|
| `data/` | 数据集、生成的清单等数据。|
| `models/`、`weights/` | 模型权重。|
| `outputs/`、`runs/` | 训练结果、日志、录制视频和标定文件等运行产物。|
| `tests/` | 速度、协议、标定和实时流程的自动化测试。|
| `docker/`、`compose.yaml`、`requirements*.txt` | 训练容器及依赖配置。|
| `README.md`、`docs/` | 快速开始、环境和项目说明。|

## 3. 主要调用流程

### 3.1 实时检测

`run_live.py` → `app.main()` → `parse_args()` → `run(args)`。`run` 加载 YOLO 权重和类别映射，按需读取平面标定，打开视频源，然后循环读取画面。每帧调用 `model.track(..., persist=True, tracker="bytetrack.yaml")` 获取检测框和跟踪 ID；框底边中心用于速度计算。处理后的记录交给 `Telemetry.write()`，画面可显示或录制。

- 摄像头速度时间基准：`time.monotonic()` 实时钟。
- 视频速度时间基准：视频时间戳；无有效递增时间戳时按视频 FPS 推算。
- 未标定时速度单位为 `px/s`；启用标定后为 `cm/s`。
- 目标框格式在遥测中为 `[x, y, width, height]` 像素，左上角为 `(x,y)`。

### 3.2 标定

`calibrate_camera.py` 调用实时模块的 `open_capture()` 读取画面。按空格后调用 `PlaneCalibration.from_chessboard()`，拟合平面像素到厘米的单应矩阵，并保存 JSON。实时程序载入时会检查标定分辨率与视频源分辨率一致。

### 3.3 训练

训练入口为 `scripts/train_garbage40.py`。默认先检查 YAML、图片目录和图片对应的标签文件，然后使用指定权重调用 Ultralytics `YOLO.train()`。默认参数是 1 epoch、5% 数据，仅用于快速试跑。`build_practice_split.py` 可先生成消除重复图片的练习清单。

## 4. Python 模块接口与函数清单

### 4.1 `invc_vision.app`

| 函数 | 参数/返回 | 作用 |
|---|---|---|
| `parse_args()` | 无参数；返回 `argparse.Namespace` | 定义并解析实时检测命令行参数。参数见下方“实时程序命令行”。|
| `open_capture(source, width, height)` | `str, int, int`；返回 `cv2.VideoCapture` | 根据纯数字 `source` 打开摄像头，否则按文件路径打开视频；Windows 摄像头优先 DirectShow，失败后尝试 MSMF；为摄像头设置期望分辨率。打开失败抛 `RuntimeError`。|
| `color_for(track_id)` | `int \| None`；返回 BGR 三元组 | 为跟踪 ID 生成稳定的绘制颜色；无 ID 使用默认颜色。|
| `draw_track(frame, box, label, trail, color)` | 画面、`(x1,y1,x2,y2)`、文字、轨迹点、BGR 色；无返回值 | 在画面上画检测框、标签和历史轨迹线。|
| `run(args)` | `argparse.Namespace`；返回退出码 `int` | 执行实时处理主循环；校验参数及模型，加载依赖，读帧、跟踪、分类、测速、写日志/串口、显示/录制，并在退出时释放资源。异常会向上传播。|
| `main()` | 无参数；返回退出码 `int` | 调用 `parse_args()` 和 `run()`，供命令行启动。|

**`run` 内部每帧主要步骤**：读取帧并计算 FPS/时间戳 → YOLO 跟踪 → 从结果提取 `track_id`、坐标、类别、置信度 → 查询垃圾大类 → 计算底边中心速度和轨迹 → 构造检测项 → 清理超时轨迹 → 生成帧记录 → 写遥测 → 叠加 FPS/延迟信息 → 可选视频写入及窗口显示。

### 4.2 `invc_vision.categories`

| 类/方法 | 参数/返回 | 作用 |
|---|---|---|
| `CategoryMap(path, model_names)` | YAML 路径、`{class_id: item_name}`；构造对象 | 读取配置中的 `categories`，建立物品小类到垃圾大类的映射；检查重复映射、模型中漏配和配置里多配。发现不匹配时抛 `ValueError`。|
| `CategoryMap.category(item)` | 物品小类名称；返回大类名称 | 查询物品所属垃圾类别；未知名称会抛 `KeyError`。|

类别映射配置位于 `configs/garbage40_categories.yaml`。已配置大类：`有害垃圾`、`可回收垃圾`、`厨余垃圾`、`其他垃圾`。

### 4.3 `invc_vision.motion`

| 类/方法 | 参数/返回 | 作用 |
|---|---|---|
| `Speed` | 不可变数据类，字段 `instant`、`average_15`、`samples` | 表示本次瞬时速度、当前历史窗口平均速度和有效采样数量。字段名 `average_15` 实际随 `MotionTracker.window` 配置，默认窗口为 15。|
| `MotionTracker(window=15, stale_seconds=1.0)` | 窗口长度、轨迹失效秒数；构造对象 | 按跟踪 ID 管理有界历史；窗口需至少 2，失效时间需大于 0。|
| `MotionTracker.update(track_id, point, timestamp)` | 跟踪 ID、二维坐标、秒时间戳；返回 `Speed` | 加入一个位置样本。时间戳必须对该 ID 严格递增，否则抛 `ValueError`；中断超过 `stale_seconds` 后重新开始历史。瞬时速度按相邻两点计算，平均速度按窗口内累计路程/时间跨度计算。|
| `MotionTracker.prune(timestamp)` | 当前时间戳；无返回值 | 删除最后一次更新距当前超过 `stale_seconds` 的跟踪历史。|
| `MotionTracker.trail(track_id)` | 跟踪 ID；返回整数像素点列表 | 返回指定目标当前保存的轨迹坐标，坐标四舍五入；没有历史时返回空列表。|

速度坐标单位由调用方决定：直接输入像素坐标得到 `px/s`；先经平面标定转换后输入厘米坐标得到 `cm/s`。

### 4.4 `invc_vision.calibration`

| 类/方法 | 参数/返回 | 作用 |
|---|---|---|
| `PlaneCalibration(matrix, image_size)` | 3×3 单应矩阵、`(宽, 高)`；构造对象 | 创建平面标定对象；矩阵形状不为 3×3 时抛 `ValueError`，内部以 `float64` 保存。|
| `PlaneCalibration.point_cm(point_px)` | `(x,y)` 像素；返回 `(x,y)` 厘米 | 用 OpenCV 透视变换把图像坐标投影到标定平面坐标。|
| `PlaneCalibration.save(path)` | JSON 文件路径；无返回值 | 创建父目录并保存 `homography_px_to_cm` 矩阵及 `image_size`。|
| `PlaneCalibration.load(path)` | JSON 路径；返回 `PlaneCalibration` | 从保存的 JSON 读取矩阵和图像尺寸并构造对象。|
| `PlaneCalibration.from_chessboard(frame, columns, rows, square_cm)` | BGR 图像、横/纵内角点数、方格边长厘米；返回 `(标定对象, 预览图)` | 检测棋盘格角点，按已知方格间距拟合像素到厘米的单应矩阵。角点数或边长无效、棋盘格未检测到、拟合内点不足时抛异常。|

标定只适用于棋盘所在平面，不包含完整镜头畸变校正。运行摄像头角度、图像尺寸或被测平面变化时应重新标定。

### 4.5 `invc_vision.telemetry`

| 类/函数 | 参数/返回 | 作用 |
|---|---|---|
| `encode_frame(record)` | 完整帧记录字典；返回 UTF-8 `bytes` | 将记录编码为紧凑 JSON，保留中文，并追加换行符，供 JSONL 日志写入。|
| `encode_wire_frame(record)` | 完整帧记录字典；返回 UTF-8 `bytes` | 将记录压缩成串口协议字段并编码为 JSON 行。垃圾类别映射为 `H/R/K/O`。|
| `Telemetry(log_path, port=None, baudrate=460800)` | 日志路径、可选串口端口、波特率；构造对象 | 创建日志目录并以追加模式打开 UTF-8 日志；指定端口时打开 pySerial URL。串口初始化异常会关闭日志后继续抛出。|
| `Telemetry.write(record)` | 完整帧记录；无返回值 | 始终将完整格式写入日志；若串口可用，另外发送紧凑格式。串口写失败时记录 `serial_error`、发出警告、关闭串口，后续仍可写日志。|
| `Telemetry.close()` | 无参数；无返回值 | 关闭日志和仍打开的串口。|
| `Telemetry.__enter__()` | 无参数；返回自身 | 支持 `with Telemetry(...) as telemetry`。|
| `Telemetry.__exit__(*args)` | 上下文退出参数；无返回值 | 离开 `with` 时调用 `close()`。|

## 5. 命令行脚本接口

### 实时检测：`scripts/run_live.py`

脚本本身只负责把 `src` 放入 Python 导入路径并调用 `invc_vision.app.main()`。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model` | `outputs/garbage40/trial/weights/best.pt` | YOLO 权重文件。|
| `--source` | `0` | 摄像头编号（字符串数字）或视频文件路径。|
| `--categories` | `configs/garbage40_categories.yaml` | 小类到垃圾大类映射 YAML。|
| `--calibration` | 无 | 平面标定 JSON；设置后速度以 `cm/s` 输出。|
| `--serial-port` | 无 | 串口或 pySerial URL；未设置时只写日志。|
| `--baudrate` | `460800` | 串口波特率。|
| `--device` | `0` | Ultralytics 推理设备；`cpu` 可用于调试。|
| `--imgsz` | `640` | YOLO 推理图像尺寸。|
| `--conf` | `0.25` | 置信度阈值，必须在 0 和 1 之间。|
| `--width` / `--height` | `1280` / `720` | 摄像头请求分辨率。|
| `--log` | 自动生成 `outputs/live/session_时间.jsonl` | 日志文件路径。|
| `--record` | 无 | 输出带框视频路径，例如 `outputs/live/demo.mp4`。|
| `--headless` | 关闭 | 不显示窗口。摄像头无窗口运行时需设置 `--max-frames`。|
| `--max-frames` | `0` | 最大处理帧数；0 表示不限制。|

### 棋盘格标定：`scripts/calibrate_camera.py`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--source` | `0` | 摄像头编号或视频路径。|
| `--columns` / `--rows` | `9` / `6` | 棋盘格横向/纵向内角点数量。|
| `--square-cm` | 必填 | 棋盘单格的实际边长，单位厘米。|
| `--width` / `--height` | `1280` / `720` | 摄像头请求分辨率。|
| `--output` | `outputs/calibration/plane.json` | 标定 JSON 输出路径。|

运行时按空格拍摄并尝试标定，按 Q 退出。标定失败会显示错误并继续取景。

### 训练：`scripts/train_garbage40.py`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--data` | `/workspace/configs/garbage40.yaml` | YOLO 数据集配置。|
| `--model` | `/workspace/models/yolo11n.pt` | 预训练权重。|
| `--epochs` | `1` | 训练轮数。|
| `--fraction` | `0.05` | 使用的训练数据比例，范围为 `(0,1]`。|
| `--imgsz` / `--batch` | `640` / `8` | 输入尺寸和批次大小。|
| `--workers` | `4` | 数据加载 worker 数。|
| `--device` | `0` | 训练设备。|
| `--name` | `trial` | 训练任务名。|
| `--check-only` | 关闭 | 仅检查数据配置和文件，不启动训练。|

脚本中的 `validate_dataset(data_yaml)` 检查配置文件、train/val 图片与标签目录以及每张图片的同名标签。`parse_args()` 创建参数对象；`main()` 校验参数并在非 `--check-only` 时调用 Ultralytics 训练。

### 练习数据划分：`scripts/build_practice_split.py`

无命令行参数，路径由脚本常量确定。`main()` 去掉 train/val 间同名且 SHA-256 相同的图片；按编号把含 `Bone` 标注的连续帧组移至验证集；生成 train/val 图片清单和练习 YAML，不复制或修改原始图片/标签。

| 函数 | 作用 |
|---|---|
| `image_files(split)` | 按文件名排序列出指定 split 下支持扩展名的图片。|
| `sha256(path)` | 分块读取文件并返回 SHA-256 摘要，用于精确内容去重。|
| `frame_group(path)` | 从文件名末尾提取数字并按每 50 帧分组；无数字时返回 `None`。|
| `has_class(path, class_id=7)` | 检查训练集同名标签文件中是否含指定类别，默认类别 ID 7（Bone）。|
| `container_path(path)` | 把宿主机数据路径转换为容器内 `/workspace/...` 路径。|
| `main()` | 构建训练/验证清单及 `garbage40_practice_clean.yaml`，检查两边不重叠并打印统计。|

### 数据集审计：`scripts/audit_dataset.py`

无命令行参数，固定检查 `configs/garbage40.yaml` 指向的数据目录。`main()` 对 train/val 统计图片、标签、未匹配图片、孤立标签和每类实例数，并检查 YOLO 标注是否为 5 列、类别 ID 有效、归一化坐标在 `[0,1]` 且宽高为正。此脚本只读，不会修复数据。

### CUDA/训练冒烟检查：`scripts/smoke_test.py`

| 函数 | 作用 |
|---|---|
| `check_torch()` | 打印 PyTorch/CUDA/GPU 信息；CUDA 不可用时断言失败。|
| `load_model(weights)` | 打印 Ultralytics 版本并载入 YOLO 权重，返回模型。|
| `smoke_train(model, data, epochs, imgsz, batch, device)` | 在 `runs/smoke/train` 执行最小训练，检查 `weights/last.pt` 生成并返回结果目录。|
| `main()` | 解析冒烟测试参数；默认权重或 coco8 配置缺失时回退到 Ultralytics 自动下载名称，然后依次检查 CUDA、载入模型和训练。|

### 串口接收：`scripts/serial_monitor.py`

无可复用的模块函数，`main()` 解析 `--port`（必填）和 `--baudrate`（默认 460800），逐行读取串口 JSON，读取紧凑字段 `s/f/d`，统计收到帧数和序号缺失。无法解析的行会打印错误并跳过；Ctrl+C 时输出汇总。

## 6. 数据接口格式

### 6.1 完整 JSONL 日志

每处理一帧追加一行 JSON，字段如下：

| 字段 | 含义 |
|---|---|
| `version` | 完整记录格式版本，当前为 `1`。|
| `seq` | 从 1 开始的帧序号。|
| `timestamp_utc` | UTC ISO 8601 时间，精确到毫秒。|
| `fps` | 平滑后的处理帧率。|
| `inference_ms` | Ultralytics 返回的预处理/推理/后处理耗时之和；无数据时为 `null`。|
| `media_time_s` | 视频媒体时间；摄像头记录为 `null`。|
| `speed_unit` | `px/s` 或 `cm/s`。|
| `frame_wh_px` | `[帧宽, 帧高]`。|
| `detections` | 本帧检测目标数组。|

`detections` 中每项包含：`track_id`、`class_id`、`item`、`category`、`confidence`、`bbox_xywh_px`、`speed_instant`、`speed_avg_15`、`speed_samples`、`speed_unit`。没有跟踪 ID 或尚无速度数据时，相关速度值为 `null`，采样数为 0。

### 6.2 串口紧凑 JSON 行协议

每帧为一行 UTF-8 JSON，行尾为换行符。字段缩写：

| 字段 | 完整日志对应字段 | 含义 |
|---|---|---|
| `v` | `version` | 协议版本。|
| `s` | `seq` | 帧序号，可用于统计丢帧。|
| `t` | `timestamp_utc` | UTC 时间。|
| `f` | `fps` | 处理帧率。|
| `u` | `speed_unit` | 速度单位。|
| `d` | `detections` | 目标数组。|

`d` 中每个目标按固定顺序编码为：`[track_id, class_id, category_code, x, y, width, height, confidence, instant_speed, avg_15_speed]`。类别码 `H/R/K/O` 分别表示有害/可回收/厨余/其他垃圾。串口发送失败不会停止日志记录；`Telemetry.serial_error` 保存最近一次错误。

## 7. 依赖与重要运行约束

- 实时功能依赖 OpenCV、Ultralytics、PyYAML；串口功能额外依赖 pySerial。标定使用 OpenCV 和 NumPy。
- YOLO 跟踪 ID 由 Ultralytics ByteTrack 提供；本项目只对有 ID 的目标计算速度。
- 类别映射必须覆盖模型 `names` 中的全部类别，且不能包含模型不存在的类别。
- 标定图像尺寸必须与当前捕获源一致；测量目标应在棋盘标定平面上运动。
- `--headless` 摄像头运行需限制 `--max-frames`，否则无法通过窗口按键退出。
- 训练脚本默认路径 `/workspace/...` 面向 Docker 容器；实时演示脚本默认路径由源码位置推导，面向项目工作区。
