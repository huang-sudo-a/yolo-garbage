# USB 摄像头识别使用说明

Windows 本机识别环境与摄像头已接通，默认加载第二版 YOLO11n。训练仍使用 Docker，实时识别使用本项目的 Windows `.venv`。

2026-11-03 更新：已接入独立卡尔曼滤波与匈牙利匹配，默认显示目标 ID 和轨迹。详情见 [跟踪实现与验证](tracking-20261103.md)。

## 怎样打开和关闭

双击项目根目录的：

`D:\invc\work\invc-yolo-garbage\启动USB识别.cmd`

窗口显示物品名称、垃圾大类、置信度、目标数量、实际处理 FPS、模型推理耗时。

- 在检测窗口按 **Q** 退出，也可以关闭窗口。
- 按 **S** 保存当前带框画面。截图保存在项目的 `outputs/live/`。
- 每次运行的检测日志自动保存在 `outputs/live/session_时间.jsonl`。
- 摄像头只能被部分软件独占使用；再次启动前先退出上一份检测窗口。

也可以在项目目录的 PowerShell 中启动：

```powershell
.\.venv\Scripts\python.exe scripts/run_live.py
```

默认摄像头编号为 0。如需另一台摄像头，运行：

```powershell
.\.venv\Scripts\python.exe scripts/run_live.py --source 1
```

## 正在运行哪个程序

入口是 `scripts/run_live.py`，它调用 `src/invc_vision/app.py`。

```text
USB 摄像头
    ↓ OpenCV 读取画面
第二版 best.pt 在 NVIDIA GPU 上检测
    ↓ 按置信度与重叠框过滤
六类物品名称 → 垃圾大类
    ↓ Pillow 渲染中文
窗口显示 + 每帧检测日志
```

当前程序使用 `model.predict()` 做检测，再接独立卡尔曼/匈牙利跟踪，不调用 YOLO 内置跟踪。跟踪 ID 已输出，正式速度字段仍保留为空。使用 `--no-tracking` 可以切回纯检测。

## 模型、类别与参数

默认模型：`outputs/self_dataset/v2_30ep/weights/best.pt`。

类别映射文件：`configs/self_categories.yaml`。

| 模型中的物品名称 | 垃圾大类 |
|---|---|
| 纸杯、塑料瓶 | 可回收垃圾 |
| 土豆 | 厨余垃圾 |
| 瓷片、石子、砖块 | 其他垃圾 |

模型当前没有有害垃圾物品类；没有把已有六类改成四类重新训练。

默认识别参数：`conf=0.5`、NMS `iou=0.5`、`agnostic_nms=True`、`imgsz=640`、`rect=False`、GPU 0。

默认请求摄像头 1280×720、30 FPS、MJPG 格式。实际输出由设备支持情况决定；程序启动时会打印实际尺寸和设备报告帧率，窗口中的 FPS 则是整条处理流程的实际速度。

例如临时调置信度，不需要重新训练：

```powershell
.\.venv\Scripts\python.exe scripts/run_live.py --conf 0.6
```

这会同时影响误报和漏检，需要用实际物品对照观察。

## 本次验证记录

- Windows 识别到 `Web Camera`，设备状态正常，编号 0 成功读取。
- 实测画面为 1280×720，设备报告 30 FPS。
- Python 3.13.7，PyTorch 2.8.0+cu128，CUDA 12.8，RTX 5060 Ti。
- Ultralytics 8.4.164，OpenCV 4.14.0（WIN32UI），Pillow 12.3.0。
- 当前项目独立环境使用已验证的旧项目依赖副本，不依赖指向旧项目的目录链接；旧环境没有改动。
- 本机 `pip check` 通过。环境版本清单：`docs/windows-live-requirements-freeze.txt`。
- 8 项测试在 Docker 和 Windows 环境均通过，包括检测过滤参数、中文渲染、日志、摄像头断开清理及保留模块。
- 真实摄像头连续处理 90 帧，日志 `outputs/live/usb_check.jsonl`，检查画面 `outputs/live/usb_check.jpg`。
- 去掉前 10 帧后，这次检查的中位处理速度约 **9.9 FPS**，中位模型推理耗时约 **4.7 ms**。不能用单次推理耗时换算整套摄像头处理速度；采集、画面显示等也占用时间。
- 实时窗口已实际打开并检查，中文与快捷键提示可见。`启动USB识别.cmd --help` 启动入口检查通过。

首次尝试在线安装大型 CUDA 包后，改用已经验证的本机依赖副本；当前环境已校验通过。`outputs/debug_logs/windows_live_setup.log` 保留了早期中断安装的历史信息，不代表当前环境失败。

## 现在需要注意的实际效果

本次现场画面是复杂室内背景、近距离手持物品，部分区域明显过曝，与训练时木板上的俯拍照片差别较大。实际观察到背景误报和手持纸杯未检出，不能把“程序已接通”理解为现场准确率已达到验证集水平。

用户选择暂时保留当前演示，待方便时再调节拍摄场景。下一次建议：固定相机朝向木板，视野约一张 A4 纸，先测试单个完整物品，再测试多个；照明均匀，避免物体和木板过曝。先在接近训练/考核的场景下评价效果，再决定是否补拍真实摄像头照片。

独立跟踪现已接入。2026-11-08 已接通通信协议与中文助手，无下位机时用本机模拟通道验证，详见 [串口通信说明](serial-communication-20261108.md)。后续继续接入标定测速。没有因此重训模型、改写数据集或更改摄像头的 Windows 隐私权限。

## 重建环境

本机已完成配置，日常识别不需要再次安装依赖或启动 Docker。

如果以后换电脑或需要重建环境，可从项目目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

该脚本在项目 `.venv` 中安装 PyTorch CUDA 12.8 与 `requirements-live.txt`，并验证 CUDA 是否可用。参考：[PyTorch 官方安装版本](https://pytorch.org/get-started/previous-versions/)、[Ultralytics 预测参数](https://docs.ultralytics.com/modes/predict/)。
