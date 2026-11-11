# INVC YOLO11n 自采数据训练项目

本项目在 Windows 的 `D:\invc\work\invc-yolo-garbage`，Docker 训练容器中对应 `/workspace`。原始标注仍保存在 `D:\invc\work\labels`，不会被准备脚本修改。

2026-11-02 第二版已完成：合并新增 47 张照片，整理为 `data/self_dataset_v2/`（272 train / 69 val），训练 30 轮，模型保存在 `outputs/self_dataset/v2_30ep/weights/best.pt`。在相同 69 张验证图上，mAP@0.5 从第一版的 97.97% 提升到第二版的 99.21%。查看 [第二版完整操作说明](docs/training-v2-walkthrough.md)，了解运行了哪个程序、命令含义及所有输出位置。第一版数据和结果保留。

## 目录

```text
D:\invc\work\
├─ labels\                         原始图片与同名 YOLO .txt 标注（不改动）
├─ dataset\name.txt                六个类别，顺序即类别编号
├─ prepare_self_dataset.py         可直接运行的数据准备入口
└─ invc-yolo-garbage\
   ├─ compose.yaml                 Docker 训练环境；项目挂载为 /workspace
   ├─ docker\Dockerfile            PyTorch/CUDA/Ultralytics 镜像
   ├─ models\yolo11n.pt            YOLO11n 预训练权重
   ├─ scripts\prepare_self_dataset.py  配对、校验、8:2 划分、生成 data.yaml
   ├─ scripts\train_self.py        自采数据训练入口
   ├─ data\self_dataset\            已整理的数据
   │  ├─ images\train\ 和 images\val\
   │  ├─ labels\train\ 和 labels\val\
   │  ├─ data.yaml
   │  └─ split_summary.json
   ├─ outputs\                     训练结果
   └─ src\invc_vision\              后续摄像头跟踪、测速代码
```

旧 40 类练习数据和旧训练结果仍留在 `D:\invc\test\invc-yolo-garbage`，没有复制到这个新项目。新项目保留了旧的练习脚本和配置供参考；新数据请使用 `prepare_self_dataset.py` 与 `train_self.py`。

## 准备数据

已经运行过一次，得到 235 张训练图、59 张验证图，原始文件未变。若以后补标了新图片，请用 `--output` 指定一个新目录，避免覆盖原划分。例如：

```powershell
py -3.13 D:\invc\work\prepare_self_dataset.py --output D:\invc\work\invc-yolo-garbage\data\self_dataset_v2
```

输出 YAML 中的 `path` 使用容器路径。注意：脚本按固定随机种子进行图片级划分；如果大量照片是同一场景连续拍摄的近似帧，正式评估应按拍摄场景或物体分组划分，以免验证成绩偏高。

## 检查与训练

2026-11-02 已完成首次自采数据调试：YOLO11n 在 RTX 5060 Ti 上成功训练 10 轮，使用 235 张训练图和 59 张验证图，验证 mAP@0.5 为 97.892%。权重保存在 `outputs/self_dataset/debug_20261102_01/weights/best.pt`。这是现有随机划分的验证结果，尚未验证新场景或摄像头效果；土豆训练图只有 22 张，需要补拍。完整记录见 [首次训练调试记录](docs/first-training-debug-20261102.md)。

打开 Docker Desktop，然后在 PowerShell 执行：

```powershell
cd D:\invc\work\invc-yolo-garbage
docker compose run --rm train python scripts/train_self.py --check-only
docker compose run --rm train python scripts/train_self.py --epochs 1 --name smoke
```

确认一轮训练和验证都成功后，再决定正式训练轮数，例如：

```powershell
docker compose run --rm train python scripts/train_self.py --epochs 100 --name full_v1
```

结果会保存在 `outputs\self_dataset\`。每次运行用新的 `--name`，以免与已有实验目录冲突。`--rm` 只删除临时容器，不删除镜像、D 盘数据或训练结果。

## 摄像头程序

Windows USB 识别已接通。双击项目根目录的 `启动USB识别.cmd`，默认使用第二版模型、GPU 0 和六类映射 `configs/self_categories.yaml`；按 Q 退出，S 截图。运行日志和截图保存在 `outputs/live/`。详情见 [USB 摄像头识别说明](docs/windows-usb-live.md)。

2026-11-03 已接入独立卡尔曼滤波和匈牙利匹配，默认显示目标 ID、平滑框和最近 15 次观测的轨迹；不调用 YOLO 内置跟踪。19 项程序测试通过，第二版模型在受控双目标视频上连续 120 帧无 ID 切换，另完成 90 帧真实摄像头运行检查。实际多目标移动效果仍需在木板场景验收。详情见 [跟踪实现与验证](docs/tracking-20261103.md)。使用 `--no-tracking` 可切回纯检测；标定测速待后续接入。Windows 本机运行时无需启动 Docker。

2026-11-08 已接入通信协议 V2、后台发送和中文接收助手。双击根目录 `启动串口助手.cmd`，选择“启动摄像头识别”或“回放历史识别结果”。无下位机时使用本机模拟通道；以后可切实际 COM。28 项程序测试通过，真实摄像头到助手接收 212 帧逐帧核对一致，CRC 错误和通信序号缺口均为 0。速度仍为空，物理串口待硬件验证。详见 [串口通信与助手使用说明](docs/serial-communication-20261108.md)。
