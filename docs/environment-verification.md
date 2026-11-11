# 环境验证报告

- **验证日期**:2026-09-28
- **验证人**:Claude Code(自动化环境准备)
- **机器**:Windows 11 企业版 Build 26100 / Intel Core Ultra 7 265KF / 31.6GB 内存 / RTX 5060 Ti 8GB

## 1. Docker Desktop 服务端连接(任务书 7.1)✅

```
Client: Version 29.4.3, windows/amd64
Server: Docker Desktop 4.74.0 (227015), Engine 29.4.3, linux/amd64
```

- WSL2 backend 确认:`docker-desktop` 发行版 Running,内核 `6.6.114.1-microsoft-standard-WSL2`,OSType: linux
- 原有镜像/容器(ROS 仿真等)未做任何删除或修改

## 2. 容器内 nvidia-smi(任务书 7.2)✅

```
NVIDIA-SMI 575.64.04  Driver Version: 577.00  CUDA Version: 12.9
GPU 0: NVIDIA GeForce RTX 5060 Ti, 8151MiB VRAM, 180W
```

## 3. PyTorch / CUDA 验证(任务书 7.3)✅

```
torch version       : 2.8.0+cu128
torch CUDA version  : 12.8
cuda available      : True
device name         : NVIDIA GeForce RTX 5060 Ti
device capability   : (12, 0)          # Blackwell 架构
device count        : 1
```

## 4. YOLO11n 官方预训练权重加载(任务书 7.4)✅

```
ultralytics version : 8.4.164
weights             : /workspace/models/yolo11n.pt(官方 GitHub Release v8.3.0,5.6MB)
task                : detect
num classes (nc)    : 80
Transferred 499/499 items from pretrained weights
```

## 5. 1-epoch 冒烟训练(任务书 7.5)✅

- 数据:`data/coco8`(官方极小示例,4 train / 4 val),本地预下载,无需容器内联网
- 参数:`epochs=1 imgsz=320 batch=8 device=0`(CUDA:0,GPU 显存占用 0.229G)
- 结果:

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss
  1/1     0.229G      1.292      2.541      1.421
val:  mAP50 = 0.68,  mAP50-95 = 0.399
```

- 产物:`runs/smoke/train/weights/{best.pt, last.pt}`(named volume `invc-yolo-runs`),labels.jpg、results 等正常生成
- loss 与验证流程均无报错,训练确认使用 GPU

## 6. 版本与镜像记录(任务书 7.6)

| 项目 | 值 |
|---|---|
| 基础镜像 | `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` |
| 基础镜像摘要 | `sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385`(12.1GB) |
| 项目镜像 | `invc-yolo11-train:latest` |
| 项目镜像摘要 | `sha256:e30643de694d7bc453fd00ffadf39383924e32199c63ae59301a897664494aea`(磁盘占用 13.9GB / 内容 4.72GB) |
| Python | 3.11.13(/opt/conda,基础镜像自带;任务书允许) |
| pip | 25.1.1 |
| PyTorch / CUDA | 2.8.0+cu128 / 12.8 |
| torchvision / torchaudio | 0.23.0+cu128 / 2.8.0+cu128 |
| Ultralytics | 8.4.164 |
| 其他直接依赖 | opencv-python-headless 5.0.0.93,albumentations 2.0.8,tensorboard 2.21.0,PyYAML 6.0.2,pandas 3.0.6,matplotlib 3.11.2,pyserial 3.5 |
| GPU | NVIDIA GeForce RTX 5060 Ti,8GB,驱动 577.00 |
| 完整依赖清单 | `docs/requirements-freeze.txt`(150 个包) |

### 磁盘空间

| 项目 | 准备前 | 准备后 | 新增 |
|---|---|---|---|
| D 盘可用 | 243GB | 233GB | **约 10GB**(基础镜像展开层) |

Docker 数据仍全部位于 `D:\RobotMaster\DockerDesktopWSL`,未迁移。

## 7. 遇到的问题与说明

1. **旧 Docker 镜像源已全部失效**:机器上配置的中科大/网易/百度三个 registry mirror 均已停服(域名无法解析),导致 `docker pull` 直接报错。**未修改 Docker Desktop 配置**(避免重启影响正在运行的 ROS 容器),改用显式镜像前缀 `docker.m.daocloud.io/` 拉取后重新打官方标签。建议后续有空时把 Docker Desktop 的 registry-mirrors 换成可用源(DaoCloud `https://docker.m.daocloud.io`、1Panel `https://docker.1panel.live` 实测可用)。
2. **容器内无法访问 GitHub**:冒烟训练时 ultralytics 尝试下载 AMP 检查用 yolo26n.pt 失败(连接拒绝),自动跳过(amp 保持默认 True),不影响训练结果。yolo11n.pt 与 coco8 均已通过主机侧镜像代理预下载,后续正式训练无需容器联网。
3. **绑定挂载读写速度警告**:训练时提示 `Slow image access detected`(读 ~28MB/s)。这是 9p 挂载的正常表现;正式训练时应按任务书要求把数据集放到 named volume 或 WSL Linux 文件系统。
4. **Ultralytics 配置目录**:容器内 `/root/.config` 不可写,已在 compose.yaml 中设置 `YOLO_CONFIG_DIR=/workspace/.ultralytics` 持久化。

## 8. 验收标准对照(任务书第 8 节)

- [x] Docker Desktop 已启动且能运行 Linux 容器
- [x] 容器中 `torch.cuda.is_available()` 为 `True`
- [x] 容器中能识别 NVIDIA GeForce RTX 5060 Ti
- [x] `YOLO("yolo11n.pt")` 能加载官方预训练权重
- [x] 1 epoch 冒烟训练使用 GPU 并正常结束
- [x] 项目目录结构、Dockerfile、Compose、依赖锁定文件和环境验证报告均已提交到目标目录
- [x] 未影响原有 Docker 镜像、ROS/仿真环境或系统配置(仅新增本项目的镜像/卷,未删除任何原有资源)

## 9. 后续训练启动命令

```bash
cd D:\invc\work\invc-yolo-garbage
docker compose build                                   # 镜像重建(如 requirements.txt 变更)
docker compose run --rm train python scripts/smoke_test.py   # 冒烟复验
docker compose run --rm train yolo detect train \
    data=configs/data.yaml model=models/yolo11n.pt \
    epochs=100 imgsz=640 batch=16 device=0              # 正式微调(数据集就绪后)
```
