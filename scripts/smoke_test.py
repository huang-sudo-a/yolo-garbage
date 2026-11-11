"""YOLO11n 环境冒烟测试:GPU 检查 + 权重加载 + 1 epoch 最小训练。

对应《环境准备任务书》第 7 节验证项 3/4/5。

用法(容器内,工作目录 /workspace):
    python scripts/smoke_test.py                       # 默认参数
    python scripts/smoke_test.py --epochs 3 --imgsz 416
"""
from __future__ import annotations

import argparse
from pathlib import Path

WORKSPACE = Path("/workspace")
LOCAL_WEIGHTS = WORKSPACE / "models" / "yolo11n.pt"
LOCAL_COCO8_YAML = WORKSPACE / "data" / "coco8" / "coco8.yaml"


def check_torch() -> None:
    """验证 PyTorch 与 CUDA(任务书 7.3)。"""
    import torch

    print("=" * 60)
    print("1. PyTorch / CUDA 检查")
    print("=" * 60)
    print(f"torch version       : {torch.__version__}")
    print(f"torch CUDA version  : {torch.version.cuda}")
    cuda_ok = torch.cuda.is_available()
    print(f"cuda available      : {cuda_ok}")
    assert cuda_ok, "CUDA 不可用,请检查容器 GPU 配置(Docker Desktop WSL2 + NVIDIA 驱动)"
    print(f"device name         : {torch.cuda.get_device_name(0)}")
    print(f"device capability   : {torch.cuda.get_device_capability(0)}")
    print(f"device count        : {torch.cuda.device_count()}")


def load_model(weights: str):
    """加载 YOLO11n 官方预训练权重(任务书 7.4)。"""
    import ultralytics
    from ultralytics import YOLO

    print()
    print("=" * 60)
    print("2. Ultralytics / YOLO11n 权重加载")
    print("=" * 60)
    print(f"ultralytics version : {ultralytics.__version__}")
    print(f"weights             : {weights}")
    model = YOLO(weights)
    print(f"task                : {model.task}")
    print(f"num classes (nc)    : {model.model.nc}")
    return model


def smoke_train(model, data: str, epochs: int, imgsz: int, batch: int, device: int) -> Path:
    """1 epoch 最小训练冒烟测试(任务书 7.5),输出到 runs/smoke。"""
    print()
    print("=" * 60)
    print("3. 1-epoch 冒烟训练")
    print("=" * 60)
    print(f"data                : {data}")
    print(f"epochs / imgsz / batch / device : {epochs} / {imgsz} / {batch} / {device}")
    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="/workspace/runs/smoke",
        name="train",
        exist_ok=True,
        verbose=True,
    )
    save_dir = Path(results.save_dir)
    print(f"save_dir            : {save_dir}")
    last_pt = save_dir / "weights" / "last.pt"
    assert last_pt.exists(), f"训练产物 {last_pt} 未生成,冒烟训练未正常完成"
    print(f"smoke train PASSED  : {last_pt} 已生成")
    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11n 环境冒烟测试")
    parser.add_argument("--weights", default=str(LOCAL_WEIGHTS),
                        help="权重路径;本地不存在时回退为 yolo11n.pt 自动下载")
    parser.add_argument("--data", default=str(LOCAL_COCO8_YAML),
                        help="数据集 yaml;本地不存在时回退为 coco8.yaml 自动下载")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    weights = args.weights
    if weights == str(LOCAL_WEIGHTS) and not LOCAL_WEIGHTS.exists():
        print(f"[warn] 本地权重 {LOCAL_WEIGHTS} 不存在,回退为 yolo11n.pt 并尝试自动下载")
        weights = "yolo11n.pt"

    data = args.data
    if data == str(LOCAL_COCO8_YAML) and not LOCAL_COCO8_YAML.exists():
        print(f"[warn] 本地数据配置 {LOCAL_COCO8_YAML} 不存在,回退为 coco8.yaml 并尝试自动下载")
        data = "coco8.yaml"

    check_torch()
    model = load_model(weights)
    smoke_train(model, data, args.epochs, args.imgsz, args.batch, args.device)

    print()
    print("=" * 60)
    print("冒烟测试全部通过 ✔")
    print("=" * 60)


if __name__ == "__main__":
    main()
