"""使用 YOLO11n 微调下载的 40 类垃圾检测数据集。

默认是快速试跑：5% 训练数据、1 epoch、640 分辨率、batch 8。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


WORKSPACE = Path("/workspace")
DEFAULT_DATA = WORKSPACE / "configs" / "garbage40.yaml"
DEFAULT_MODEL = WORKSPACE / "models" / "yolo11n.pt"
DEFAULT_PROJECT = WORKSPACE / "outputs" / "garbage40"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_dataset(data_yaml: Path) -> None:
    """在训练前检查 YAML、图片目录和同名标签是否存在。"""
    if not data_yaml.is_file():
        raise FileNotFoundError(f"找不到数据配置：{data_yaml}")

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data["path"])
    print(f"数据集根目录：{root}")
    print(f"类别数量：{len(data['names'])}")

    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"{split} 目录不完整：{image_dir} / {label_dir}"
            )

        images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        missing = [p.name for p in images if not (label_dir / f"{p.stem}.txt").is_file()]
        print(f"{split}: {len(images)} 张图片，缺少标签 {len(missing)} 张")
        if missing:
            raise RuntimeError(f"{split} 中存在缺少标签的图片，例如：{missing[:5]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11n 40 类垃圾数据集微调")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="trial")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction 必须在 0 到 1 之间")
    if not args.model.is_file():
        raise FileNotFoundError(f"找不到预训练权重：{args.model}")

    validate_dataset(args.data)
    if args.check_only:
        print("数据集目录检查通过，未启动训练。")
        return

    print(
        f"开始试跑：epochs={args.epochs}, fraction={args.fraction}, "
        f"imgsz={args.imgsz}, batch={args.batch}, device={args.device}"
    )
    model = YOLO(str(args.model))
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        fraction=args.fraction,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        pretrained=True,
        amp=True,
        cache=False,
        seed=42,
        deterministic=True,
        project=str(DEFAULT_PROJECT),
        name=args.name,
        exist_ok=False,
        plots=True,
    )
    print(f"训练完成，结果目录：{results.save_dir}")


if __name__ == "__main__":
    main()
