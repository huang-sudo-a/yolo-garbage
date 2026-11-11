"""Train YOLO11n on the prepared self-labeled detection dataset inside Docker."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


WORKSPACE = Path("/workspace")
DEFAULT_DATA = WORKSPACE / "data" / "self_dataset" / "data.yaml"
DEFAULT_MODEL = WORKSPACE / "models" / "yolo11n.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11n on the self-labeled dataset")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="trial")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Pretrained model not found: {args.model}")
    data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(data["path"])
    for split in ("train", "val"):
        images = root / data[split]
        labels = root / "labels" / split
        if not images.is_dir() or not labels.is_dir():
            raise FileNotFoundError(f"Missing {split} images or labels: {images}, {labels}")
        image_files = [item for item in images.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES]
        missing = [item.name for item in image_files if not (labels / f"{item.stem}.txt").is_file()]
        if not image_files or missing:
            raise ValueError(f"{split}: {len(image_files)} images; missing labels: {missing[:5]}")
        print(f"{split}: {len(image_files)} images, {len(image_files) - len(missing)} matching labels")
    print(f"classes: {len(data['names'])}; model: {args.model}")
    if args.check_only:
        print("Dataset check passed; training was not started")
        return
    if args.epochs < 1 or args.batch < 1:
        raise ValueError("--epochs and --batch must be positive")
    model = YOLO(str(args.model))
    result = model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        project=str(WORKSPACE / "outputs" / "self_dataset"),
        name=args.name,
        exist_ok=False,
        amp=True,
        seed=42,
        plots=True,
    )
    print(f"Training results: {result.save_dir}")


if __name__ == "__main__":
    main()
