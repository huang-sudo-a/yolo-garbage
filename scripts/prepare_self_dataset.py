"""Copy paired YOLO annotations into a reproducible train/validation split.

The original labeled files are never moved or modified.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT.parent / "labels"
DEFAULT_NAMES = PROJECT.parent / "dataset" / "name.txt"
DEFAULT_OUTPUT = PROJECT / "data" / "self_dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare paired YOLO images and labels for training")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--names", type=Path, default=DEFAULT_NAMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_dataset(source: Path, names_file: Path) -> tuple[list[str], list[tuple[Path, Path, set[int]]]]:
    if not source.is_dir():
        raise FileNotFoundError(f"Image/label source not found: {source}")
    if not names_file.is_file():
        raise FileNotFoundError(f"Class names file not found: {names_file}")
    names = [line.strip() for line in names_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("Class names must be nonempty and unique")

    images = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) < 2:
        raise ValueError("At least two images are required for train/validation splitting")
    if len({image.stem.lower() for image in images}) != len(images):
        raise ValueError("Images with duplicate stems would overwrite each other")

    pairs = []
    for image in images:
        label = source / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing YOLO label for {image.name}")
        present_classes: set[int] = set()
        for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label.name}:{line_number}: expected class x y w h")
            try:
                class_id = int(fields[0])
                box = [float(value) for value in fields[1:]]
            except ValueError as error:
                raise ValueError(f"{label.name}:{line_number}: invalid number") from error
            if class_id < 0 or class_id >= len(names) or not all(0 <= value <= 1 for value in box) or box[2] <= 0 or box[3] <= 0:
                raise ValueError(f"{label.name}:{line_number}: invalid class or normalized box")
            present_classes.add(class_id)
        pairs.append((image, label, present_classes))
    return names, pairs


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    project = PROJECT.resolve()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    try:
        relative_output = output.relative_to(project)
    except ValueError as error:
        raise ValueError("Output must be inside the project so Docker can access it") from error
    if output.exists():
        raise FileExistsError(f"Output already exists; originals were not changed: {output}")
    if output == source or output.is_relative_to(source):
        raise ValueError("Output cannot be inside the source folder")

    names, pairs = read_dataset(source, args.names.resolve())
    random.Random(args.seed).shuffle(pairs)
    validation_count = max(1, min(len(pairs) - 1, round(len(pairs) * args.val_ratio)))
    val_pairs = pairs[:validation_count]
    train_pairs = pairs[validation_count:]
    train_classes = set().union(*(pair[2] for pair in train_pairs))
    val_classes = set().union(*(pair[2] for pair in val_pairs))
    all_classes = set().union(*(pair[2] for pair in pairs))
    if train_classes != all_classes or val_classes != all_classes:
        raise ValueError("One or more classes are missing from a split; retry with another --seed")

    for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=False)
        label_dir.mkdir(parents=True, exist_ok=False)
        for image, label, _ in split_pairs:
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(label, label_dir / label.name)

    data_yaml = output / "data.yaml"
    container_path = "/workspace/" + relative_output.as_posix()
    data_yaml.write_text(
        f"path: {container_path}\ntrain: images/train\nval: images/val\nnames: "
        + json.dumps(names, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "source": str(source),
        "train_images": len(train_pairs),
        "val_images": len(val_pairs),
        "seed": args.seed,
        "classes": names,
        "train_images_per_class": dict(sorted(Counter(class_id for _, _, classes in train_pairs for class_id in classes).items())),
        "val_images_per_class": dict(sorted(Counter(class_id for _, _, classes in val_pairs for class_id in classes).items())),
    }
    (output / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(train_pairs)} train and {len(val_pairs)} validation images in {output}")
    for class_id, name in enumerate(names):
        print(f"  {class_id} {name}: train={summary['train_images_per_class'].get(class_id, 0)}, val={summary['val_images_per_class'].get(class_id, 0)}")
    print(f"Docker dataset config: {container_path}/data.yaml")


if __name__ == "__main__":
    main()
