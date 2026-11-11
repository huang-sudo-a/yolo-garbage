"""Read-only dataset check: image/label matches and class frequencies."""

from collections import Counter
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT / "configs" / "garbage40.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    config = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    root = PROJECT / "data" / "garbage_datasets" / "datasets"
    names = {int(k): v for k, v in config["names"].items()}
    for split in ("train", "val"):
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        label_files = list(labels_dir.glob("*.txt"))
        image_stems = {p.stem for p in images}
        unmatched_images = [p.name for p in images if not (labels_dir / f"{p.stem}.txt").exists()]
        orphan_labels = [p.name for p in label_files if p.stem not in image_stems]
        classes: Counter[int] = Counter()
        bad = []
        for image in images:
            label = labels_dir / f"{image.stem}.txt"
            if not label.exists():
                continue
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                try:
                    if len(parts) != 5:
                        raise ValueError("应有 5 列")
                    class_id = int(parts[0])
                    values = [float(v) for v in parts[1:]]
                    if class_id not in names or any(not 0 <= v <= 1 for v in values) or values[2] <= 0 or values[3] <= 0:
                        raise ValueError("类别编号或归一化框无效")
                    classes[class_id] += 1
                except ValueError as exc:
                    bad.append(f"{label.name}:{line_number}: {exc}")
        print(f"\n{split}: 图片 {len(images)}, 标注文件 {len(label_files)}, 无标注图片 {len(unmatched_images)}, 孤立标注 {len(orphan_labels)}, 错误框 {len(bad)}")
        for class_id, name in names.items():
            print(f"  {class_id:2d} {name:20s} {classes[class_id]}")
        for title, rows in (("无标注图片", unmatched_images), ("孤立标注", orphan_labels), ("错误框", bad)):
            if rows:
                print(f"  {title}示例：{rows[:5]}")


if __name__ == "__main__":
    main()
