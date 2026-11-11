"""Build image manifests without changing the downloaded dataset.

Removes byte-identical train/val overlaps and moves complete numbered frame
groups containing Bone into validation, so Bone has validation examples.
"""

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import re

import yaml


PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "data" / "garbage_datasets" / "datasets"
OUT = PROJECT / "data" / "practice_split"
CONFIG = PROJECT / "configs" / "garbage40_practice_clean.yaml"
SOURCE_CONFIG = PROJECT / "configs" / "garbage40.yaml"
CONTAINER_DATASET = Path("/workspace/data/garbage_datasets/datasets")
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(split: str) -> list[Path]:
    directory = DATASET / "images" / split
    return sorted((p for p in directory.iterdir() if p.suffix.lower() in SUFFIXES), key=lambda p: p.name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_group(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) // 50 if match else None


def has_class(path: Path, class_id: int = 7) -> bool:
    label = DATASET / "labels" / "train" / f"{path.stem}.txt"
    return any(line.split() and line.split()[0] == str(class_id) for line in label.read_text(encoding="utf-8").splitlines())


def container_path(path: Path) -> str:
    return str(CONTAINER_DATASET / path.relative_to(DATASET)).replace("\\", "/")


def main() -> None:
    train = image_files("train")
    val = image_files("val")
    by_name = {p.name: p for p in train}
    overlap = [p for p in val if p.name in by_name and sha256(p) == sha256(by_name[p.name])]
    overlap_names = {p.name for p in overlap}
    val = [p for p in val if p.name not in overlap_names]

    groups: dict[int, list[Path]] = defaultdict(list)
    for path in train:
        group = frame_group(path)
        if group is not None:
            groups[group].append(path)
    bone_per_group = Counter({group: sum(has_class(p) for p in images) for group, images in groups.items()})
    selected_groups: set[int] = set()
    bone_images = 0
    for group, amount in bone_per_group.most_common():
        if amount == 0 or bone_images >= 30:
            break
        selected_groups.add(group)
        bone_images += amount
    if bone_images < 30:
        raise RuntimeError(f"训练集找不到足够的 Bone 图片供验证：{bone_images}")

    moved = [p for p in train if frame_group(p) in selected_groups]
    remaining = [p for p in train if frame_group(p) not in selected_groups]
    val.extend(moved)
    train_set = {container_path(p) for p in remaining}
    val_set = {container_path(p) for p in val}
    if train_set & val_set:
        raise RuntimeError("清单中仍有交叉图片")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.txt").write_text("\n".join(sorted(train_set)) + "\n", encoding="utf-8")
    (OUT / "val.txt").write_text("\n".join(sorted(val_set)) + "\n", encoding="utf-8")
    names = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))["names"]
    clean_config = {
        "path": str(CONTAINER_DATASET).replace("\\", "/"),
        "train": "/workspace/data/practice_split/train.txt",
        "val": "/workspace/data/practice_split/val.txt",
        "names": names,
    }
    CONFIG.write_text(yaml.safe_dump(clean_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"排除 train/val 完全重复图片：{len(overlap)} 张")
    print(f"划到验证集的 Bone 图片：{bone_images} 张，涉及 {len(selected_groups)} 个连续编号组")
    print(f"新清单：train={len(train_set)}，val={len(val_set)}")
    print(f"训练配置：{CONFIG}")


if __name__ == "__main__":
    main()
