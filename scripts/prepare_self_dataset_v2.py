"""Build the second dataset snapshot while preserving the first split and all sources."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT.parent
V1 = PROJECT / "data/self_dataset"
OUT = PROJECT / "data/self_dataset_v2"
NEW_IMAGES = BASE / "数据集11.2"
NEW_LABELS = BASE / "labels2"

# Consecutive views of the same arrangement stay together. The selected groups
# cover potatoes, pale ceramic backs, mixed ceramic/bricks, and grey bricks.
HOLDOUT_GROUPS = {
    "potato_pair": ["183019", "183034"],
    "single_ceramic_back": ["183115", "183123"],
    "mixed_small_arrangement": ["183253", "183258", "183305"],
    "seven_bricks_arrangement": ["183638", "183646", "183648"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_pair(image: Path, label: Path, names: list[str], check_json: bool) -> dict:
    if not image.is_file() or not label.is_file():
        raise FileNotFoundError(f"Missing image or annotation: {image}, {label}")
    with Image.open(image) as im:
        width, height = im.size
        if im.getexif().get(274, 1) != 1:
            raise ValueError(f"Unexpected EXIF orientation: {image}")
        im.verify()
    rows = []
    for number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected five columns: {label}:{number}")
        c = int(fields[0])
        x, y, w, h = map(float, fields[1:])
        if not (0 <= c < len(names) and all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)) and w > 0 and h > 0):
            raise ValueError(f"Invalid class or coordinates: {label}:{number}")
        if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
            raise ValueError(f"Box outside image: {label}:{number}")
        rows.append([c, x, y, w, h])
    if check_json:
        annotation = json.loads(image.with_suffix(".json").read_text(encoding="utf-8-sig"))
        if (annotation["imageWidth"], annotation["imageHeight"]) != (width, height):
            raise ValueError(f"JSON dimensions differ: {image}")
        if len(annotation["shapes"]) != len(rows):
            raise ValueError(f"JSON object count differs: {image}")
        for shape, row in zip(annotation["shapes"], rows):
            if shape["shape_type"] != "rectangle" or shape["label"] != names[row[0]]:
                raise ValueError(f"JSON category/order differs: {image}")
            xs = [p[0] for p in shape["points"]]
            ys = [p[1] for p in shape["points"]]
            expected = [(min(xs)+max(xs))/2/width, (min(ys)+max(ys))/2/height, (max(xs)-min(xs))/width, (max(ys)-min(ys))/height]
            if max(abs(a-b) for a, b in zip(expected, row[1:])) > 2e-6:
                raise ValueError(f"JSON coordinates differ: {image}")
    record = {
        "image": str(image), "label": str(label), "filename": image.name,
        "image_sha256": sha256(image), "label_sha256": sha256(label),
        "classes": sorted({r[0] for r in rows}),
        "objects_per_class": dict(Counter(r[0] for r in rows)),
    }
    if check_json:
        record["json"] = str(image.with_suffix(".json"))
        record["json_sha256"] = sha256(image.with_suffix(".json"))
    return record


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {OUT}")
    names = (BASE / "dataset/name.txt").read_text(encoding="utf-8-sig").splitlines()
    original_manifest = json.loads((PROJECT / "outputs/debug_logs/source_manifest_20261102.json").read_text(encoding="utf-8"))
    records = []
    for split in ("train", "val"):
        for image in sorted((V1 / "images" / split).glob("*.jpg")):
            label = V1 / "labels" / split / f"{image.stem}.txt"
            record = inspect_pair(image, label, names, False)
            initial = original_manifest[image.name]
            if record["image_sha256"] != initial["image_sha256"] or record["label_sha256"] != initial["label_sha256"]:
                raise ValueError(f"First dataset snapshot changed: {image.name}")
            original_image = BASE / "labels" / image.name
            original_label = original_image.with_suffix(".txt")
            if sha256(original_image) != initial["image_sha256"] or sha256(original_label) != initial["label_sha256"]:
                raise ValueError(f"Original source changed: {image.name}")
            record.update(split=split, batch="20261028", group="inherited_v1_split")
            records.append(record)

    lookup = {f"IMG_20261102_{time}.jpg": group for group, times in HOLDOUT_GROUPS.items() for time in times}
    new_images = sorted(NEW_IMAGES.glob("*.jpg"))
    if not set(lookup).issubset({p.name for p in new_images}):
        raise ValueError("One or more selected scene-group images are missing")
    if {p.stem for p in new_images} != {p.stem for p in NEW_LABELS.glob("*.txt")}:
        raise ValueError("New image and label filenames do not pair exactly")
    for image in new_images:
        record = inspect_pair(image, NEW_LABELS / f"{image.stem}.txt", names, True)
        record.update(split="val" if image.name in lookup else "train", batch="20261102", group=lookup.get(image.name, "new_training_arrangements"))
        records.append(record)
    if len({r["filename"].lower() for r in records}) != len(records):
        raise ValueError("Duplicate filename across batches")
    if len({r["image_sha256"] for r in records}) != len(records):
        raise ValueError("Exact duplicate image across batches")

    splits = {}
    for split in ("train", "val"):
        selected = [r for r in records if r["split"] == split]
        counts = Counter(c for r in selected for c in r["classes"])
        if any(counts[c] == 0 for c in range(len(names))):
            raise ValueError(f"A class is absent from {split}")
        if split == "train" and any(counts[c] < 30 for c in range(len(names))):
            raise ValueError("A training class has fewer than 30 images")
        splits[split] = {
            "images": len(selected), "backgrounds": sum(not r["classes"] for r in selected),
            "old_batch_images": sum(r["batch"] == "20261028" for r in selected),
            "new_batch_images": sum(r["batch"] == "20261102" for r in selected),
            "images_per_class": {name: counts[c] for c, name in enumerate(names)},
        }
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=False)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=False)
    for record in records:
        image_out = OUT / "images" / record["split"] / record["filename"]
        label_out = OUT / "labels" / record["split"] / f"{Path(record['filename']).stem}.txt"
        shutil.copy2(record["image"], image_out)
        shutil.copy2(record["label"], label_out)
        if sha256(image_out) != record["image_sha256"] or sha256(label_out) != record["label_sha256"]:
            raise ValueError(f"Copied file failed checksum: {image_out}")
    (OUT / "data.yaml").write_text(
        "path: /workspace/data/self_dataset_v2\ntrain: images/train\nval: images/val\nnames: " + json.dumps(names, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "images": len(records), "new_images": len(new_images), "classes": names,
        "splits": splits, "new_validation_scene_groups": HOLDOUT_GROUPS,
        "policy": "Preserve v1 train/val assignments; hold out complete selected new arrangement groups before training. V1 scene leakage limitations remain inherited.",
        "new_holdout_limitation": "New validation batch covers ceramic, bricks, potato; it does not newly cover paper cups, bottles or pebbles.",
    }
    (OUT / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
