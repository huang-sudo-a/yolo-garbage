"""Audit original YOLO annotations and their prepared split without changing either."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT.parent / "labels"
DATASET = PROJECT / "data" / "self_dataset"
OUTPUT = PROJECT / "outputs" / "debug_logs"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    names = (PROJECT.parent / "dataset" / "name.txt").read_text(encoding="utf-8-sig").splitlines()
    images = sorted(SOURCE.glob("*.jpg"))
    issues: list[str] = []
    records = {}
    sizes: Counter[str] = Counter()
    duplicates: dict[str, str] = {}
    hashes: dict[str, str] = {}
    instances: Counter[int] = Counter()
    image_classes: Counter[int] = Counter()
    annotations = {}
    for image in images:
        with Image.open(image) as im:
            width, height = im.size
            sizes[f"{width}x{height}"] += 1
            if im.getexif().get(274, 1) != 1:
                issues.append(f"Unexpected EXIF orientation: {image.name}")
            im.verify()
        image_hash = digest(image)
        if image_hash in hashes:
            duplicates[image.name] = hashes[image_hash]
        hashes[image_hash] = image.name
        label = image.with_suffix(".txt")
        if not label.is_file():
            issues.append(f"Missing label: {image.name}")
            continue
        rows = []
        for number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError("expected five columns")
                c = int(fields[0])
                x, y, w, h = map(float, fields[1:])
                if not (0 <= c < len(names) and all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)) and w > 0 and h > 0):
                    raise ValueError("invalid class or coordinates")
                if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
                    raise ValueError("box crosses image boundary")
                rows.append((c, x, y, w, h))
                instances[c] += 1
            except ValueError as exc:
                issues.append(f"{label.name}:{number}: {exc}")
        classes = {row[0] for row in rows}
        image_classes.update(classes)
        annotations[image.name] = rows
        records[image.name] = {"image_sha256": image_hash, "label_sha256": digest(label)}
        original_json = PROJECT.parent / "raw" / f"{image.stem}.json"
        if original_json.is_file():
            original = json.loads(original_json.read_text(encoding="utf-8-sig"))
            original_names = Counter(s["label"] for s in original["shapes"] if s["shape_type"] == "rectangle")
            if original_names != Counter(names[row[0]] for row in rows):
                issues.append(f"Original JSON class counts disagree: {image.name}")

    stems = {p.stem for p in images}
    for label in SOURCE.glob("*.txt"):
        if label.stem not in stems:
            issues.append(f"Orphan label: {label.name}")
    splits = {}
    split_names = {}
    for split in ("train", "val"):
        counts: Counter[int] = Counter()
        files = sorted((DATASET / "images" / split).glob("*.jpg"))
        split_names[split] = {image.name for image in files}
        for image in files:
            if image.name not in records:
                issues.append(f"Prepared image absent from source: {image.name}")
                continue
            label = DATASET / "labels" / split / f"{image.stem}.txt"
            if digest(image) != records[image.name]["image_sha256"] or not label.is_file() or digest(label) != records[image.name]["label_sha256"]:
                issues.append(f"Prepared image or label differs: {split}/{image.name}")
            counts.update({row[0] for row in annotations[image.name]})
        splits[split] = {"images": len(files), "images_per_class": {name: counts[c] for c, name in enumerate(names)}}
    if split_names["train"] & split_names["val"]:
        issues.append("Same filenames appear in both splits")
    if split_names["train"] | split_names["val"] != set(records):
        issues.append("Split union does not equal original dataset")
    report = {
        "source": str(SOURCE), "dataset": str(DATASET), "images": len(images),
        "objects": sum(instances.values()), "classes": names,
        "background_images": [name for name, rows in annotations.items() if not rows],
        "images_per_class": {name: image_classes[c] for c, name in enumerate(names)},
        "objects_per_class": {name: instances[c] for c, name in enumerate(names)},
        "image_sizes": dict(sizes), "exact_duplicate_images": duplicates,
        "splits": splits, "issues": issues,
        "training_classes_below_30_images": [name for name, count in splits["train"]["images_per_class"].items() if count < 30],
        "split_limitation": "Existing seed-42 image-level random split; similar scenes may cross splits. No scene-level generalization claim.",
    }
    (OUTPUT / "dataset_audit_20261102.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    manifest = OUTPUT / "source_manifest_20261102.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != records:
            raise RuntimeError("Original dataset changed since the recorded audit; manifest was preserved")
    else:
        manifest.write_text(json.dumps(records, indent=2)+"\n", encoding="utf-8")

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    canvas = Image.new("RGB", (1440, 1600), "#f0f0f0")
    draw = ImageDraw.Draw(canvas)
    colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080"]
    selected = []
    for c in range(len(names)):
        candidates = [p for p in images if any(row[0] == c for row in annotations[p.name])]
        for p in (min(candidates, key=lambda p: len(annotations[p.name])), max(candidates, key=lambda p: len(annotations[p.name]))):
            selected.append((c, p))
    for index, (focus_class, path) in enumerate(selected):
        col, row = index % 3, index // 3
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((464, 344))
            offset_x = col*480+(480-im.width)//2
            offset_y = row*400+45+(344-im.height)//2
            canvas.paste(im, (offset_x, offset_y))
            for c, x, y, w, h in annotations[path.name]:
                left, top = offset_x+(x-w/2)*im.width, offset_y+(y-h/2)*im.height
                right, bottom = offset_x+(x+w/2)*im.width, offset_y+(y+h/2)*im.height
                draw.rectangle((left, top, right, bottom), outline=colors[c], width=2)
                draw.text((left, top), names[c], font=font, fill=colors[c], stroke_width=1, stroke_fill="white")
        draw.text((col*480+8, row*400+5), f"{names[focus_class]} / {path.stem}", fill="black", font=font)
    canvas.save(OUTPUT / "annotation_review_20261102.jpg", quality=90)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
