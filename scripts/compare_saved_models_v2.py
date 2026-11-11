"""Compare saved v1/v2 models on one common validation set and operating point."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix


ROOT = Path("/workspace")
DATA = ROOT / "data/self_dataset_v2"
OUTPUT = ROOT / "outputs/self_dataset/comparison_v1_v2"
EXAMPLES = ["IMG_20261102_183123.jpg", "IMG_20261102_183258.jpg"]


def counts(matrix: np.ndarray) -> dict:
    correct = int(np.trace(matrix[:-1, :-1]))
    extra = int(matrix[:-1, -1].sum())
    missed = int(matrix[-1, :-1].sum())
    wrong = int(matrix[:-1, :-1].sum()) - correct
    return {
        "ground_truth_objects": int(matrix[:, :-1].sum()), "correct": correct,
        "extra_unmatched_boxes": extra, "wrong_class": wrong,
        "missed_unmatched_objects": missed,
        "precision_micro": correct / max(correct + extra + wrong, 1),
        "recall_micro": correct / max(correct + missed + wrong, 1),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def main() -> None:
    OUTPUT.mkdir(exist_ok=False)
    models = {
        "v1": ROOT / "outputs/self_dataset/debug_20261102_01/weights/best.pt",
        "v2": ROOT / "outputs/self_dataset/v2_30ep/weights/best.pt",
    }
    summary = {
        "dataset": str(DATA), "validation_images": 69,
        "scope": "59 preserved old validation images plus 10 new held-out images; neither model trained on these exact files. Old split's scene similarity remains a limitation.",
        "training_difference": "V1 trained 10 epochs on 235 images, V2 trained 30 epochs on 272 images; gains cannot be attributed solely to added data.",
        "operating_point": {"conf": 0.5, "nms_iou": 0.5, "agnostic_nms": True, "rect": False, "imgsz": 640, "matching_iou": 0.5},
        "models": {},
    }
    canvas = Image.new("RGB", (1440, 1100), "#f3f3f3")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(ROOT / ".ultralytics/Ultralytics/Arial.Unicode.ttf"), 16)
    colors = ["#e6194b", "#079337", "#2366ce", "#db7900", "#961ba5", "#008080"]
    for column, (tag, path) in enumerate(models.items()):
        model = YOLO(str(path))
        metrics = model.val(
            data=str(DATA / "data.yaml"), device=0, imgsz=640, batch=16, workers=4,
            plots=False, verbose=False, project=str(OUTPUT), name=f"{tag}_validation", exist_ok=False,
        )
        matrices = {key: np.zeros((7, 7)) for key in ("all", "old_batch", "new_batch")}
        image_counts = {key: 0 for key in matrices}
        predictions = []
        for result in model.predict(
            source=str(DATA / "images/val"), stream=True, device=0, imgsz=640,
            rect=False, batch=8, conf=0.5, iou=0.5, agnostic_nms=True, save=False, verbose=False,
        ):
            filename = Path(result.path).name
            rows = [list(map(float, line.split())) for line in (DATA / "labels/val" / f"{Path(filename).stem}.txt").read_text().splitlines() if line.strip()]
            gt = torch.tensor(rows, dtype=torch.float32, device=result.boxes.data.device).reshape(-1, 5)
            height, width = result.orig_shape
            boxes = torch.empty((len(gt), 4), device=gt.device)
            boxes[:, 0] = (gt[:, 1] - gt[:, 3] / 2) * width
            boxes[:, 1] = (gt[:, 2] - gt[:, 4] / 2) * height
            boxes[:, 2] = (gt[:, 1] + gt[:, 3] / 2) * width
            boxes[:, 3] = (gt[:, 2] + gt[:, 4] / 2) * height
            cm = ConfusionMatrix(names=model.names)
            cm.process_batch({"cls": result.boxes.cls, "conf": result.boxes.conf, "bboxes": result.boxes.xyxy}, {"cls": gt[:, 0], "bboxes": boxes}, conf=0.5, iou_thres=0.5)
            batch = "new_batch" if filename.startswith("IMG_20261102") else "old_batch"
            for key in ("all", batch):
                matrices[key] += cm.matrix
                image_counts[key] += 1
            detections = [{"class_id": int(box.cls.item()), "confidence": float(box.conf.item()), "xyxy": box.xyxy[0].tolist()} for box in result.boxes]
            predictions.append({"image": filename, "detections": detections})
            if filename in EXAMPLES:
                row = EXAMPLES.index(filename)
                image = Image.fromarray(result.orig_img[:, :, ::-1])
                image.thumbnail((700, 480))
                ox = column * 720 + (720 - image.width) // 2
                oy = row * 550 + 60 + (480 - image.height) // 2
                canvas.paste(image, (ox, oy))
                draw.text((column * 720 + 10, row * 550 + 8), f"{tag.upper()} | {filename}", font=font, fill="black")
                for box in detections:
                    c = box["class_id"]
                    x1, y1, x2, y2 = box["xyxy"]
                    x1, x2 = ox+x1/width*image.width, ox+x2/width*image.width
                    y1, y2 = oy+y1/height*image.height, oy+y2/height*image.height
                    draw.rectangle((x1, y1, x2, y2), outline=colors[c], width=2)
                    label = f"{model.names[c]} {box['confidence']:.2f}"
                    draw.text((x1, max(oy, y1-18)), label, font=font, fill=colors[c], stroke_width=1, stroke_fill="white")
        record = {
            "weights": str(path),
            "standard_validation_metrics": {k: float(v) for k, v in metrics.results_dict.items()},
            "per_class_map50_95": {model.names[c]: float(value) for c, value in enumerate(metrics.box.maps)},
            "operating_point_results": {key: {"images": image_counts[key], **counts(matrix)} for key, matrix in matrices.items()},
        }
        assert image_counts == {"all": 69, "old_batch": 59, "new_batch": 10}
        summary["models"][tag] = record
        (OUTPUT / f"{tag}_predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUTPUT / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"model": tag, "validation": record["standard_validation_metrics"], "all": record["operating_point_results"]["all"], "new_batch": record["operating_point_results"]["new_batch"]}, ensure_ascii=False), flush=True)
    canvas.save(OUTPUT / "before_after_examples.jpg", quality=92)
    print(f"Comparison saved: {OUTPUT}")


if __name__ == "__main__":
    main()
