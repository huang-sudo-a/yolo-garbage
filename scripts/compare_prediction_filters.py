"""Compare prediction filters on the existing validation split, without training."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix


def main() -> None:
    root = Path("/workspace")
    dataset = root / "data/self_dataset"
    weights = root / "outputs/self_dataset/debug_20261102_01/weights/best.pt"
    output = root / "outputs/self_dataset/debug_20261102_01/filter_comparison"
    output.mkdir(exist_ok=False)
    model = YOLO(str(weights))
    settings = [
        (0.25, 0.7, False),
        (0.5, 0.7, False),
        (0.7, 0.7, False),
        (0.9, 0.7, False),
        (0.5, 0.7, True),
        (0.5, 0.5, True),
        (0.7, 0.5, True),
    ]
    rows = []
    for conf, nms_iou, agnostic in settings:
        matrix = ConfusionMatrix(names=model.names)
        images = 0
        for result in model.predict(
            source=str(dataset / "images/val"), stream=True, device=0,
            imgsz=640, rect=False, conf=conf, iou=nms_iou,
            agnostic_nms=agnostic, batch=8, save=False, verbose=False,
        ):
            height, width = result.orig_shape
            label = dataset / "labels/val" / f"{Path(result.path).stem}.txt"
            gt = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
            gt = torch.tensor(gt, dtype=torch.float32, device=result.boxes.data.device).reshape(-1, 5)
            xyxy = torch.empty((len(gt), 4), dtype=torch.float32, device=gt.device)
            xyxy[:, 0] = (gt[:, 1] - gt[:, 3] / 2) * width
            xyxy[:, 1] = (gt[:, 2] - gt[:, 4] / 2) * height
            xyxy[:, 2] = (gt[:, 1] + gt[:, 3] / 2) * width
            xyxy[:, 3] = (gt[:, 2] + gt[:, 4] / 2) * height
            matrix.process_batch(
                {"cls": result.boxes.cls, "conf": result.boxes.conf, "bboxes": result.boxes.xyxy},
                {"cls": gt[:, 0], "bboxes": xyxy}, conf=conf, iou_thres=0.5,
            )
            images += 1
        m = matrix.matrix
        correct = int(np.trace(m[:-1, :-1]))
        extra = int(m[:-1, -1].sum())
        missed = int(m[-1, :-1].sum())
        wrong_class = int(m[:-1, :-1].sum()) - correct
        fp, fn = extra + wrong_class, missed + wrong_class
        precision = correct / max(correct + fp, 1)
        recall = correct / max(correct + fn, 1)
        row = {
            "conf": conf, "nms_iou": nms_iou, "agnostic_nms": agnostic,
            "images": images, "ground_truth_objects": correct + fn,
            "correct": correct, "extra_unmatched_boxes": extra,
            "wrong_class": wrong_class, "missed_unmatched_objects": missed,
            "false_positive_including_wrong_class": fp,
            "false_negative_including_wrong_class": fn,
            "precision_micro": round(precision, 6), "recall_micro": round(recall, 6),
            "f1_micro": round(2 * precision * recall / max(precision + recall, 1e-12), 6),
        }
        rows.append(row)
        tag = f"conf{conf}_iou{nms_iou}_agnostic{agnostic}"
        (output / f"{tag}.json").write_text(json.dumps(m.tolist()), encoding="utf-8")
        report = {
            "model": str(weights), "split": "val", "image_size": 640,
            "rect": False, "matching_iou": 0.5,
            "pipeline": "model.predict (single-label NMS), then official ConfusionMatrix matching",
            "limitation": "Threshold selection on the existing validation split, not independent testing. Different from training-validator multi-label NMS and its default 0.45 matching IoU.",
            "settings": rows,
        }
        (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
