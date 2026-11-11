"""Exercise real YOLO + tracking with controlled motion from an annotated photo.

This is a reproducible generated-video check, not a live-camera accuracy test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/tracking_validation_20261103")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(PROJECT)
    output.mkdir(exist_ok=False)
    source = PROJECT / "data/self_dataset_v2/images/val/IMG_20261102_183019.jpg"
    labels = PROJECT / "data/self_dataset_v2/labels/val/IMG_20261102_183019.txt"
    original = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width = original.shape[:2]
    scale = min(1280/width, 720/height)
    image = cv2.resize(original, (round(width*scale), round(height*scale)))
    ox, oy = (1280-image.shape[1])//2, (720-image.shape[0])//2
    base = np.full((720, 1280, 3), 110, dtype=np.uint8)
    base[oy:oy+image.shape[0], ox:ox+image.shape[1]] = image
    objects = []
    for line in labels.read_text().splitlines():
        c, x, y, w, h = map(float, line.split())
        objects.append({"class_id": int(c), "bbox": [ox+(x-w/2)*width*scale, oy+(y-h/2)*height*scale, w*width*scale, h*height*scale]})
    assert len(objects) == 2
    input_video = output / "controlled_motion.avi"
    writer = cv2.VideoWriter(str(input_video), cv2.VideoWriter_fourcc(*"MJPG"), 20, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError("Cannot create validation video")
    truth = []
    try:
        for frame in range(120):
            dx, dy = round(18*math.sin(frame*2*math.pi/120)), round(8*math.sin(frame*4*math.pi/120))
            transform = np.array([[1, 0, dx], [0, 1, dy]], dtype=float)
            writer.write(cv2.warpAffine(base, transform, (1280, 720), borderMode=cv2.BORDER_REFLECT_101))
            truth.append([{**obj, "bbox": [obj["bbox"][0]+dx, obj["bbox"][1]+dy, *obj["bbox"][2:]]} for obj in objects])
    finally:
        writer.release()
    (output / "expected_motion.json").write_text(json.dumps(truth), encoding="utf-8")
    command = [sys.executable, str(PROJECT / "scripts/run_live.py"), "--source", str(input_video), "--headless", "--max-frames", "120", "--log", str(output / "detections.jsonl"), "--snapshot", str(output / "preview.jpg"), "--record", str(output / "annotated.mp4")]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    (output / "run.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Video check failed; inspect {output / 'run.log'}")
    records = [json.loads(line) for line in (output / "detections.jsonl").read_text(encoding="utf-8").splitlines()]
    ids, switches, missing, previous = [set(), set()], [0, 0], [0, 0], [None, None]
    for expected, record in zip(truth, records):
        used = set()
        for index, obj in enumerate(expected):
            x, y, w, h = obj["bbox"]
            candidates = []
            for position, detection in enumerate(record["detections"]):
                if position in used or detection["class_id"] != obj["class_id"]:
                    continue
                a, b, c, d = detection["bbox_xywh_px"]
                distance = math.hypot(a+c/2-x-w/2, b+d/2-y-h/2)
                if distance < 40:
                    candidates.append((distance, position, detection["track_id"]))
            if not candidates:
                missing[index] += 1
                continue
            _, position, track_id = min(candidates)
            used.add(position)
            ids[index].add(track_id)
            if previous[index] is not None and previous[index] != track_id:
                switches[index] += 1
            previous[index] = track_id
    summary = {
        "kind": "generated_video_from_annotated_photo", "frames": len(records), "fps": 20,
        "model": str(PROJECT / "outputs/self_dataset/v2_30ep/weights/best.pt"),
        "identity_ids": [sorted(values) for values in ids], "id_switches": switches, "missing_frames": missing,
        "source_photo": str(source), "limitation": "Controlled global translation of two visible objects; not evidence of real-camera crossing or occlusion performance.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(records) != 120 or any(len(values) != 1 for values in ids) or sum(switches) or sum(missing):
        raise RuntimeError("Identity or detection check failed; see summary.json")


if __name__ == "__main__":
    main()
