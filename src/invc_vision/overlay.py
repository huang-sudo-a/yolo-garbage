"""Render Chinese labels with one Pillow conversion per displayed frame."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class DetectionOverlay:
    def __init__(self, project: Path) -> None:
        candidates = [Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf"), project / ".ultralytics/Ultralytics/Arial.Unicode.ttf"]
        font_path = next((p for p in candidates if p.is_file()), None)
        if font_path is None:
            raise FileNotFoundError("找不到中文字体，请安装微软雅黑或提供 Arial.Unicode.ttf")
        self.font = ImageFont.truetype(str(font_path), 22)
        self.small = ImageFont.truetype(str(font_path), 18)
        self.colors = [(240, 90, 80), (70, 210, 130), (80, 160, 255), (255, 190, 60), (210, 120, 240), (60, 220, 220)]

    def draw(self, frame, detections: list[dict], fps: float, inference_ms: float, device: str, tracking: bool = False):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for detection in detections:
            x, y, w, h = detection["bbox_xywh_px"]
            color_index = detection.get("track_id") if detection.get("track_id") is not None else detection["class_id"]
            color = self.colors[color_index % len(self.colors)]
            trail = detection.get("trail_px", [])
            if len(trail) > 1:
                draw.line([tuple(point) for point in trail], fill=color, width=2)
            draw.rectangle((x, y, x+w, y+h), outline=color, width=3)
            text = f"{detection['item']}  {detection['category']}  {detection['confidence']:.0%}"
            if detection.get("track_id") is not None:
                status = "?" if detection.get("track_state") == "tentative" else ""
                text = f"ID {detection['track_id']}{status}  " + text
            box = draw.textbbox((0, 0), text, font=self.font)
            tw, th = box[2]-box[0]+10, box[3]-box[1]+8
            left = max(0, min(x, width-tw))
            top = max(44, min(y-th, height-th-30))
            draw.rectangle((left, top, left+tw, top+th), fill=(20, 25, 32))
            draw.text((left+5, top+3-box[1]), text, font=self.font, fill=color)
        draw.rectangle((0, 0, width, 42), fill=(20, 25, 32))
        title = "检测+跟踪" if tracking else "实时检测"
        draw.text((12, 6), f"{title}  |  FPS {fps:.1f}  |  推理 {inference_ms:.1f} ms  |  目标 {len(detections)}", font=self.font, fill="white")
        draw.rectangle((0, height-30, width, height), fill=(20, 25, 32))
        mode = "卡尔曼+匈牙利" if tracking else "纯检测"
        draw.text((12, height-27), f"Q 退出   S 保存截图   |   {'CPU' if device == 'cpu' else 'GPU '+device}   |   {mode}", font=self.small, fill=(200, 210, 220))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
