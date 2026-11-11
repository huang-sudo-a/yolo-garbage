"""Native Windows detection with independent Kalman/Hungarian tracking."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import time

import cv2
import torch
from ultralytics import YOLO

from .categories import CategoryMap
from .overlay import DetectionOverlay
from .telemetry import Telemetry
from .tracking import Detection, MultiObjectTracker


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT / "outputs/self_dataset/v2_30ep/weights/best.pt"
DEFAULT_CATEGORIES = PROJECT / "configs/self_categories.yaml"
WINDOW = "INVC USB Camera - Q: quit / S: snapshot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USB 摄像头六类垃圾检测与独立目标跟踪（Windows GPU）")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", default="0", help="摄像头编号或视频文件路径")
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--agnostic-nms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tracking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--track-max-lost", type=float, default=1.0, help="短时漏检保留轨迹的秒数")
    parser.add_argument("--track-min-hits", type=int, default=2, help="确认轨迹需要的检测命中次数")
    parser.add_argument("--track-distance-factor", type=float, default=1.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--capture-fps", type=float, default=30)
    parser.add_argument("--serial-port", help="可选数据发送端口，默认不连接")
    parser.add_argument("--baudrate", type=int, default=460800)
    parser.add_argument("--serial-hz", type=float, default=10, help="串口最大发送频率，默认每秒 10 帧，保持最新结果")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--record", type=Path, help="可选：保存带检测框的视频")
    parser.add_argument("--snapshot", type=Path, help="退出时保存最后一帧检测画面")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stop-file", type=Path, help="助手创建此文件时正常退出")
    return parser.parse_args()


def open_capture(source: str, width: int, height: int, fps: float = 30) -> cv2.VideoCapture:
    if not source.isdecimal():
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"无法打开视频文件：{source}")
        return capture
    backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF) if sys.platform == "win32" else (cv2.CAP_ANY,)
    for backend in backends:
        capture = cv2.VideoCapture(int(source), backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            for _ in range(10):
                ok, frame = capture.read()
                if ok and frame is not None and frame.size:
                    return capture
                time.sleep(0.03)
        capture.release()
    raise RuntimeError(f"摄像头 {source} 无法读取画面；检查编号、Windows 相机权限及是否被其他软件占用")


def save_frame(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", frame)
    if not ok:
        raise RuntimeError(f"无法编码截图：{path}")
    encoded.tofile(str(path))


def run(args: argparse.Namespace) -> int:
    if not args.model.is_file():
        raise FileNotFoundError(f"找不到模型权重：{args.model}")
    if not 0 < args.conf < 1 or not 0 < args.iou <= 1:
        raise ValueError("conf 和 iou 必须在合理的 0~1 范围内")
    if min(args.width, args.height, args.imgsz, args.capture_fps) <= 0 or args.max_frames < 0:
        raise ValueError("尺寸、帧率必须为正，max-frames 不能为负")
    if args.headless and args.source.isdecimal() and args.max_frames == 0:
        raise ValueError("无窗口摄像头模式必须指定 --max-frames")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请先运行 scripts/setup_windows.ps1 配置显卡环境")

    model = YOLO(str(args.model))
    categories = CategoryMap(args.categories, model.names)
    overlay = DetectionOverlay(PROJECT)
    tracker = MultiObjectTracker(args.track_max_lost, args.track_min_hits, args.track_distance_factor) if args.tracking else None
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    log_path = args.log or PROJECT / "outputs/live" / f"session_{now}.jsonl"
    capture = open_capture(args.source, args.width, args.height, args.capture_fps)
    writer = None
    last_display = None
    sequence = 0
    previous_start = None
    previous_media_time = None
    fps = 0.0
    is_camera = args.source.isdecimal()
    actual_width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = capture.get(cv2.CAP_PROP_FPS)
    device_name = torch.cuda.get_device_name(int(args.device)) if args.device.isdecimal() else args.device
    mode_text = "检测 + 独立卡尔曼/匈牙利跟踪（未启用测速）" if tracker else "纯检测"
    print(f"模型：{args.model}\n设备：{device_name}\n视频源：{args.source}，实际 {actual_width}x{actual_height}，报告帧率 {reported_fps:g}\n模式：{mode_text}\n日志：{log_path}\nQ 退出，S 保存截图", flush=True)
    try:
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, min(actual_width, 1280), min(actual_height, 800))
        with Telemetry(log_path, args.serial_port, args.baudrate, args.serial_hz) as telemetry:
            if args.serial_port:
                print(f"通信：{args.serial_port}，{args.baudrate} / 8N1，最多 {args.serial_hz:g} 帧/秒；断线自动重连", flush=True)
            while True:
                if args.stop_file and args.stop_file.exists():
                    break
                started = time.perf_counter()
                ok, frame = capture.read()
                if not ok or frame is None or not frame.size:
                    if is_camera:
                        raise RuntimeError("摄像头读帧失败或连接已断开")
                    break
                if is_camera:
                    tracking_time = time.monotonic()
                else:
                    media_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    interval = 1 / (reported_fps if reported_fps > 0 else args.capture_fps)
                    fallback = (sequence+1)*interval if previous_media_time is None else previous_media_time+interval
                    tracking_time = media_time if media_time > 0 and (previous_media_time is None or media_time > previous_media_time) else fallback
                    previous_media_time = tracking_time
                if previous_start is not None:
                    current_fps = 1 / max(started - previous_start, 1e-6)
                    fps = current_fps if fps == 0 else 0.85 * fps + 0.15 * current_fps
                previous_start = started
                result = model.predict(frame, conf=args.conf, iou=args.iou, agnostic_nms=args.agnostic_nms, imgsz=args.imgsz, rect=False, device=args.device, verbose=False)[0]
                detections = []
                if result.boxes is not None:
                    coords = result.boxes.xyxy.cpu().tolist()
                    classes = result.boxes.cls.int().cpu().tolist()
                    confidences = result.boxes.conf.cpu().tolist()
                    height, width = frame.shape[:2]
                    for xyxy, class_id, confidence in zip(coords, classes, confidences):
                        x1, y1, x2, y2 = [round(float(v)) for v in xyxy]
                        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
                        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        item = model.names[class_id]
                        detections.append({
                            "track_id": None, "class_id": class_id, "item": item,
                            "category": categories.category(item), "confidence": round(confidence, 4),
                            "bbox_xywh_px": [x1, y1, max(0, x2-x1), max(0, y2-y1)],
                            "speed_instant": None, "speed_avg_15": None, "speed_samples": 0, "speed_unit": None,
                        })
                tracking_started = time.perf_counter()
                if tracker:
                    observations = tracker.update([Detection(tuple(d["bbox_xywh_px"]), d["class_id"], d["confidence"]) for d in detections], tracking_time)
                    for detection, observation in zip(detections, observations):
                        detection["detector_bbox_xywh_px"] = detection["bbox_xywh_px"]
                        x, y, w, h = observation.bbox
                        left, top = max(0, min(frame.shape[1]-1, round(x))), max(0, min(frame.shape[0]-1, round(y)))
                        right, bottom = max(left+1, min(frame.shape[1], round(x+w))), max(top+1, min(frame.shape[0], round(y+h)))
                        detection.update(
                            track_id=observation.track_id,
                            track_state="confirmed" if observation.confirmed else "tentative",
                            track_hits=observation.hits,
                            bbox_xywh_px=[left, top, right-left, bottom-top],
                            trail_px=[[round(px), round(py)] for px, py in observation.trail],
                        )
                tracking_ms = round((time.perf_counter()-tracking_started)*1000, 3)
                sequence += 1
                inference_ms = round(float(result.speed.get("inference", 0)), 2)
                pipeline_ms = round(sum(result.speed.values()), 2)
                record = {
                    "version": 1, "mode": "detect_track" if tracker else "detect", "seq": sequence,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "fps": round(fps, 2), "inference_ms": inference_ms, "pipeline_ms": pipeline_ms,
                    "speed_unit": None, "frame_wh_px": [frame.shape[1], frame.shape[0]],
                    "model": str(args.model), "device": args.device,
                    "thresholds": {"conf": args.conf, "iou": args.iou, "agnostic_nms": args.agnostic_nms},
                    "tracking_ms": tracking_ms,
                    "tracking_time_s": tracking_time,
                    "tracker": {"algorithm": "kalman_hungarian", "retained_tracks": tracker.retained_tracks,
                                "max_lost_seconds": args.track_max_lost, "min_hits": args.track_min_hits,
                                "distance_factor": args.track_distance_factor} if tracker else None,
                    "detections": detections,
                    "serial": telemetry.status(),
                }
                telemetry.write(record)
                display = overlay.draw(frame, detections, fps, inference_ms, args.device, tracking=tracker is not None)
                last_display = display
                if args.record:
                    if writer is None:
                        args.record.parent.mkdir(parents=True, exist_ok=True)
                        output_fps = reported_fps if reported_fps > 0 else args.capture_fps
                        writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (frame.shape[1], frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError(f"无法创建录像：{args.record}")
                    writer.write(display)
                if not args.headless:
                    cv2.imshow(WINDOW, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key in (ord("s"), ord("S")):
                        path = log_path.parent / f"snapshot_{now}_{sequence}.jpg"
                        save_frame(path, display)
                        print(f"截图已保存：{path}", flush=True)
                if args.max_frames and sequence >= args.max_frames:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if args.snapshot and last_display is not None:
            save_frame(args.snapshot, last_display)
    if sequence == 0:
        raise RuntimeError("视频源没有提供任何可处理画面")
    print(f"已处理 {sequence} 帧；日志：{log_path}", flush=True)
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"启动或运行失败：{exc}", file=sys.stderr, flush=True)
        return 1
