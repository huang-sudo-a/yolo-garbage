"""Capture one chessboard view and save a planar pixel-to-cm mapping."""

import argparse
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from invc_vision.calibration import PlaneCalibration
from invc_vision.app import open_capture


def main() -> None:
    parser = argparse.ArgumentParser(description="棋盘格平面标定；按空格拍摄，Q 退出")
    parser.add_argument("--source", default="0")
    parser.add_argument("--columns", type=int, default=9, help="横向内角点数")
    parser.add_argument("--rows", type=int, default=6, help="纵向内角点数")
    parser.add_argument("--square-cm", type=float, required=True, help="每个方格真实边长，单位 cm")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration/plane.json"))
    args = parser.parse_args()
    capture = open_capture(args.source, args.width, args.height)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("未能读取棋盘格画面")
            cv2.putText(frame, "Place board on object plane. SPACE=capture Q=quit", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow("Plane calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                return
            if key == 32:
                try:
                    calibration, preview = PlaneCalibration.from_chessboard(frame, args.columns, args.rows, args.square_cm)
                except RuntimeError as exc:
                    print(exc)
                    continue
                calibration.save(args.output)
                cv2.imshow("Detected corners", preview)
                cv2.waitKey(1000)
                print(f"标定已保存：{args.output.resolve()}")
                return
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
