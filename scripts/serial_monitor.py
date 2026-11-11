"""CRC-checked incremental serial receiver and localhost test listener."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from invc_vision.serial_receiver import SerialReceiver


def main() -> int:
    parser = argparse.ArgumentParser(description="INVC 通信协议 V2 接收与校验")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--port", help="接收端 COM 口，如 COM6")
    group.add_argument("--listen", type=int, help="仅监听本机 TCP 端口，供无下位机模拟")
    parser.add_argument("--baudrate", type=int, default=460800)
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/serial" / datetime.now().strftime("monitor_%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    def show(kind, payload):
        if kind == "frame":
            print(f"通信帧={payload['s']} 识别帧={payload['q']} 目标={len(payload['d'])} FPS={payload['f']} CRC通过 来源={payload['src']}", flush=True)
        else:
            print(payload, flush=True)

    receiver = SerialReceiver(args.output, show, listen_port=args.listen, port=args.port, baudrate=args.baudrate)
    receiver.start()
    started = time.monotonic()
    try:
        while not args.seconds or time.monotonic()-started < args.seconds:
            time.sleep(0.1)
            if not receiver._thread.is_alive():
                break
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
    print(json.dumps(receiver.snapshot(), ensure_ascii=False, indent=2))
    return 1 if receiver.error and not receiver.stats.received else 0


if __name__ == "__main__":
    raise SystemExit(main())
