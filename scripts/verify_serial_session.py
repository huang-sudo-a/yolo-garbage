"""Compare each assistant reception with the exact detector record that produced it."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from invc_vision.protocol import FrameDecoder, encode_wire_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--min-frames", type=int, default=30)
    args = parser.parse_args()
    received = []
    for log in args.session.glob("receive_*/received.jsonl"):
        received += [json.loads(line)["payload"] for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    sources = {}
    for pattern in ("camera_*.jsonl", "replay_*.jsonl"):
        for log in args.session.glob(pattern):
            records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
            for record in records:
                sources[(record.get("data_source", "live_inference"), record["seq"])] = record
    mismatches = []
    for payload in received:
        key = (payload["src"], payload["q"])
        record = sources.get(key)
        if record is None:
            mismatches.append({"seq": payload["q"], "reason": "no source"})
            continue
        expected = FrameDecoder().feed(encode_wire_frame(record, payload["sid"], payload["s"]))[0]
        if expected != payload:
            mismatches.append({"seq": payload["q"], "reason": "field mismatch"})
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in args.session.glob("receive_*/summary.json")]
    result = {"received": len(received), "source_types": sorted({p["src"] for p in received}),
              "every_received_field_matches_detector_log": not mismatches,
              "mismatches": mismatches, "crc_errors": sum(s["bad_crc"] for s in summaries),
              "format_errors": sum(s["bad_format"] for s in summaries),
              "wire_sequence_gaps": sum(s["missing"] for s in summaries),
              "note": "Local TCP software validation; does not validate a physical UART, baud rate, wiring or lower controller."}
    result["passed"] = len(received) >= args.min_frames and not mismatches and not any(result[k] for k in ("crc_errors", "format_errors", "wire_sequence_gaps"))
    args.session.joinpath("verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
