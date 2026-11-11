"""Launch the native Windows camera application from the project root."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invc_vision.app import main


if __name__ == "__main__":
    raise SystemExit(main())
