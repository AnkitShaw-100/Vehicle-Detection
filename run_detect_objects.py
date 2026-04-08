"""
Multi-class video detection: Human (person), Car, Truck, Ambulance (if in model), etc.
Uses Ultralytics YOLOv8. Default weights `yolov8n.pt` are downloaded on first run (COCO classes).
Standard COCO does not include "ambulance" — use custom .pt trained with that class.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Detect humans and vehicle types in a video file (OpenCV reads the file you pass with "
            "--video). No webcam: input is always your video path."
        )
    )
    p.add_argument(
        "--video",
        required=True,
        help="Path to your input video file (e.g. .mp4). Processing runs on this file only.",
    )
    p.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="YOLOv8 weights: local .pt path or hub name (e.g. yolov8n.pt downloads automatically)",
    )
    p.add_argument(
        "--labels",
        default="human,vehicle",
        help=(
            "Comma-separated. Use `human` for people, `vehicle` for cars/trucks/buses/motorcycles "
            "(shown as Vehicle). Or list `car,truck` separately for distinct labels. "
            "`ambulance` only if your .pt includes that class."
        ),
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.38,
        help="Confidence threshold (higher = fewer false boxes)",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Inference size")
    p.add_argument("--device", default="cpu", help="cpu or 0, 1, …")
    p.add_argument(
        "--show",
        action="store_true",
        help="Open a preview window while the video file is processed (not a camera feed)",
    )
    p.add_argument("--save-video", default=None, help="Save annotated MP4")
    p.add_argument("--save-csv", default="outputs/detections.csv", help="CSV path")
    p.add_argument("--no-csv", action="store_true", help="Do not write CSV")
    p.add_argument("--frame-skip", type=int, default=0, help="Skip frames (0 = all)")
    return p


def _weights_ok(path_or_name: str) -> bool:
    if os.path.isfile(path_or_name):
        return True
    # Ultralytics can download bare hub names like yolov8n.pt (no path separators).
    norm = path_or_name.replace("\\", "/")
    if "/" not in norm and path_or_name.endswith(".pt"):
        return True
    return False


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.isfile(args.video):
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        return 1
    if not _weights_ok(args.weights):
        print(
            f"Error: weights not found: {args.weights} (use yolov8n.pt or a valid .pt path)",
            file=sys.stderr,
        )
        return 1

    from anpr.object_logging import CsvObjectLogger
    from anpr.object_video_runner import ObjectVideoRunner
    from anpr.yolo_objects import YoloObjectDetector

    try:
        det = YoloObjectDetector(
            args.weights,
            args.conf,
            args.imgsz,
            args.device,
            args.labels,
        )
    except Exception as e:
        print(f"Error loading model or classes: {e}", file=sys.stderr)
        return 1

    for w in det.warnings:
        print(f"Note: {w}", file=sys.stderr)

    logger: CsvObjectLogger | None = None
    if not args.no_csv:
        logger = CsvObjectLogger(args.save_csv)
        logger.open()
        logger.write_header()

    runner = ObjectVideoRunner(
        video_path=args.video,
        detector=det,
        logger=logger,
        show=args.show,
        save_video_path=args.save_video,
        frame_skip=args.frame_skip,
    )
    try:
        runner.run()
    finally:
        if logger:
            logger.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
