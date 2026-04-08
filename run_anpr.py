from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python run_anpr.py` without PYTHONPATH
_ROOT = Path(__file__).resolve().parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "ANPR: license-plate detection + OCR only. "
            "For people and vehicles (Human / Vehicle), use run_detect_objects.py instead."
        )
    )
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--weights", required=True, help="Path to YOLOv8 .pt weights")
    p.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    p.add_argument("--device", default="cpu", help="Device: cpu, 0, 1, ...")
    p.add_argument("--show", action="store_true", help="Display live annotated video window")
    p.add_argument("--save-video", default=None, help="Path to save annotated mp4")
    p.add_argument("--save-csv", default="outputs/anpr.csv", help="Path to save CSV log")
    p.add_argument("--no-csv", action="store_true", help="Do not write CSV log")
    p.add_argument("--save-crops", default=None, help="Directory to save cropped plate images")
    p.add_argument("--frame-skip", type=int, default=0, help="Process every Nth frame (0 = all frames)")
    p.add_argument(
        "--max-plates-per-frame",
        type=int,
        default=5,
        help="Max plates to OCR per frame (performance knob)",
    )
    p.add_argument("--tesseract-cmd", default=None, help="Path to tesseract.exe")
    p.add_argument("--dry-run-detect", action="store_true", help="Run detector on first frame only")
    p.add_argument(
        "--bbox-pad",
        type=float,
        default=0.12,
        help="Fractional padding around each detection box before OCR",
    )
    p.add_argument("--ocr-psm", type=int, default=7, help="Tesseract PSM mode (7=line, 8=word)")
    return p


def _check_tesseract(tesseract_cmd: str | None) -> None:
    import pytesseract

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    pytesseract.get_tesseract_version()


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.isfile(args.video):
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.weights):
        print(f"Error: weights not found: {args.weights}", file=sys.stderr)
        return 1

    from anpr.detector import YoloPlateDetector
    import cv2

    if args.dry_run_detect:
        det = YoloPlateDetector(args.weights, args.conf, args.imgsz, args.device)
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: cannot open video: {args.video}", file=sys.stderr)
            return 1
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print("Error: could not read first frame", file=sys.stderr)
            return 1
        dets = det.detect(frame)
        print(f"dry-run-detect: {len(dets)} plate(s) on first frame")
        return 0

    if not args.no_csv:
        try:
            _check_tesseract(args.tesseract_cmd)
        except Exception as e:
            print(
                "Error: Tesseract OCR not available. Install Tesseract for Windows and/or pass "
                "`--tesseract-cmd` to `tesseract.exe`.\n"
                f"Details: {e}",
                file=sys.stderr,
            )
            return 1

    from anpr.logging import CsvPlateLogger
    from anpr.video_runner import VideoRunner

    det = YoloPlateDetector(args.weights, args.conf, args.imgsz, args.device)

    logger: CsvPlateLogger | None = None
    if not args.no_csv:
        logger = CsvPlateLogger(args.save_csv)
        logger.open()
        logger.write_header()

    runner = VideoRunner(
        video_path=args.video,
        detector=det,
        tesseract_cmd=args.tesseract_cmd,
        logger=logger,
        show=args.show,
        save_video_path=args.save_video,
        save_crops_dir=args.save_crops,
        frame_skip=args.frame_skip,
        max_plates_per_frame=args.max_plates_per_frame,
        bbox_pad=args.bbox_pad,
        ocr_psm=args.ocr_psm,
    )
    try:
        runner.run()
    finally:
        if logger:
            logger.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
