# Number Plate Detection & Recognition (ANPR) — YOLOv8 + OpenCV + Tesseract (Design)

Date: 2026-04-08  
Owner: Ankit  
Scope: v1 “working pipeline” for offline video input; real-time capable depending on hardware.

## Goals

- Process a video frame-by-frame with OpenCV.
- Detect number plates per frame using an Ultralytics YOLOv8 model.
- Extract plate ROI crops and run OCR using Tesseract to obtain alphanumeric text.
- Visualize detections (bounding boxes + recognized text) on the output frames.
- Persist results for analysis (CSV in v1; optional annotated video output).

## Non-goals (v1)

- Cross-frame tracking / deduplication of plate reads (no SORT/ByteTrack in v1).
- Country-specific plate formatting rules beyond basic cleanup (regex/whitelist only).
- Multi-camera ingestion, streaming input, or a web UI.
- Automated model training pipeline (assumes you provide weights).

## Assumptions

- Windows environment with Python 3.10+.
- Tesseract is installed locally (native app), accessible via PATH or configured in code.
- You have a YOLOv8 plate detector weight file (`.pt`) trained to detect `license_plate` (or similar single-class).

## High-level Architecture

Single-process pipeline with small, testable modules:

- `run_anpr.py` (CLI entry)
  - Parses args, wires components, runs `VideoRunner`.
- `src/anpr/video_runner.py`
  - Opens input video, iterates frames, optionally displays (`cv2.imshow`) and/or writes annotated video, and emits structured events to a logger.
- `src/anpr/detector.py`
  - Wraps Ultralytics inference; returns a list of plate detections per frame: bbox + score (+ class id/name).
- `src/anpr/ocr.py`
  - ROI preprocessing, Tesseract invocation, text normalization and simple filtering.
- `src/anpr/logging.py`
  - CSV writer for per-detection events.

## Data Flow (per frame)

1. Read frame from `cv2.VideoCapture`.
2. Plate detection:
   - Run YOLOv8 inference on the frame (`imgsz` configurable).
   - Filter detections by confidence threshold.
3. For each detected plate bbox:
   - Apply padding to bbox and clamp to frame bounds.
   - Crop ROI from the original frame.
   - Preprocess ROI for OCR (OpenCV):
     - Convert to grayscale.
     - Resize (upscale to improve OCR on small plates).
     - Denoise (median/bilateral).
     - Binarize (adaptive threshold / Otsu).
     - Optional: morphological close to connect strokes.
   - OCR:
     - Tesseract with a plate-friendly configuration:
       - `--oem 3`
       - `--psm 7` (single text line) or `--psm 8` (single word) as a toggle.
       - Character whitelist: `A-Z0-9` (configurable).
     - Parse `image_to_data` to get text + confidence when possible.
   - Normalize:
     - Uppercase.
     - Remove spaces and non-alphanumerics.
     - Drop obviously invalid reads (too short/too long by configurable thresholds).
4. Output:
   - Draw bbox and label (best text) onto frame.
   - Log one CSV row per detection per frame.
   - Write annotated frame to output video (optional).
   - Display live window (optional).

## Output Formats

### Annotated video (optional)

- Same resolution and FPS as input (default).
- MP4 output using a broadly-supported codec (platform-dependent; default to `mp4v` with a configurable fallback).
- Overlay includes:
  - bbox rectangle
  - `text` (if available) + detection confidence

### CSV log (default in v1)

One row per detected plate per frame:

- `frame_idx` (int)
- `timestamp_ms` (int, derived from `CAP_PROP_POS_MSEC` when available; else `frame_idx / fps * 1000`)
- `x1,y1,x2,y2` (ints)
- `det_conf` (float)
- `ocr_text` (string, cleaned)
- `ocr_conf` (float or empty if unavailable)

## CLI (v1)

`python run_anpr.py --video <path> --weights <path> [options]`

Key options:

- `--conf <float>`: detector confidence threshold (default ~0.25).
- `--imgsz <int>`: YOLO inference size (default 640).
- `--device <cpu|0|1|...>`: inference device.
- `--show / --no-show`: display window.
- `--save-video <path>`: write annotated output mp4.
- `--save-csv <path>`: write CSV log.
- `--save-crops <dir>`: save ROI crops + text as filename (optional).
- `--frame-skip <int>`: process every Nth frame (performance knob).
- `--max-plates-per-frame <int>`: cap OCR work per frame (performance knob).
- `--tesseract-cmd <path>`: explicit Tesseract executable path.

## Error Handling

- If input video can’t be opened: exit with clear message.
- If YOLO weights missing/unreadable: exit with clear message.
- If Tesseract not found:
  - Fail fast with actionable guidance (install + set `--tesseract-cmd` or PATH).
- If OCR fails on a crop: return empty text; still draw bbox.

## Performance Considerations

- OCR is the main bottleneck; provide `--max-plates-per-frame` or short-circuit if too many detections.
- Optional `--frame-skip` to improve throughput.
- Keep preprocessing lightweight by default; expose toggles for heavier steps.

## Privacy / Safety Notes

- This system can be used for surveillance; ensure usage complies with local laws and policies.
- Prefer storing only necessary data; disable saving crops unless explicitly needed.

## Success Criteria (v1)

- Runs end-to-end on a local video file.
- Shows bounding boxes and OCR text overlay in a live preview window.
- Produces a CSV log of detections.
- Achieves usable throughput on a typical laptop GPU/CPU (with knobs to tune).

