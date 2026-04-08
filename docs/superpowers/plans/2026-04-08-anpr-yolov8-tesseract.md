# ANPR (YOLOv8 + OpenCV + Tesseract) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Python CLI that takes a video, detects license plates with YOLOv8, OCRs plate crops with Tesseract, overlays results, and saves a CSV (and optionally an annotated MP4).

**Architecture:** A small `src/anpr/` package with separate modules for video I/O, detection, OCR, and CSV logging. A top-level `run_anpr.py` wires components and exposes flags for performance and outputs.

**Tech Stack:** Python 3.10+, OpenCV (`opencv-python`), Ultralytics YOLOv8 (`ultralytics`), Tesseract OCR (`pytesseract` + native Tesseract install), NumPy, PyTest.

---

## File Structure (to create)

- Create: `requirements.txt` — Python dependencies
- Create: `README.md` — setup + run instructions (Windows-focused)
- Create: `run_anpr.py` — CLI entrypoint
- Create: `src/anpr/__init__.py`
- Create: `src/anpr/types.py` — small dataclasses for detections/log rows
- Create: `src/anpr/detector.py` — YOLO wrapper
- Create: `src/anpr/ocr.py` — preprocessing + pytesseract wrapper
- Create: `src/anpr/logging.py` — CSV logging helper
- Create: `src/anpr/video_runner.py` — frame loop + overlay + writing video
- Create: `tests/test_ocr_normalize.py` — unit tests for normalization logic
- Create: `tests/test_bbox.py` — unit tests for bbox padding/clamping

## Conventions

- Absolute paths accepted by CLI; outputs created if parent dirs exist.
- Default outputs:
  - CSV enabled by default: `outputs/anpr.csv`
  - Show window disabled by default (works headless); enable with `--show`
- Error messages should be actionable (especially for missing Tesseract).

---

### Task 1: Bootstrap project layout + deps

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `src/anpr/__init__.py`
- Create: `run_anpr.py` (stub)

- [ ] **Step 1: Add `requirements.txt`**

Contents:

```txt
ultralytics
opencv-python
numpy
pytesseract
pytest
```

- [ ] **Step 2: Add `README.md` with Windows setup**

Include:
- Python venv creation
- `pip install -r requirements.txt`
- Tesseract installation note (Windows installer) and how to pass `--tesseract-cmd`
- Example run command

- [ ] **Step 3: Create package init**

`src/anpr/__init__.py` minimal (empty or version string).

- [ ] **Step 4: Create CLI stub that prints help**

`run_anpr.py` should parse `--video` and `--weights` and print “not implemented yet” if run.

- [ ] **Step 5: Sanity-check imports**

Run:
- `python -c "import cv2, numpy, pytesseract; print('ok')"`
Expected: prints `ok`

---

### Task 2: Core types + bbox utilities (TDD)

**Files:**
- Create: `src/anpr/types.py`
- Create: `tests/test_bbox.py`

- [ ] **Step 1: Write failing tests for bbox padding/clamping**

```python
from anpr.video_runner import pad_and_clip_bbox

def test_pad_and_clip_bbox_clamps_to_frame():
    x1, y1, x2, y2 = pad_and_clip_bbox((10, 10, 20, 20), pad=0.5, w=25, h=25)
    assert 0 <= x1 < x2 <= 25
    assert 0 <= y1 < y2 <= 25

def test_pad_and_clip_bbox_handles_negative_and_overflow():
    x1, y1, x2, y2 = pad_and_clip_bbox((-10, -10, 40, 40), pad=0.0, w=30, h=30)
    assert (x1, y1, x2, y2) == (0, 0, 30, 30)
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest -q`
Expected: FAIL (module/function not found)

- [ ] **Step 3: Implement minimal bbox helper**

Create `pad_and_clip_bbox` in `src/anpr/video_runner.py` (or a dedicated util file if preferred), then re-run tests.

- [ ] **Step 4: Add `src/anpr/types.py` dataclasses**

Define:
- `PlateDetection(bbox: tuple[int,int,int,int], conf: float)`
- `PlateRead(frame_idx: int, timestamp_ms: int, bbox: tuple[int,int,int,int], det_conf: float, ocr_text: str, ocr_conf: float | None)`

- [ ] **Step 5: Re-run tests**

Run: `pytest -q`
Expected: PASS

---

### Task 3: OCR module (preprocess + normalize) (TDD where feasible)

**Files:**
- Create: `src/anpr/ocr.py`
- Create: `tests/test_ocr_normalize.py`

- [ ] **Step 1: Write failing tests for text normalization**

```python
from anpr.ocr import normalize_plate_text

def test_normalize_plate_text_basic():
    assert normalize_plate_text(" mh 12 ab 1234 ") == "MH12AB1234"

def test_normalize_plate_text_strips_non_alnum():
    assert normalize_plate_text("AB@12-34") == "AB1234"

def test_normalize_plate_text_empty():
    assert normalize_plate_text("") == ""
```

- [ ] **Step 2: Run tests to verify failures**

Run: `pytest -q`
Expected: FAIL (function not found)

- [ ] **Step 3: Implement normalization + simple validity filter**

In `src/anpr/ocr.py` implement:
- `normalize_plate_text(text: str) -> str`
- optional `is_plausible_plate(text: str, min_len: int = 4, max_len: int = 12) -> bool`

- [ ] **Step 4: Implement preprocessing function**

Implement:
- `preprocess_plate_roi(bgr_roi: np.ndarray) -> np.ndarray` returning a single-channel image suitable for OCR.

- [ ] **Step 5: Implement OCR wrapper**

Implement:
- `ocr_plate(bgr_roi: np.ndarray, tesseract_cmd: str | None, psm: int = 7, whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") -> tuple[str, float | None]`
Use `pytesseract.image_to_data(..., output_type=Output.DICT)` when possible.

- [ ] **Step 6: Re-run tests**

Run: `pytest -q`
Expected: PASS (normalization tests)

---

### Task 4: YOLOv8 detector module

**Files:**
- Create: `src/anpr/detector.py`

- [ ] **Step 1: Implement `YoloPlateDetector`**

API:
- `__init__(weights_path: str, conf: float, imgsz: int, device: str)`
- `detect(frame_bgr: np.ndarray) -> list[PlateDetection]`

Notes:
- Convert model outputs to integer pixel bboxes.
- Sort by confidence desc.

- [ ] **Step 2: Add a quick smoke command**

Add to `run_anpr.py` a `--dry-run-detect` mode that loads the model and runs on the first frame only, printing number of detections.

---

### Task 5: CSV logger

**Files:**
- Create: `src/anpr/logging.py`

- [ ] **Step 1: Implement `CsvPlateLogger`**

API:
- `__init__(csv_path: str)`
- `write_header()`
- `log(read: PlateRead)`
- `close()`

Ensure parent directory exists or error clearly.

---

### Task 6: Video runner (frame loop, overlay, save video)

**Files:**
- Create: `src/anpr/video_runner.py`
- Modify: `run_anpr.py`

- [ ] **Step 1: Implement `VideoRunner` skeleton**

Inputs:
- `video_path`
- detector
- ocr
- logger
- options: `show`, `save_video_path`, `frame_skip`, `max_plates_per_frame`

- [ ] **Step 2: Implement timestamp extraction**

Use `CAP_PROP_POS_MSEC` when available; else compute from FPS.

- [ ] **Step 3: Implement overlay**

Draw:
- bbox rectangle
- text label (empty text allowed)

- [ ] **Step 4: Implement video writer (optional)**

Default codec: `mp4v`; create writer using input width/height/fps.

- [ ] **Step 5: Add graceful quit**

If `--show` enabled, allow quitting with `q` / ESC.

---

### Task 7: CLI wiring + end-to-end run

**Files:**
- Modify: `run_anpr.py`
- Modify: `README.md`

- [ ] **Step 1: Implement full CLI args**

Required:
- `--video`
- `--weights`

Optional:
- `--conf`, `--imgsz`, `--device`
- `--show`
- `--save-video`
- `--save-csv` (default `outputs/anpr.csv`)
- `--frame-skip` (default 0)
- `--max-plates-per-frame` (default e.g. 5)
- `--tesseract-cmd`

- [ ] **Step 2: Run a real video**

Run (example):
- `python run_anpr.py --video "input.mp4" --weights "plate.pt" --show --save-csv outputs/anpr.csv`
Expected:
- Window shows detections
- CSV gets populated

- [ ] **Step 3: Update README with final usage examples**

---

## Plan Self-Review (against spec)

- Spec coverage: goals, modules, per-frame flow, outputs, error handling, performance knobs are all mapped to tasks.
- Placeholder scan: no TODO/TBD; each task includes concrete APIs and commands.
- Type consistency: `PlateDetection` and `PlateRead` used across detector/logger/runner consistently.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-08-anpr-yolov8-tesseract.md`. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration  
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints  

Which approach?

