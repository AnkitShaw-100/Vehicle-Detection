# Video detection (two tools)

## 1) People & vehicles — `run_detect_objects.py` (recommended for your case)

**Input:** a **video file** you provide with `--video "path\to\file.mp4"`. There is **no live camera** in this script—detection runs on that file as fast as your PC allows (sequential frames from disk, same as typical “real-time speed” playback).

Detect and **label** objects in each frame: **Human** (COCO `person`) and **Vehicle** (groups **car, truck, bus, motorcycle** when you use `--labels human,vehicle`). This is **not** license-plate detection — use the ANPR script only for reading plate text.

**Do not use `run_anpr.py`** to classify people vs vehicles: it uses a **plate** detector + OCR and will label crops in ways that are wrong for “who is a person vs vehicle.”

- Default **`--labels human,vehicle`**: **Human** + **Vehicle** (car, truck, bus, motorcycle). Use `--labels human,car,truck` if you want separate **Car** / **Truck** labels instead of **Vehicle**.
- Default weights: **`yolov8n.pt`** (downloads automatically; trained on COCO).
- **COCO does not have an `ambulance` class.** For ambulances you need **custom weights** (`.pt`) trained with an `ambulance` label, then pass `--weights your_model.pt`.

```powershell
py -3.12 run_detect_objects.py --video "input.mp4" --show --save-csv "outputs/detections.csv"
```

Only certain classes (comma-separated):

```powershell
py -3.12 run_detect_objects.py --video "input.mp4" --labels "human,car,truck" --show
```

CSV columns: `frame_idx`, `timestamp_ms`, bbox, **`label`** (Human / Vehicle / Car / …), `class_name_raw`, `class_id`, `det_conf`.

---

## 2) ANPR (plates + OCR) — `run_anpr.py`

Detect number plates in a video using **YOLOv8 (Ultralytics)**, then read plate text using **Tesseract OCR**, frame-by-frame with **OpenCV**.

## Setup (Windows)

### 1) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install Python dependencies

```powershell
pip install -r requirements.txt
```

If you have multiple Python installs, use the same interpreter for running the app and tests (for example `py -3.12` on Windows).

### Tests

```powershell
py -3.12 -m pytest -q
```

### 3) Install Tesseract (required **only** for `run_anpr.py`, not for `run_detect_objects.py`)

- Install the Windows Tesseract app (commonly from the `tesseract-ocr` installer).
- Make sure `tesseract.exe` is either:
  - on your PATH, or
  - passed explicitly via `--tesseract-cmd "C:\Path\To\tesseract.exe"`

## Run

Basic (CSV to `outputs/anpr.csv`, no UI window):

```powershell
py -3.12 run_anpr.py --video "input.mp4" --weights "plate.pt"
```

Skip CSV (detector + OCR only on screen / video file):

```powershell
py -3.12 run_anpr.py --video "input.mp4" --weights "plate.pt" --no-csv --show
```

Show live window:

```powershell
python run_anpr.py --video "input.mp4" --weights "plate.pt" --show
```

Save annotated video + CSV:

```powershell
python run_anpr.py --video "input.mp4" --weights "plate.pt" --show --save-video "outputs/annotated.mp4" --save-csv "outputs/anpr.csv"
```

Dry-run detector (first frame only):

```powershell
python run_anpr.py --video "input.mp4" --weights "plate.pt" --dry-run-detect
```

## Notes

- For performance, try `--frame-skip 1` (skip one frame between processed frames, i.e. every 2nd frame) and/or smaller `--imgsz`.
- OCR is usually the bottleneck; `--max-plates-per-frame` caps OCR workload.

