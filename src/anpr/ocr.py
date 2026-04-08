from __future__ import annotations

import re

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def normalize_plate_text(text: str) -> str:
    s = text.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def is_plausible_plate(text: str, min_len: int = 4, max_len: int = 12) -> bool:
    if not text:
        return False
    return min_len <= len(text) <= max_len


def preprocess_plate_roi(bgr_roi: np.ndarray) -> np.ndarray:
    if bgr_roi.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 40 or w < 100:
        scale = max(40 / max(h, 1), 100 / max(w, 1), 1.0)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    return thr


def ocr_plate(
    bgr_roi: np.ndarray,
    tesseract_cmd: str | None,
    psm: int = 7,
    whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
) -> tuple[str, float | None]:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    proc = preprocess_plate_roi(bgr_roi)
    config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}"
    try:
        data = pytesseract.image_to_data(proc, config=config, output_type=Output.DICT)
    except pytesseract.TesseractNotFoundError:
        raise
    except Exception:
        return "", None

    texts: list[str] = []
    confs: list[float] = []
    for i, t in enumerate(data.get("text", [])):
        t = (t or "").strip()
        if not t:
            continue
        try:
            c = float(data["conf"][i])
        except (KeyError, IndexError, ValueError):
            c = -1.0
        if c < 0:
            continue
        texts.append(t)
        confs.append(c)

    if not texts:
        raw = pytesseract.image_to_string(proc, config=config).strip()
        norm = normalize_plate_text(raw)
        return norm, None

    joined = "".join(texts)
    norm = normalize_plate_text(joined)
    ocr_conf = sum(confs) / len(confs) if confs else None
    return norm, ocr_conf
