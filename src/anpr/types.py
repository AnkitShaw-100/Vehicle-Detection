from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlateDetection:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    conf: float


@dataclass(frozen=True)
class PlateRead:
    frame_idx: int
    timestamp_ms: int
    bbox: tuple[int, int, int, int]
    det_conf: float
    ocr_text: str
    ocr_conf: float | None


@dataclass(frozen=True)
class ObjectDetection:
    """YOLO detection after class filtering (one row per box)."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    conf: float
    class_id: int
    label: str  # e.g. Human, Car, Truck, Ambulance
    class_name_raw: str  # as in model.names, e.g. person
