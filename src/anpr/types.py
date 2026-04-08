from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectDetection:
    """YOLO detection after class filtering (one row per box)."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    conf: float
    class_id: int
    label: str  # e.g. Human, Car, Truck, Vehicle
    class_name_raw: str  # as in model.names, e.g. person
