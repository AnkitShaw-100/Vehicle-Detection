from __future__ import annotations

import csv
import os

from anpr.types import ObjectDetection


class CsvObjectLogger:
    def __init__(self, csv_path: str) -> None:
        self._path = csv_path
        self._f = None
        self._writer: csv.DictWriter | None = None

    def open(self) -> None:
        parent = os.path.dirname(os.path.abspath(self._path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        self._f = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._f,
            fieldnames=[
                "frame_idx",
                "timestamp_ms",
                "x1",
                "y1",
                "x2",
                "y2",
                "label",
                "class_name_raw",
                "class_id",
                "det_conf",
            ],
        )

    def write_header(self) -> None:
        if self._writer:
            self._writer.writeheader()

    def log(
        self,
        frame_idx: int,
        timestamp_ms: int,
        det: ObjectDetection,
    ) -> None:
        if not self._writer:
            return
        x1, y1, x2, y2 = det.bbox
        self._writer.writerow(
            {
                "frame_idx": frame_idx,
                "timestamp_ms": timestamp_ms,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "label": det.label,
                "class_name_raw": det.class_name_raw,
                "class_id": det.class_id,
                "det_conf": det.conf,
            }
        )

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None
            self._writer = None
