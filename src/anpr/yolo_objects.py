from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from anpr.types import ObjectDetection

# User-facing keywords -> canonical COCO / common YOLO class names in model.names
_SYNONYMS: dict[str, str] = {
    "human": "person",
    "person": "person",
    "pedestrian": "person",
    "car": "car",
    "automobile": "car",
    "truck": "truck",
    "ambulance": "ambulance",
}

# COCO classes treated as "road vehicles" when user passes label `vehicle` / `vehicles`
_COCO_ROAD_VEHICLE_RAW: frozenset[str] = frozenset(
    {"car", "truck", "bus", "motorcycle"}
)


def _display_for_raw(raw: str) -> str:
    if raw == "person":
        return "Human"
    return raw[:1].upper() + raw[1:] if raw else raw


def _is_vehicle_group_label(requested: list[str]) -> bool:
    return any(x in ("vehicle", "vehicles", "vehical") for x in requested)


def parse_requested_labels(requested: str) -> list[str]:
    parts = [p.strip().lower() for p in requested.split(",") if p.strip()]
    return parts


def build_class_allowlist(
    model_names: dict[int, str],
    requested: list[str],
) -> tuple[set[int], dict[int, str], list[str]]:
    """
    Map user keywords to model class ids.
    model_names: ultralytics model.names (id -> str).
    Returns: allowed_ids, id_to_display_label, warnings.

    If the user includes the keyword ``vehicle`` / ``vehicles``, all COCO road-vehicle
    classes present in the model (car, truck, bus, motorcycle) are enabled and shown
    as the single label **Vehicle**. Listing ``car`` / ``truck`` alone (without ``vehicle``)
    keeps separate labels **Car** / **Truck**.
    """
    warnings: list[str] = []
    present_raw = {v.lower() for v in model_names.values()}
    use_vehicle_group = _is_vehicle_group_label(requested)

    want_raw: set[str] = set()
    for key in requested:
        k = key.strip().lower()
        if k in ("vehicle", "vehicles", "vehical"):
            for v in _COCO_ROAD_VEHICLE_RAW:
                if v in present_raw:
                    want_raw.add(v)
            continue
        canon = _SYNONYMS.get(k, k)
        want_raw.add(canon)

    for w in list(want_raw):
        if w not in present_raw:
            want_raw.discard(w)
            if w == "ambulance":
                warnings.append(
                    "Class 'ambulance' is not in this model's label set (skipping). "
                    "Standard COCO models have no 'ambulance'; use custom weights trained with that class."
                )
            else:
                warnings.append(f"Class '{w}' is not in this model's label set (skipping).")

    allowed_ids: set[int] = set()
    id_to_display: dict[int, str] = {}
    for cid, name in model_names.items():
        nl = name.lower()
        if nl not in want_raw:
            continue
        allowed_ids.add(int(cid))
        if nl == "person":
            id_to_display[int(cid)] = "Human"
        elif use_vehicle_group and nl in _COCO_ROAD_VEHICLE_RAW:
            id_to_display[int(cid)] = "Vehicle"
        else:
            id_to_display[int(cid)] = _display_for_raw(nl)

    return allowed_ids, id_to_display, warnings


class YoloObjectDetector:
    """Multi-class YOLO: filter to requested labels (human=person, car, truck, ambulance, ...)."""

    def __init__(
        self,
        weights_path: str,
        conf: float,
        imgsz: int,
        device: str,
        requested_labels: str,
        half: bool = False,
    ) -> None:
        self._model = YOLO(weights_path)
        self._conf = conf
        self._imgsz = imgsz
        self._device = device
        self._half = bool(half) and str(device).lower() != "cpu"
        names = self._model.names
        if not isinstance(names, dict):
            names = dict(names)  # type: ignore[arg-type]
        req = parse_requested_labels(requested_labels)
        self._allowed_ids, self._id_to_display, self.warnings = build_class_allowlist(names, req)
        if not self._allowed_ids:
            raise ValueError(
                "No matching classes for your --labels vs this model. "
                "Check model.names or use different weights."
            )

    def detect(self, frame_bgr: np.ndarray) -> list[ObjectDetection]:
        h, w = frame_bgr.shape[:2]
        results = self._model.predict(
            frame_bgr,
            conf=self._conf,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )
        out: list[ObjectDetection] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cfs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                cid = int(clss[i])
                if cid not in self._allowed_ids:
                    continue
                x1, y1, x2, y2 = xyxy[i]
                c = float(cfs[i])
                xi1 = int(max(0, min(w - 1, round(x1))))
                yi1 = int(max(0, min(h - 1, round(y1))))
                xi2 = int(max(0, min(w, round(x2))))
                yi2 = int(max(0, min(h, round(y2))))
                if xi2 <= xi1 or yi2 <= yi1:
                    continue
                raw = self._model.names.get(cid, str(cid))
                if isinstance(raw, (list, tuple)):
                    raw = str(raw)
                raw_s = str(raw).lower()
                label = self._id_to_display.get(cid, _display_for_raw(raw_s))
                out.append(
                    ObjectDetection(
                        bbox=(xi1, yi1, xi2, yi2),
                        conf=c,
                        class_id=cid,
                        label=label,
                        class_name_raw=raw_s,
                    )
                )
        out.sort(key=lambda d: d.conf, reverse=True)
        return out
