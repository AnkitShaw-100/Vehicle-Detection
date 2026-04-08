from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from anpr.types import PlateDetection


class YoloPlateDetector:
    def __init__(
        self,
        weights_path: str,
        conf: float,
        imgsz: int,
        device: str,
    ) -> None:
        self._model = YOLO(weights_path)
        self._conf = conf
        self._imgsz = imgsz
        self._device = device

    def detect(self, frame_bgr: np.ndarray) -> list[PlateDetection]:
        h, w = frame_bgr.shape[:2]
        results = self._model.predict(
            frame_bgr,
            conf=self._conf,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )
        out: list[PlateDetection] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cfs = boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                c = float(cfs[i])
                xi1 = int(max(0, min(w - 1, round(x1))))
                yi1 = int(max(0, min(h - 1, round(y1))))
                xi2 = int(max(0, min(w, round(x2))))
                yi2 = int(max(0, min(h, round(y2))))
                if xi2 <= xi1 or yi2 <= yi1:
                    continue
                out.append(PlateDetection(bbox=(xi1, yi1, xi2, yi2), conf=c))
        out.sort(key=lambda d: d.conf, reverse=True)
        return out
