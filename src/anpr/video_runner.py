from __future__ import annotations

import os

import cv2
import numpy as np

from anpr.logging import CsvPlateLogger
from anpr.ocr import is_plausible_plate, ocr_plate
from anpr.types import PlateRead


def pad_and_clip_bbox(
    bbox: tuple[int, int, int, int],
    pad: float,
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    dx = pad * bw
    dy = pad * bh
    x1 = int(round(x1 - dx))
    y1 = int(round(y1 - dy))
    x2 = int(round(x2 + dx))
    y2 = int(round(y2 + dy))
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def _timestamp_ms(cap: cv2.VideoCapture, frame_idx: int, fps: float) -> int:
    t = cap.get(cv2.CAP_PROP_POS_MSEC)
    if t and t > 0:
        return int(round(t))
    if fps and fps > 0:
        return int(round(1000.0 * frame_idx / fps))
    return 0


class VideoRunner:
    def __init__(
        self,
        video_path: str,
        detector,
        tesseract_cmd: str | None,
        logger: CsvPlateLogger | None,
        show: bool,
        save_video_path: str | None,
        save_crops_dir: str | None,
        frame_skip: int,
        max_plates_per_frame: int,
        bbox_pad: float = 0.12,
        ocr_psm: int = 7,
    ) -> None:
        self._video_path = video_path
        self._detector = detector
        self._tesseract_cmd = tesseract_cmd
        self._logger = logger
        self._show = show
        self._save_video_path = save_video_path
        self._save_crops_dir = save_crops_dir
        self._frame_skip = max(0, frame_skip)
        self._max_plates = max(1, max_plates_per_frame)
        self._bbox_pad = bbox_pad
        self._ocr_psm = ocr_psm

    def run(self) -> None:
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self._video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        writer: cv2.VideoWriter | None = None
        if self._save_video_path:
            parent = os.path.dirname(os.path.abspath(self._save_video_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                self._save_video_path, fourcc, fps if fps > 0 else 25.0, (width, height)
            )
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Cannot create video writer: {self._save_video_path}")

        if self._save_crops_dir:
            os.makedirs(self._save_crops_dir, exist_ok=True)

        frame_idx = 0
        processed = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                h, w = frame.shape[:2]

                if self._frame_skip > 0 and processed % (self._frame_skip + 1) != 0:
                    if writer is not None:
                        writer.write(frame)
                    if self._show:
                        cv2.imshow("ANPR", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                    processed += 1
                    frame_idx += 1
                    continue

                ts_ms = _timestamp_ms(cap, frame_idx, fps)
                dets = self._detector.detect(frame)
                dets = dets[: self._max_plates]

                annotated = frame.copy()
                for det in dets:
                    bx1, by1, bx2, by2 = pad_and_clip_bbox(det.bbox, self._bbox_pad, w, h)
                    crop = frame[by1:by2, bx1:bx2]
                    text, ocr_conf = ocr_plate(
                        crop,
                        self._tesseract_cmd,
                        psm=self._ocr_psm,
                    )
                    if text and not is_plausible_plate(text):
                        text = ""

                    if self._logger:
                        self._logger.log(
                            PlateRead(
                                frame_idx=frame_idx,
                                timestamp_ms=ts_ms,
                                bbox=(bx1, by1, bx2, by2),
                                det_conf=det.conf,
                                ocr_text=text,
                                ocr_conf=ocr_conf,
                            )
                        )

                    if self._save_crops_dir and crop.size > 0:
                        safe = text or "nodetect"
                        fname = f"f{frame_idx:06d}_{safe}_{bx1}_{by1}.png"
                        path = os.path.join(self._save_crops_dir, fname)
                        cv2.imwrite(path, crop)

                    # License-plate pipeline only — not "vehicle" / not generic "plate" label
                    label = f"{text} ({det.conf:.2f})" if text else f"LP? ({det.conf:.2f})"
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        label[:80],
                        (bx1, max(0, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                if writer is not None:
                    writer.write(annotated)
                if self._show:
                    cv2.imshow("ANPR", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

                processed += 1
                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if self._show:
                cv2.destroyAllWindows()
