from __future__ import annotations

import os

import cv2


def _timestamp_ms(cap: cv2.VideoCapture, frame_idx: int, fps: float) -> int:
    t = cap.get(cv2.CAP_PROP_POS_MSEC)
    if t and t > 0:
        return int(round(t))
    if fps and fps > 0:
        return int(round(1000.0 * frame_idx / fps))
    return 0


def _color_bgr(label: str) -> tuple[int, int, int]:
    h = hash(label) % (256**3)
    b = h & 255
    g = (h >> 8) & 255
    r = (h >> 16) & 255
    # avoid too-dark colors
    b = max(40, b)
    g = max(40, g)
    r = max(40, r)
    return int(b), int(g), int(r)


class ObjectVideoRunner:
    def __init__(
        self,
        video_path: str,
        detector,
        logger,
        show: bool,
        save_video_path: str | None,
        frame_skip: int,
    ) -> None:
        self._video_path = video_path
        self._detector = detector
        self._logger = logger
        self._show = show
        self._save_video_path = save_video_path
        self._frame_skip = max(0, frame_skip)

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

        frame_idx = 0
        processed = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                if self._frame_skip > 0 and processed % (self._frame_skip + 1) != 0:
                    if writer is not None:
                        writer.write(frame)
                    if self._show:
                        cv2.imshow("Humans & vehicles", frame)
                        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                            break
                    processed += 1
                    frame_idx += 1
                    continue

                ts_ms = _timestamp_ms(cap, frame_idx, fps)
                dets = self._detector.detect(frame)
                annotated = frame.copy()

                for det in dets:
                    x1, y1, x2, y2 = det.bbox
                    if self._logger:
                        self._logger.log(frame_idx, ts_ms, det)
                    color = _color_bgr(det.label)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cap_txt = f"{det.label} {det.conf:.2f}"
                    cv2.putText(
                        annotated,
                        cap_txt[:64],
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                if writer is not None:
                    writer.write(annotated)
                if self._show:
                    cv2.imshow("Humans & vehicles", annotated)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break

                processed += 1
                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if self._show:
                cv2.destroyAllWindows()
