"""Adaptive frame extractor for RTSP/video streams."""

import logging
import threading
import time
from queue import Empty, Full, Queue

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Connects to RTSP/video source and extracts frames at adaptive rates
    based on scene activity (frame differencing)."""

    def __init__(
        self,
        source: str,
        camera_id: str = "cam-0",
        min_fps: int = 1,
        max_fps: int = 5,
        activity_threshold: float = 0.02,
        queue_size: int = 30,
    ):
        self.source = source
        self.camera_id = camera_id
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.activity_threshold = activity_threshold

        self._queue: Queue = Queue(maxsize=queue_size)
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_index = 0
        self._prev_gray: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _connect(self) -> cv2.VideoCapture:
        backoff = 1.0
        while self._running:
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                logger.info("Connected to %s", self.source)
                return cap
            logger.warning("Cannot open %s – retrying in %.1fs", self.source, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        return cv2.VideoCapture()  # dummy, loop will exit

    # ------------------------------------------------------------------
    def _compute_activity(self, gray: np.ndarray) -> float:
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return 0.0
        diff = cv2.absdiff(self._prev_gray, gray).astype(np.float32) / 255.0
        activity = float(np.mean(diff))
        self._prev_gray = gray
        return activity

    # ------------------------------------------------------------------
    def _target_interval(self, activity: float) -> float:
        """Return seconds between frames based on activity level."""
        if activity < self.activity_threshold:
            return 1.0 / self.min_fps
        ratio = min(activity / (self.activity_threshold * 5), 1.0)
        fps = self.min_fps + ratio * (self.max_fps - self.min_fps)
        return 1.0 / fps

    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        cap = self._connect()
        last_emit = 0.0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Lost stream, reconnecting…")
                cap.release()
                cap = self._connect()
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            activity = self._compute_activity(gray)
            interval = self._target_interval(activity)

            now = time.time()
            if now - last_emit < interval:
                continue

            meta = {
                "camera_id": self.camera_id,
                "timestamp": now,
                "frame_index": self._frame_index,
                "scene_activity": round(activity, 4),
            }
            try:
                self._queue.put_nowait((frame, meta))
                self._frame_index += 1
                last_emit = now
            except Full:
                pass  # drop frame if consumer is slow

        cap.release()

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("FrameExtractor started for %s", self.camera_id)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("FrameExtractor stopped for %s", self.camera_id)

    def get_frame(self, timeout: float = 1.0):
        """Returns (frame, metadata) or None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None
