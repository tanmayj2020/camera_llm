"""On-edge privacy engine — blurs faces and license plates before data leaves the device."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PrivacyEngine:
    """Detects and blurs faces (Haar cascade) and license-plate-like rectangles."""

    def __init__(self, blur_faces: bool = True, blur_plates: bool = True, blur_ksize: int = 99):
        self.blur_faces = blur_faces
        self.blur_plates = blur_plates
        self.ksize = blur_ksize | 1  # must be odd

        self._face_cascade = None
        if blur_faces:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            if self._face_cascade.empty():
                logger.warning("Haar cascade not found – face blurring disabled")
                self._face_cascade = None

    def _blur_region(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return
        frame[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (self.ksize, self.ksize), 30)

    def _detect_faces(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._face_cascade is None:
            return []
        return list(self._face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30)))

    def _detect_plates(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Heuristic plate detection via edge contours with plate-like aspect ratio."""
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        plates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            aspect = w / h
            area = w * h
            if 2.0 <= aspect <= 5.5 and 800 < area < 30000:
                plates.append((x, y, w, h))
        return plates

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Returns a copy of frame with faces and plates blurred."""
        out = frame.copy()
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        n_faces = n_plates = 0
        if self.blur_faces:
            for x, y, w, h in self._detect_faces(gray):
                self._blur_region(out, x, y, w, h)
                n_faces += 1

        if self.blur_plates:
            for x, y, w, h in self._detect_plates(gray):
                self._blur_region(out, x, y, w, h)
                n_plates += 1

        if n_faces or n_plates:
            logger.debug("Blurred %d faces, %d plates", n_faces, n_plates)
        return out
