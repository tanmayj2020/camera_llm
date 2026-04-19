"""On-edge privacy engine — blurs faces and license plates before data leaves the device.

Face detection priority: InsightFace SCRFD (SOTA) → MediaPipe → Haar cascade fallback.
Plate detection priority: YOLO-trained plate detector → contour heuristic fallback.

SCRFD achieves 96.1% mAP on WiderFace hard set vs MediaPipe ~85% and Haar ~60%.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PrivacyEngine:
    """Detects and blurs faces and license plates with SOTA models.

    Face: InsightFace SCRFD (handles extreme poses, occlusion, low-light)
    Plates: YOLO-based plate detector for reliable detection
    """

    def __init__(self, blur_faces: bool = True, blur_plates: bool = True, blur_ksize: int = 99):
        self.blur_faces = blur_faces
        self.blur_plates = blur_plates
        self.ksize = blur_ksize | 1

        # Face detection backends (priority order)
        self._insightface_app = None
        self._mp_face = None
        self._face_cascade = None
        self._face_backend = "none"

        # Plate detection backends
        self._plate_model = None
        self._plate_backend = "none"

        if blur_faces:
            self._init_face_detector()

        if blur_plates:
            self._init_plate_detector()

    def _init_face_detector(self):
        """Initialize face detector: InsightFace SCRFD → MediaPipe → Haar."""
        # Priority 1: InsightFace SCRFD (96.1% mAP on WiderFace hard)
        try:
            from insightface.app import FaceAnalysis
            self._insightface_app = FaceAnalysis(
                name="buffalo_sc",  # SCRFD model, lightweight
                allowed_modules=["detection"],
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._insightface_app.prepare(ctx_id=0, det_size=(640, 640))
            self._face_backend = "insightface"
            logger.info("Privacy engine: InsightFace SCRFD face detection enabled")
            return
        except Exception as e:
            logger.info("InsightFace unavailable (%s), trying MediaPipe", e)

        # Priority 2: MediaPipe Face Detection
        try:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)
            self._face_backend = "mediapipe"
            logger.info("Privacy engine: MediaPipe face detection enabled")
            return
        except Exception as e:
            logger.info("MediaPipe unavailable (%s), falling back to Haar cascade", e)

        # Priority 3: Haar cascade (last resort)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        if self._face_cascade.empty():
            logger.warning("Haar cascade not found — face blurring disabled")
            self._face_cascade = None
        else:
            self._face_backend = "haar"
            logger.info("Privacy engine: Haar cascade face detection (degraded accuracy)")

    def _init_plate_detector(self):
        """Initialize plate detector: YOLO plate model → contour heuristic."""
        try:
            from ultralytics import YOLO
            # Try dedicated plate detection model
            self._plate_model = YOLO("yolov8n-lp.pt")  # plate-trained model
            self._plate_backend = "yolo"
            logger.info("Privacy engine: YOLO plate detection enabled")
        except Exception as e:
            logger.info("YOLO plate model unavailable (%s), using contour heuristic", e)
            self._plate_backend = "contour"

    def _blur_region(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        # Clamp to frame bounds
        fh, fw = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w = min(w, fw - x)
        h = min(h, fh - y)
        roi = frame[y: y + h, x: x + w]
        if roi.size == 0:
            return
        frame[y: y + h, x: x + w] = cv2.GaussianBlur(roi, (self.ksize, self.ksize), 30)

    def _detect_faces(self, frame: np.ndarray, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        # InsightFace SCRFD
        if self._face_backend == "insightface" and self._insightface_app:
            try:
                faces_result = self._insightface_app.get(frame)
                faces = []
                for face in faces_result:
                    bbox = face.bbox.astype(int)
                    x, y = int(bbox[0]), int(bbox[1])
                    w, h = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
                    faces.append((x, y, w, h))
                return faces
            except Exception as e:
                logger.debug("InsightFace detection error: %s", e)
                return []

        # MediaPipe
        if self._face_backend == "mediapipe" and self._mp_face:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._mp_face.process(rgb)
            faces = []
            if results.detections:
                fh, fw = frame.shape[:2]
                for det in results.detections:
                    bb = det.location_data.relative_bounding_box
                    x = max(0, int(bb.xmin * fw))
                    y = max(0, int(bb.ymin * fh))
                    bw = int(bb.width * fw)
                    bh = int(bb.height * fh)
                    faces.append((x, y, bw, bh))
            return faces

        # Haar cascade fallback
        if self._face_cascade is not None:
            return list(self._face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30)))
        return []

    def _detect_plates(self, frame: np.ndarray, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        # YOLO plate model
        if self._plate_backend == "yolo" and self._plate_model:
            try:
                results = self._plate_model(frame, conf=0.4, verbose=False)
                plates = []
                for r in results:
                    if r.boxes is None:
                        continue
                    for i in range(len(r.boxes)):
                        x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                        plates.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                return plates
            except Exception as e:
                logger.debug("YOLO plate detection error: %s", e)

        # Contour heuristic fallback
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
        out = frame.copy()
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        n_faces = n_plates = 0
        if self.blur_faces:
            for x, y, w, h in self._detect_faces(out, gray):
                self._blur_region(out, x, y, w, h)
                n_faces += 1

        if self.blur_plates:
            for x, y, w, h in self._detect_plates(out, gray):
                self._blur_region(out, x, y, w, h)
                n_plates += 1

        if n_faces or n_plates:
            logger.debug("Blurred %d faces (%s), %d plates (%s)",
                         n_faces, self._face_backend, n_plates, self._plate_backend)
        return out
