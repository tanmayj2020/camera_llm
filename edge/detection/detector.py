"""Open-vocabulary object detection and pose estimation using Ultralytics."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class OpenVocabDetector:
    """Wraps Ultralytics YOLO for detection + tracking.

    Uses yolov8n by default; swap to yoloe-26 model path when available.
    """

    # COCO class names for filtering when using standard YOLO
    _COCO_NAMES: list[str] = []  # populated on first model load

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.3, device: str = ""):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.device = device
        self._target_classes: Optional[list[str]] = None
        self._target_indices: Optional[list[int]] = None

        # cache coco names from model
        if hasattr(self.model, "names"):
            self._COCO_NAMES = list(self.model.names.values())

    def set_classes(self, class_names: list[str]) -> None:
        """Set which classes to detect. For standard YOLO, filters to matching COCO classes."""
        self._target_classes = [c.lower() for c in class_names]
        self._target_indices = [
            i for i, name in enumerate(self._COCO_NAMES) if name.lower() in self._target_classes
        ]
        logger.info("Detection targets: %s (indices %s)", self._target_classes, self._target_indices)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run detection + tracking, return list of detections."""
        kwargs = {"conf": self.confidence, "verbose": False, "persist": True}
        if self.device:
            kwargs["device"] = self.device
        if self._target_indices:
            kwargs["classes"] = self._target_indices

        results = self.model.track(frame, **kwargs)
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                det = {
                    "class_name": self._COCO_NAMES[cls_id] if cls_id < len(self._COCO_NAMES) else str(cls_id),
                    "bbox": boxes.xyxy[i].tolist(),
                    "confidence": float(boxes.conf[i]),
                    "track_id": int(boxes.id[i]) if boxes.id is not None else -1,
                }
                detections.append(det)
        return detections


class PoseEstimator:
    """Wraps Ultralytics YOLO-Pose for human pose estimation."""

    def __init__(self, model_name: str = "yolov8n-pose.pt"):
        from ultralytics import YOLO

        self.model = YOLO(model_name)

    def estimate(self, frame: np.ndarray) -> list[dict]:
        results = self.model.track(frame, verbose=False, persist=True)
        poses = []
        for r in results:
            if r.keypoints is None:
                continue
            boxes = r.boxes
            for i in range(len(r.keypoints)):
                kpts = r.keypoints[i].data.cpu().numpy().tolist()  # [[x,y,conf],...]
                pose = {
                    "bbox": boxes.xyxy[i].tolist() if boxes is not None else [],
                    "keypoints": kpts[0] if len(kpts) == 1 else kpts,
                    "track_id": int(boxes.id[i]) if boxes is not None and boxes.id is not None else -1,
                }
                poses.append(pose)
        return poses
