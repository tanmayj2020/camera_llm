"""State-of-the-art open-vocabulary object detection and pose estimation.

Priority chain:
  1. YOLO-World v2 (true open-vocab with arbitrary text prompts)
  2. YOLO11 (latest generation, COCO filtering)
  3. YOLOv8 fallback

Pose: ViTPose++ via MMPose → YOLO11-pose fallback → YOLOv8-pose fallback
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class OpenVocabDetector:
    """True open-vocabulary detector with multi-model fallback chain.

    YOLO-World v2: accepts arbitrary text prompts like 'person carrying crowbar',
    'abandoned backpack', 'vehicle with open door'.
    YOLO11/YOLOv8: COCO class filtering only.
    """

    _COCO_NAMES: list[str] = []

    def __init__(self, model_name: str = "yolo11n.pt", confidence: float = 0.3, device: str = ""):
        from ultralytics import YOLO

        self.confidence = confidence
        self.device = device
        self._target_classes: Optional[list[str]] = None
        self._target_indices: Optional[list[int]] = None
        self._open_vocab = False

        # Priority 1: YOLO-World v2 (true open vocabulary)
        try:
            self.model = YOLO("yolov8s-worldv2.pt")
            if hasattr(self.model, "set_classes"):
                self._open_vocab = True
                logger.info("YOLO-World v2 open-vocabulary detection enabled")
            else:
                raise RuntimeError("Model lacks set_classes")
        except Exception as e:
            logger.info("YOLO-World unavailable (%s), trying YOLO11", e)
            # Priority 2: YOLO11 (latest generation)
            try:
                self.model = YOLO("yolo11n.pt")
                logger.info("YOLO11 detection enabled")
            except Exception as e2:
                logger.info("YOLO11 unavailable (%s), falling back to YOLOv8", e2)
                # Priority 3: YOLOv8 fallback
                self.model = YOLO(model_name if "yolo" in model_name else "yolov8n.pt")
                logger.info("YOLOv8 fallback detection enabled: %s", model_name)

        if hasattr(self.model, "names"):
            self._COCO_NAMES = list(self.model.names.values())

    def set_classes(self, class_names: list[str]) -> None:
        """Set detection target classes.

        YOLO-World v2: accepts arbitrary text prompts (e.g. 'person carrying crowbar').
        YOLO11/YOLOv8: filters to matching COCO classes only.
        """
        self._target_classes = [c.lower() for c in class_names]

        if self._open_vocab:
            try:
                self.model.set_classes(class_names)
                logger.info("YOLO-World open-vocab classes set: %s", class_names)
                self._target_indices = None
                # Re-read class names after set_classes updates model head
                if hasattr(self.model, "names"):
                    self._COCO_NAMES = list(self.model.names.values())
                return
            except Exception as e:
                logger.warning("YOLO-World set_classes failed: %s, using index filter", e)

        self._target_indices = [
            i for i, name in enumerate(self._COCO_NAMES) if name.lower() in self._target_classes
        ]
        logger.info("Detection targets: %s (indices %s)", self._target_classes, self._target_indices)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run detection + tracking, return list of detections."""
        kwargs = {"conf": self.confidence, "verbose": False, "persist": True}
        if self.device:
            kwargs["device"] = self.device
        if self._target_indices and not self._open_vocab:
            kwargs["classes"] = self._target_indices

        results = self.model.track(frame, **kwargs)
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            names = r.names if hasattr(r, "names") and r.names else {}
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                if names:
                    class_name = names.get(cls_id, str(cls_id))
                elif cls_id < len(self._COCO_NAMES):
                    class_name = self._COCO_NAMES[cls_id]
                else:
                    class_name = str(cls_id)

                det = {
                    "class_name": class_name,
                    "bbox": boxes.xyxy[i].tolist(),
                    "confidence": float(boxes.conf[i]),
                    "track_id": int(boxes.id[i]) if boxes.id is not None else -1,
                }
                detections.append(det)
        return detections

    @property
    def is_open_vocab(self) -> bool:
        return self._open_vocab


class PoseEstimator:
    """Human pose estimation with multi-model fallback.

    Priority: ViTPose++ (MMPose) → YOLO11-pose → YOLOv8-pose
    ViTPose++ achieves 81.1 AP on COCO, significantly ahead of YOLO-pose variants.
    """

    def __init__(self, model_name: str = "yolo11n-pose.pt"):
        self._mmpose_model = None
        self._yolo_model = None
        self._use_mmpose = False

        # Priority 1: ViTPose++ via MMPose
        try:
            from mmpose.apis import MMPoseInferencer
            self._mmpose_model = MMPoseInferencer(
                pose2d="td-hm_ViTPose-huge_8xb64-210e_coco-256x192",
                det_model="rtmdet_m_8xb32-300e_coco",
            )
            self._use_mmpose = True
            logger.info("ViTPose++ (MMPose) pose estimation enabled")
        except Exception as e:
            logger.info("MMPose/ViTPose unavailable (%s), using YOLO-pose", e)
            # Priority 2: YOLO11-pose → YOLOv8-pose
            from ultralytics import YOLO
            try:
                self._yolo_model = YOLO("yolo11n-pose.pt")
                logger.info("YOLO11-pose estimation enabled")
            except Exception:
                self._yolo_model = YOLO(model_name if "pose" in model_name else "yolov8n-pose.pt")
                logger.info("YOLOv8-pose fallback enabled")

    def estimate(self, frame: np.ndarray) -> list[dict]:
        if self._use_mmpose and self._mmpose_model:
            return self._estimate_mmpose(frame)
        return self._estimate_yolo(frame)

    def _estimate_mmpose(self, frame: np.ndarray) -> list[dict]:
        """ViTPose++ inference via MMPose."""
        try:
            results_gen = self._mmpose_model(frame, return_vis=False)
            result = next(results_gen)
            poses = []
            preds = result.get("predictions", [[]])[0]
            for pred in preds:
                kpts = pred.get("keypoints", [])
                scores = pred.get("keypoint_scores", [])
                bbox = pred.get("bbox", [[]])[0] if pred.get("bbox") else []
                keypoints = [[kpts[j][0], kpts[j][1], scores[j]] for j in range(len(kpts))]
                poses.append({
                    "bbox": bbox,
                    "keypoints": keypoints,
                    "track_id": pred.get("track_id", -1),
                })
            return poses
        except Exception as e:
            logger.error("MMPose inference failed: %s, falling back to YOLO", e)
            return self._estimate_yolo(frame)

    def _estimate_yolo(self, frame: np.ndarray) -> list[dict]:
        """YOLO-pose fallback."""
        results = self._yolo_model.track(frame, verbose=False, persist=True)
        poses = []
        for r in results:
            if r.keypoints is None:
                continue
            boxes = r.boxes
            for i in range(len(r.keypoints)):
                kpts = r.keypoints[i].data.cpu().numpy().tolist()
                pose = {
                    "bbox": boxes.xyxy[i].tolist() if boxes is not None else [],
                    "keypoints": kpts[0] if len(kpts) == 1 else kpts,
                    "track_id": int(boxes.id[i]) if boxes is not None and boxes.id is not None else -1,
                }
                poses.append(pose)
        return poses
