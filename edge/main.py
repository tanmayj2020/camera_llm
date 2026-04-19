"""VisionBrain Edge Pipeline — main entry point.

Ties together: frame capture → privacy → detection → audio → event emission.
"""

import argparse
import json
import logging
import signal
import sys
import time
import uuid

from capture.frame_extractor import FrameExtractor
from detection.detector import OpenVocabDetector, PoseEstimator
from emitter.event_emitter import EventEmitter, encode_keyframe
from privacy.engine import PrivacyEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("visionbrain.edge")


def build_event(camera_id, meta, detections, poses, audio_events, keyframe_b64):
    objects = []
    pose_map = {p["track_id"]: p["keypoints"] for p in poses}
    for d in detections:
        obj = {
            "class_name": d["class_name"],
            "bbox": d["bbox"],
            "track_id": d["track_id"],
            "confidence": d["confidence"],
        }
        if d["track_id"] in pose_map:
            obj["pose"] = pose_map[d["track_id"]]
        objects.append(obj)

    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": meta["timestamp"],
        "camera_id": camera_id,
        "event_type": "detection",
        "frame_index": meta["frame_index"],
        "scene_activity": meta["scene_activity"],
        "objects": objects,
        "audio_events": audio_events,
        "keyframe_b64": keyframe_b64,
        "privacy_applied": True,
    }


def main():
    parser = argparse.ArgumentParser(description="VisionBrain Edge Pipeline")
    parser.add_argument("--source", default="0", help="RTSP URL, video file, or camera index")
    parser.add_argument("--camera-id", default="cam-0")
    parser.add_argument("--classes", nargs="+", default=["person"], help="Detection target classes")
    parser.add_argument("--mode", choices=["local", "pubsub"], default="local")
    parser.add_argument("--pubsub-topic", default="visionbrain-events")
    parser.add_argument("--project-id", default="")
    parser.add_argument("--no-privacy", action="store_true")
    parser.add_argument("--no-pose", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--min-fps", type=int, default=1)
    parser.add_argument("--max-fps", type=int, default=5)
    args = parser.parse_args()

    # Try to interpret source as int (webcam index)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # Initialize components
    extractor = FrameExtractor(source, args.camera_id, args.min_fps, args.max_fps)
    privacy = PrivacyEngine() if not args.no_privacy else None
    detector = OpenVocabDetector(args.model, args.confidence)
    detector.set_classes(args.classes)
    pose_estimator = PoseEstimator() if not args.no_pose else None
    emitter = EventEmitter(mode=args.mode, pubsub_topic=args.pubsub_topic, project_id=args.project_id)

    # Audio (optional)
    audio_classifier = None
    audio_capture = None
    if not args.no_audio:
        try:
            from audio.processor import AudioCapture, SoundClassifier
            audio_classifier = SoundClassifier()
            audio_capture = AudioCapture()
            audio_capture.start()
            logger.info("Audio processing enabled")
        except Exception as e:
            logger.warning("Audio disabled: %s", e)

    # Graceful shutdown
    running = True

    def _shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    extractor.start()
    logger.info("Edge pipeline running — classes=%s, mode=%s", args.classes, args.mode)

    frame_count = 0
    t0 = time.time()

    try:
        while running:
            result = extractor.get_frame(timeout=1.0)
            if result is None:
                continue
            frame, meta = result

            # 1. Privacy
            processed = privacy.process(frame) if privacy else frame

            # 2. Detection
            detections = detector.detect(processed)

            # 3. Pose
            poses = []
            if pose_estimator and detections:
                try:
                    poses = pose_estimator.estimate(processed)
                except Exception:
                    pass

            # 4. Audio
            audio_events = []
            if audio_classifier and audio_capture:
                chunk = audio_capture.get_chunk(timeout=0.01)
                if chunk is not None:
                    audio_events = audio_classifier.classify(chunk)

            # 5. Encode keyframe
            keyframe_b64 = encode_keyframe(processed) if detections else None

            # 6. Build and emit event
            event = build_event(args.camera_id, meta, detections, poses, audio_events, keyframe_b64)
            emitter.emit(event)

            frame_count += 1
            if frame_count % 50 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "Processed %d frames (%.1f fps) | objects=%d activity=%.3f",
                    frame_count, frame_count / elapsed, len(detections), meta["scene_activity"],
                )
    finally:
        extractor.stop()
        if audio_capture:
            audio_capture.stop()
        logger.info("Edge pipeline stopped after %d frames", frame_count)


if __name__ == "__main__":
    main()
