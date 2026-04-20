"""VisionBrain Edge Pipeline — main entry point.

Ties together: frame capture → privacy → detection → pose → activity → edge VLM → event emission.
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
from detection.activity_recognizer import ActivityRecognizer
from emitter.event_emitter import EventEmitter, encode_keyframe
from privacy.engine import PrivacyEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("visionbrain.edge")


def build_event(camera_id, meta, detections, poses, audio_events, keyframe_b64,
                activity_events=None, vlm_desc=None):
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

    event = {
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

    if activity_events:
        event["activity_events"] = [
            {"track_id": a.track_id, "activity": a.activity, "confidence": round(a.confidence, 3)}
            for a in activity_events
        ]

    if vlm_desc:
        event["edge_vlm_description"] = vlm_desc.description
        event["edge_vlm_anomaly_hints"] = vlm_desc.anomaly_hints

    return event


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
    parser.add_argument("--no-vlm", action="store_true", help="Disable edge VLM scene descriptions")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--min-fps", type=int, default=1)
    parser.add_argument("--max-fps", type=int, default=5)
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # ── Startup banner ──
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  VisionBrain Edge Pipeline               ║")
    logger.info("║  Camera: %-30s ║", args.camera_id)
    logger.info("║  Source: %-30s ║", str(source)[:30])
    logger.info("║  Model: %-30s  ║", args.model)
    logger.info("║  Classes: %-28s ║", ", ".join(args.classes)[:28])
    logger.info("║  Mode: %-31s ║", args.mode)
    logger.info("╚══════════════════════════════════════════╝")

    # ── Initialize components with validation ──
    extractor = FrameExtractor(source, args.camera_id, args.min_fps, args.max_fps)
    privacy = PrivacyEngine() if not args.no_privacy else None

    detector = OpenVocabDetector(args.model, args.confidence)
    detector.set_classes(args.classes)
    if not detector.is_open_vocab:
        logger.warning("Open-vocab detection unavailable — using standard YOLO classes only")

    pose_estimator = PoseEstimator() if not args.no_pose else None
    activity_recognizer = ActivityRecognizer() if not args.no_pose else None

    emitter = EventEmitter(mode=args.mode, pubsub_topic=args.pubsub_topic, project_id=args.project_id)

    # Validate Pub/Sub connectivity if needed
    if args.mode == "pubsub":
        if not args.project_id:
            logger.error("--project-id required for pubsub mode")
            sys.exit(1)
        try:
            emitter.health_check()
            logger.info("✓ Pub/Sub topic verified")
        except Exception as e:
            logger.warning("✗ Pub/Sub connectivity not verified: %s", e)

    # Validate video source
    extractor.start()
    test_frame = extractor.get_frame(timeout=5.0)
    if test_frame is None:
        logger.error("Cannot read from video source: %s — check RTSP URL or camera index", source)
        sys.exit(1)
    logger.info("✓ Video source accessible (%dx%d)", test_frame[0].shape[1], test_frame[0].shape[0])

    # Edge VLM (optional)
    vlm_scheduler = None
    if not args.no_vlm:
        try:
            from detection.edge_vlm import EdgeVLM, EdgeVLMScheduler
            vlm_scheduler = EdgeVLMScheduler(EdgeVLM(), interval_s=30.0, anomaly_threshold=0.05)
            logger.info("Edge VLM enabled")
        except Exception as e:
            logger.warning("Edge VLM disabled: %s", e)

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

    running = True

    def _shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Edge pipeline running — classes=%s, mode=%s, open_vocab=%s",
                args.classes, args.mode, detector.is_open_vocab)

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

            # 4. Activity recognition from pose sequences
            activity_events = []
            if activity_recognizer and poses:
                for p in poses:
                    if p["track_id"] >= 0 and p.get("keypoints"):
                        activity_recognizer.update(
                            p["track_id"], p["keypoints"],
                            p["bbox"], meta["timestamp"])
                activity_events = activity_recognizer.classify_all(meta["timestamp"])
                if frame_count % 100 == 0:
                    activity_recognizer.cleanup()

            # 5. Audio
            audio_events = []
            if audio_classifier and audio_capture:
                chunk = audio_capture.get_chunk(timeout=0.01)
                if chunk is not None:
                    audio_events = audio_classifier.classify(chunk)

            # 6. Edge VLM — check on anomalous frames or periodic interval
            vlm_desc = None
            if vlm_scheduler:
                vlm_desc = vlm_scheduler.check(
                    processed, args.camera_id, anomaly_score=meta["scene_activity"])

            # 7. Encode keyframe
            keyframe_b64 = encode_keyframe(processed) if detections else None

            # 8. Build and emit event
            event = build_event(
                args.camera_id, meta, detections, poses, audio_events, keyframe_b64,
                activity_events=activity_events, vlm_desc=vlm_desc)
            emitter.emit(event)

            frame_count += 1
            if frame_count % 50 == 0:
                elapsed = time.time() - t0
                act_str = ", ".join(f"{a.activity}({a.track_id})" for a in activity_events) or "none"
                logger.info(
                    "Processed %d frames (%.1f fps) | objects=%d activity=%.3f actions=[%s]",
                    frame_count, frame_count / elapsed, len(detections),
                    meta["scene_activity"], act_str,
                )
    finally:
        extractor.stop()
        if audio_capture:
            audio_capture.stop()
        logger.info("Edge pipeline stopped after %d frames", frame_count)


if __name__ == "__main__":
    main()
