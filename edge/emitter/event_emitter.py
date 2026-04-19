"""Edge event emitter — publishes fused events to GCP Pub/Sub or local file."""

import base64
import json
import logging
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def encode_keyframe(frame: np.ndarray, quality: int = 80) -> str:
    """JPEG-encode a frame and return base64 string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


class EventEmitter:
    """Publishes VisionEvents as JSON to Pub/Sub or local file with SQLite buffering."""

    def __init__(self, mode: str = "local", pubsub_topic: str = "", project_id: str = "",
                 output_dir: str = "events_out", buffer_db: str = "event_buffer.db"):
        self.mode = mode  # "local" or "pubsub"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self._publisher = None
        self._topic_path = ""

        if mode == "pubsub" and pubsub_topic:
            try:
                from google.cloud import pubsub_v1
                self._publisher = pubsub_v1.PublisherClient()
                self._topic_path = self._publisher.topic_path(project_id, pubsub_topic)
                logger.info("Pub/Sub emitter: %s", self._topic_path)
            except Exception as e:
                logger.warning("Pub/Sub init failed: %s — falling back to local + buffer", e)
                self.mode = "local"

        # SQLite buffer for offline resilience
        self._db = sqlite3.connect(buffer_db, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS buffer (id INTEGER PRIMARY KEY, payload TEXT, created_at REAL)"
        )
        self._db.commit()

    def emit(self, event: dict) -> None:
        payload = json.dumps(event, default=str)

        if self.mode == "pubsub" and self._publisher:
            try:
                self._publisher.publish(self._topic_path, payload.encode())
                self._flush_buffer()
                return
            except Exception as e:
                logger.warning("Pub/Sub publish failed, buffering: %s", e)
                self._buffer(payload)
                return

        # Local mode: write JSON file
        fname = self.output_dir / f"{event.get('event_id', int(time.time()))}.json"
        fname.write_text(payload)

    def _buffer(self, payload: str) -> None:
        self._db.execute("INSERT INTO buffer (payload, created_at) VALUES (?, ?)", (payload, time.time()))
        self._db.commit()

    def _flush_buffer(self) -> None:
        rows = self._db.execute("SELECT id, payload FROM buffer ORDER BY id LIMIT 50").fetchall()
        if not rows:
            return
        ids = []
        for row_id, payload in rows:
            try:
                self._publisher.publish(self._topic_path, payload.encode())
                ids.append(row_id)
            except Exception:
                break
        if ids:
            self._db.execute(f"DELETE FROM buffer WHERE id IN ({','.join('?' * len(ids))})", ids)
            self._db.commit()
            logger.info("Flushed %d buffered events", len(ids))
