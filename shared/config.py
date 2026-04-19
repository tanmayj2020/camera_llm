"""Shared configuration — all secrets loaded from environment variables."""

import os

# Edge defaults
DEFAULT_RTSP_URL = os.environ.get("RTSP_URL", "rtsp://localhost:8554/stream")
DEFAULT_MIN_FPS = int(os.environ.get("MIN_FPS", "1"))
DEFAULT_MAX_FPS = int(os.environ.get("MAX_FPS", "5"))
DEFAULT_ACTIVITY_THRESHOLD = float(os.environ.get("ACTIVITY_THRESHOLD", "0.02"))
DEFAULT_KEYFRAME_QUALITY = int(os.environ.get("KEYFRAME_QUALITY", "80"))

# Detection defaults
DEFAULT_DETECTION_CONFIDENCE = float(os.environ.get("DETECTION_CONFIDENCE", "0.3"))
DEFAULT_TRACKING_IOU_THRESHOLD = float(os.environ.get("TRACKING_IOU_THRESHOLD", "0.5"))
DEFAULT_DETECTION_CLASSES = os.environ.get("DETECTION_CLASSES", "person").split(",")

# Audio defaults
DEFAULT_AUDIO_SAMPLE_RATE = int(os.environ.get("AUDIO_SAMPLE_RATE", "16000"))
DEFAULT_AUDIO_CHUNK_DURATION = float(os.environ.get("AUDIO_CHUNK_DURATION", "1.0"))

# Cloud defaults
PUBSUB_TOPIC = os.environ.get("PUBSUB_TOPIC", "visionbrain-events")
PUBSUB_SUBSCRIPTION = os.environ.get("PUBSUB_SUBSCRIPTION", "visionbrain-events-sub")
GCS_KEYFRAME_BUCKET = os.environ.get("GCS_KEYFRAME_BUCKET", "visionbrain-keyframes")
BIGQUERY_DATASET = os.environ.get("BIGQUERY_DATASET", "visionbrain")
BIGQUERY_EVENTS_TABLE = os.environ.get("BIGQUERY_EVENTS_TABLE", "events")

# Neo4j / Graphiti — NEVER hardcode passwords
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")  # Must be set via env

# Qdrant
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "visionbrain-keyframes")

# PostgreSQL (replaces SQLite for production)
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///visionbrain.db")

# Auth
JWT_SECRET = os.environ.get("JWT_SECRET", "")  # Must be set in production
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "false").lower() in ("1", "true", "yes")

# Observability
OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
OTEL_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "visionbrain")
