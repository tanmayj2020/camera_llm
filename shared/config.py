"""Shared configuration constants."""

# Edge defaults
DEFAULT_RTSP_URL = "rtsp://localhost:8554/stream"
DEFAULT_MIN_FPS = 1
DEFAULT_MAX_FPS = 5
DEFAULT_ACTIVITY_THRESHOLD = 0.02  # frame diff threshold for activity detection
DEFAULT_KEYFRAME_QUALITY = 80  # JPEG quality for keyframes

# Detection defaults
DEFAULT_DETECTION_CONFIDENCE = 0.3
DEFAULT_TRACKING_IOU_THRESHOLD = 0.5
DEFAULT_DETECTION_CLASSES = ["person"]

# Audio defaults
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_CHUNK_DURATION = 1.0  # seconds

# Cloud defaults
PUBSUB_TOPIC = "visionbrain-events"
PUBSUB_SUBSCRIPTION = "visionbrain-events-sub"
GCS_KEYFRAME_BUCKET = "visionbrain-keyframes"
BIGQUERY_DATASET = "visionbrain"
BIGQUERY_EVENTS_TABLE = "events"

# Neo4j / Graphiti
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "visionbrain-keyframes"
