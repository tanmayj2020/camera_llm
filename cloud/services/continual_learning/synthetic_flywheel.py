"""Synthetic data flywheel — every real anomaly seeds synthetic training variations.

Augmentation strategies: time shift, spatial shift, object augment, counterfactual mirror,
anomaly injection into normal scenes. Compounding data moat.
"""

import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class AugmentationType(str, Enum):
    TIME_SHIFT = "time_shift"
    SPATIAL_SHIFT = "spatial_shift"
    OBJECT_AUGMENT = "object_augment"
    COUNTERFACTUAL_MIRROR = "counterfactual_mirror"
    ANOMALY_INJECTION = "anomaly_injection"


@dataclass
class SyntheticSample:
    sample_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_event_id: str = ""
    augmentation: AugmentationType = AugmentationType.TIME_SHIFT
    event_data: dict = field(default_factory=dict)
    label: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class FlywheelStats:
    real_seeds: int = 0
    synthetic_generated: int = 0
    augmentation_counts: dict = field(default_factory=dict)


class SyntheticFlywheel:
    """Every real anomaly becomes a seed for generating synthetic training variations.

    The flywheel compounds: more real data → more synthetic variations → better models →
    catch more anomalies → more real data.
    """

    AUGMENTATIONS = list(AugmentationType)
    VARIATIONS_PER_SEED = 5

    def __init__(self, store_path: str = "/tmp/visionbrain_synthetic"):
        self._seeds: list[dict] = []
        self._samples: list[SyntheticSample] = []
        self._normal_pool: list[dict] = []  # pool of normal events for injection
        self._store_path = store_path
        self._stats = FlywheelStats()

    def ingest_seed(self, anomaly_event: dict, label: str = "anomaly"):
        """Register a real anomaly as a seed for synthetic generation."""
        self._seeds.append({"event": anomaly_event, "label": label, "timestamp": time.time()})
        self._stats.real_seeds += 1
        logger.info("Flywheel seed ingested: %s (total seeds: %d)",
                    anomaly_event.get("event_id", "?"), len(self._seeds))

    def ingest_normal(self, event: dict):
        """Add a normal event to the pool (for anomaly injection augmentation)."""
        self._normal_pool.append(event)
        if len(self._normal_pool) > 1000:
            self._normal_pool = self._normal_pool[-500:]

    def generate_batch(self, max_samples: int = 50) -> list[SyntheticSample]:
        """Generate a batch of synthetic samples from all seeds."""
        if not self._seeds:
            return []

        samples = []
        for seed in self._seeds[-10:]:  # use most recent seeds
            for aug_type in self.AUGMENTATIONS:
                if len(samples) >= max_samples:
                    break
                sample = self._augment(seed["event"], seed["label"], aug_type)
                if sample:
                    samples.append(sample)
                    self._samples.append(sample)
                    self._stats.synthetic_generated += 1
                    self._stats.augmentation_counts[aug_type.value] = \
                        self._stats.augmentation_counts.get(aug_type.value, 0) + 1

        logger.info("Generated %d synthetic samples", len(samples))
        return samples

    def get_stats(self) -> FlywheelStats:
        return self._stats

    def get_training_set(self) -> list[SyntheticSample]:
        """Return all synthetic samples for training."""
        return list(self._samples)

    def _augment(self, event: dict, label: str, aug_type: AugmentationType) -> SyntheticSample | None:
        dispatch = {
            AugmentationType.TIME_SHIFT: self._time_shift,
            AugmentationType.SPATIAL_SHIFT: self._spatial_shift,
            AugmentationType.OBJECT_AUGMENT: self._object_augment,
            AugmentationType.COUNTERFACTUAL_MIRROR: self._counterfactual_mirror,
            AugmentationType.ANOMALY_INJECTION: self._anomaly_injection,
        }
        handler = dispatch.get(aug_type)
        if not handler:
            return None
        return handler(event, label)

    def _time_shift(self, event: dict, label: str) -> SyntheticSample:
        """Shift event to different time of day."""
        augmented = dict(event)
        shift_hours = random.choice([-6, -3, 3, 6, 12])
        augmented["timestamp"] = event.get("timestamp", time.time()) + shift_hours * 3600
        return SyntheticSample(
            source_event_id=event.get("event_id", ""),
            augmentation=AugmentationType.TIME_SHIFT,
            event_data=augmented, label=label,
            metadata={"shift_hours": shift_hours},
        )

    def _spatial_shift(self, event: dict, label: str) -> SyntheticSample:
        """Shift object positions within frame."""
        augmented = dict(event)
        objects = []
        dx, dy = random.uniform(-50, 50), random.uniform(-50, 50)
        for obj in event.get("objects", []):
            new_obj = dict(obj)
            if "bbox" in obj:
                b = obj["bbox"]
                if isinstance(b, dict):
                    new_obj["bbox"] = {k: v + (dx if 'x' in k else dy) for k, v in b.items()}
                elif isinstance(b, list) and len(b) == 4:
                    new_obj["bbox"] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
            objects.append(new_obj)
        augmented["objects"] = objects
        return SyntheticSample(
            source_event_id=event.get("event_id", ""),
            augmentation=AugmentationType.SPATIAL_SHIFT,
            event_data=augmented, label=label,
            metadata={"dx": dx, "dy": dy},
        )

    def _object_augment(self, event: dict, label: str) -> SyntheticSample:
        """Add/remove objects to create variations."""
        augmented = dict(event)
        objects = list(event.get("objects", []))
        if objects and random.random() < 0.5:
            # Add a synthetic bystander
            objects.append({
                "class_name": "person", "track_id": random.randint(9000, 9999),
                "confidence": 0.7,
                "bbox": [random.uniform(100, 1500), random.uniform(200, 800),
                         random.uniform(100, 1500) + 80, random.uniform(200, 800) + 200],
            })
        elif len(objects) > 1:
            # Remove a random non-primary object
            objects.pop(random.randint(1, len(objects) - 1))
        augmented["objects"] = objects
        return SyntheticSample(
            source_event_id=event.get("event_id", ""),
            augmentation=AugmentationType.OBJECT_AUGMENT,
            event_data=augmented, label=label,
            metadata={"object_count": len(objects)},
        )

    def _counterfactual_mirror(self, event: dict, label: str) -> SyntheticSample:
        """Create a 'normal' version of the anomaly (what if nothing was wrong)."""
        augmented = dict(event)
        # Remove audio alerts
        augmented["audio_events"] = [a for a in event.get("audio_events", [])
                                     if not a.get("is_alert")]
        # Reduce scene activity
        augmented["scene_activity"] = max(0, event.get("scene_activity", 0.5) - 0.3)
        return SyntheticSample(
            source_event_id=event.get("event_id", ""),
            augmentation=AugmentationType.COUNTERFACTUAL_MIRROR,
            event_data=augmented, label="normal",  # flipped label
            metadata={"original_label": label},
        )

    def _anomaly_injection(self, event: dict, label: str) -> SyntheticSample | None:
        """Inject anomaly pattern into a normal scene."""
        if not self._normal_pool:
            return None
        normal = dict(random.choice(self._normal_pool))
        # Inject the anomalous objects into the normal scene
        normal["objects"] = event.get("objects", [])
        normal["audio_events"] = event.get("audio_events", [])
        normal["scene_activity"] = event.get("scene_activity", 0.5)
        return SyntheticSample(
            source_event_id=event.get("event_id", ""),
            augmentation=AugmentationType.ANOMALY_INJECTION,
            event_data=normal, label=label,
            metadata={"injected_into": normal.get("event_id", "unknown")},
        )
