"""Audio-Visual Fusion — correlates what is heard with where it's seen."""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Audio events that have known visual correlates
_VISUAL_CORRELATES = {
    "scream": {"visual_cues": ["person", "running", "falling"], "false_positive_cues": []},
    "gunshot": {"visual_cues": ["person", "running"], "false_positive_cues": ["construction", "nail_gun", "hammer"]},
    "glass_break": {"visual_cues": ["person", "broken_glass"], "false_positive_cues": ["construction", "recycling"]},
    "alarm": {"visual_cues": ["person", "running", "door"], "false_positive_cues": ["test", "drill"]},
    "explosion": {"visual_cues": ["smoke", "fire", "person", "running"], "false_positive_cues": ["construction", "demolition"]},
    "dog_bark": {"visual_cues": ["dog", "animal"], "false_positive_cues": []},
    "vehicle_horn": {"visual_cues": ["car", "truck", "bus"], "false_positive_cues": []},
}


@dataclass
class AudioVisualCorrelation:
    audio_event: str
    visual_evidence: list[str]
    correlation_type: str  # confirmed, contradicted, unresolved
    fused_confidence: float
    description: str


class AudioVisualFusion:
    """Fuses audio events with visual scene state to confirm, contradict, or enrich alerts."""

    def __init__(self, spatial=None):
        self._spatial = spatial

    def fuse(self, audio_events: list[dict], scene_state: dict, spatial=None) -> list[AudioVisualCorrelation]:
        spatial = spatial or scene_state.get("spatial") or self._spatial
        if not audio_events:
            return []

        # Gather visual context
        objects = scene_state.get("objects", [])
        obj_classes = {o.get("class_name", "").lower() for o in objects}

        # Add spatial entity classes
        if spatial:
            for ent in getattr(spatial, "_entities", {}).values():
                obj_classes.add(ent.class_name.lower())

        # Check for motion indicators
        has_running = False
        if spatial:
            for ent in getattr(spatial, "_entities", {}).values():
                speed = float((ent.velocity ** 2).sum() ** 0.5) if hasattr(ent, "velocity") else 0
                if speed > 2.0 and ent.class_name == "person":
                    has_running = True
                    break
        if has_running:
            obj_classes.add("running")

        results = []
        for audio in audio_events:
            event_type = audio.get("class", audio.get("event_type", "unknown")).lower()
            audio_conf = audio.get("confidence", 0.5)
            correlates = _VISUAL_CORRELATES.get(event_type)

            if not correlates:
                results.append(AudioVisualCorrelation(
                    event_type, list(obj_classes)[:5], "unresolved", audio_conf,
                    f"{event_type} detected — no visual correlation model available",
                ))
                continue

            # Check for false positive indicators
            fp_matches = [c for c in correlates["false_positive_cues"] if c in obj_classes]
            if fp_matches:
                results.append(AudioVisualCorrelation(
                    event_type, fp_matches, "contradicted", max(0.1, audio_conf * 0.3),
                    f"{event_type} likely false positive — visual shows {', '.join(fp_matches)}",
                ))
                continue

            # Check for confirming visual evidence
            confirmed = [c for c in correlates["visual_cues"] if c in obj_classes]
            if confirmed:
                fused = min(0.99, audio_conf * (1 + 0.2 * len(confirmed)))
                results.append(AudioVisualCorrelation(
                    event_type, confirmed, "confirmed", fused,
                    f"{event_type} confirmed by visual: {', '.join(confirmed)}",
                ))
            else:
                results.append(AudioVisualCorrelation(
                    event_type, list(obj_classes)[:5], "unresolved", audio_conf * 0.8,
                    f"{event_type} detected but no confirming visual evidence",
                ))

        return results
