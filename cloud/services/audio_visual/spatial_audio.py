"""Spatial Audio Localization — triangulate sound source from multi-mic arrays."""

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SPEED_OF_SOUND = 343.0  # m/s at 20°C


@dataclass
class AudioLocation:
    sound_class: str
    estimated_position: tuple[float, float]  # (x, y) in meters
    confidence: float
    bearing_deg: float
    camera_id: str


class SpatialAudioLocalizer:
    """Triangulates sound source position from time-difference-of-arrival (TDOA) across mics."""

    def __init__(self):
        self._mic_positions: dict[str, tuple[float, float]] = {}  # mic_id -> (x, y)

    def register_mic(self, mic_id: str, x: float, y: float):
        self._mic_positions[mic_id] = (x, y)

    def localize(self, tdoa_pairs: list[dict], sound_class: str = "unknown",
                 camera_id: str = "") -> AudioLocation | None:
        """Localize sound from TDOA measurements.

        tdoa_pairs: [{"mic_a": id, "mic_b": id, "tdoa_s": float}, ...]
        """
        if len(self._mic_positions) < 2 or len(tdoa_pairs) < 1:
            return None

        # For 2 mics: compute bearing from TDOA
        if len(tdoa_pairs) == 1:
            pair = tdoa_pairs[0]
            pos_a = np.array(self._mic_positions.get(pair["mic_a"], (0, 0)))
            pos_b = np.array(self._mic_positions.get(pair["mic_b"], (0, 0)))
            mic_dist = float(np.linalg.norm(pos_b - pos_a))
            if mic_dist < 0.01:
                return None
            tdoa = pair["tdoa_s"]
            ratio = np.clip(tdoa * SPEED_OF_SOUND / mic_dist, -1, 1)
            angle = math.acos(ratio)
            bearing = math.degrees(angle)
            midpoint = (pos_a + pos_b) / 2
            direction = np.array([math.cos(math.radians(bearing)), math.sin(math.radians(bearing))])
            estimated = midpoint + direction * 5.0  # 5m estimate along bearing
            return AudioLocation(sound_class, tuple(estimated.tolist()), 0.5, round(bearing, 1), camera_id)

        # For 3+ mics: least-squares TDOA multilateration
        try:
            positions = []
            for pair in tdoa_pairs:
                pa = np.array(self._mic_positions.get(pair["mic_a"], (0, 0)))
                pb = np.array(self._mic_positions.get(pair["mic_b"], (0, 0)))
                d_diff = pair["tdoa_s"] * SPEED_OF_SOUND
                positions.append((pa, pb, d_diff))

            # Linearized least squares
            ref_a, ref_b, ref_d = positions[0]
            A_rows, b_rows = [], []
            for pa, pb, d_diff in positions[1:]:
                A_rows.append(2 * (pa - ref_a))
                b_rows.append(np.dot(pa, pa) - np.dot(ref_a, ref_a) - (d_diff**2 - ref_d**2))

            if not A_rows:
                return None
            A = np.array(A_rows)
            b = np.array(b_rows)
            pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            bearing = math.degrees(math.atan2(pos[1], pos[0]))
            return AudioLocation(sound_class, tuple(pos.tolist()), 0.7, round(bearing, 1), camera_id)
        except Exception as e:
            logger.debug("Audio localization failed: %s", e)
            return None
