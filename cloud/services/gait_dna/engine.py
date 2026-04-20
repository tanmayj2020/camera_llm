"""Behavioral Biometric Fingerprinting (Gait DNA) — creates a unique
movement signature for each person that persists even if they change
clothes, wear a hat, or carry different objects.

Extracted features (from trajectory + pose keypoints):
  1. Stride length distribution (mean, std)
  2. Walking speed distribution
  3. Cadence (steps per second)
  4. Arm swing amplitude
  5. Shoulder sway ratio
  6. Head bob magnitude
  7. Turn radius preference
  8. Acceleration/deceleration profile
  9. Path curvature tendency (straight walker vs. weaver)
  10. Idle micro-movement signature (fidget pattern when stationary)

The fingerprint is a 32-dimensional embedding that can:
  - Re-identify a person who changed clothes between visits
  - Link the same person across days without face recognition
  - Work entirely on movement data (no appearance features)

Novel because: Existing Re-ID systems rely on visual appearance (clothing
color, body shape).  This uses ONLY movement kinematics — immune to
disguise, clothing change, or lighting variation.
"""

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Fingerprint dimensionality
FP_DIM = 32
# Minimum observations needed to compute a stable fingerprint
MIN_OBSERVATIONS = 30
# Window size for feature extraction
WINDOW_SIZE = 60  # frames

# Per-feature normalization scales (approximate physical max values).
# Equalizes contribution of each dimension before unit normalization
# so gait-shape features (cadence, autocorrelation) aren't overwhelmed
# by speed-magnitude features.
_FEATURE_SCALES = np.array([
    2.0,   # 0: stride length (m)
    1.0,   # 1: stride std
    5.0,   # 2: mean speed (m/s)
    2.0,   # 3: speed std
    5.0,   # 4: p25 speed
    5.0,   # 5: p75 speed
    3.0,   # 6: cadence (Hz)
    1.0,   # 7: cadence std
    10.0,  # 8: accel mean
    10.0,  # 9: accel std
    1.0,   # 10: turn rate mean
    1.0,   # 11: turn rate std
    1.0,   # 12: path straightness
    3.0,   # 13: direction entropy
    1.0,   # 14: pause freq
    5.0,   # 15: pause duration
    0.5,   # 16: micro-movement
    1.0,   # 17: autocorrelation lag-1
    1.0,   # 18: autocorrelation lag-3
    3.14,  # 19: heading skew
    2.0,   # 20: speed CoV
    10.0,  # 21: step regularity
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # 22-31: pose
], dtype=np.float32)


@dataclass
class GaitFingerprint:
    """A person's unique movement signature."""
    entity_id: str
    fingerprint: np.ndarray       # shape (32,)
    confidence: float             # 0.0–1.0 (higher = more observations)
    observation_count: int
    first_seen: float
    last_updated: float
    feature_names: list[str] = field(default_factory=list)


@dataclass
class GaitMatch:
    """A match between a current entity and a stored fingerprint."""
    query_entity_id: str
    matched_entity_id: str
    similarity: float             # 0.0–1.0
    match_type: str               # "strong" | "moderate" | "weak"


class GaitDNAEngine:
    """Extracts and matches behavioral biometric fingerprints from
    trajectory and pose data."""

    FEATURE_NAMES = [
        "stride_length_mean", "stride_length_std",
        "speed_mean", "speed_std", "speed_p25", "speed_p75",
        "cadence_mean", "cadence_std",
        "accel_mean", "accel_std",
        "turn_rate_mean", "turn_rate_std",
        "path_straightness",      # ratio of displacement to distance
        "direction_entropy",      # how unpredictable the heading changes are
        "pause_frequency",        # how often they stop
        "pause_duration_mean",    # avg stop duration
        "micro_movement_energy",  # fidget intensity when stationary
        "velocity_autocorr_1",    # lag-1 speed autocorrelation (rhythmicity)
        "velocity_autocorr_3",    # lag-3 speed autocorrelation
        "heading_change_skew",    # left vs right turn preference
        "speed_variability_ratio",  # CoV of speed
        "step_regularity",        # periodicity in position signal
        # Pose-based features (if available)
        "arm_swing_amplitude",
        "shoulder_sway",
        "head_bob",
        "knee_lift_asymmetry",
        "torso_lean",
        "hip_sway",
        "elbow_angle_mean",
        "ankle_clearance",
        "gait_symmetry",         # left-right symmetry score
        "overall_rhythm",        # FFT dominant frequency of speed signal
    ]

    def __init__(self):
        # entity_id -> deque of (timestamp, x, y, vx, vy, pose_keypoints?)
        self._observations: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=500))
        # entity_id -> GaitFingerprint
        self._gallery: dict[str, GaitFingerprint] = {}
        # Similarity threshold for matching
        self._strong_threshold = 0.85
        self._moderate_threshold = 0.70

    # ── Public API ────────────────────────────────────────────────────────

    def observe(self, entity_id: str, x: float, y: float,
                vx: float = 0.0, vy: float = 0.0,
                timestamp: float | None = None,
                pose_keypoints: list | None = None):
        """Record one observation of an entity's position and velocity."""
        t = timestamp or time.time()
        self._observations[entity_id].append({
            "t": t, "x": x, "y": y, "vx": vx, "vy": vy,
            "pose": pose_keypoints,
        })

        # Auto-compute fingerprint when enough data
        obs = self._observations[entity_id]
        if len(obs) >= MIN_OBSERVATIONS and len(obs) % 10 == 0:
            self._update_fingerprint(entity_id)

    def get_fingerprint(self, entity_id: str) -> GaitFingerprint | None:
        """Get the computed fingerprint for an entity."""
        return self._gallery.get(entity_id)

    def match(self, entity_id: str) -> list[GaitMatch]:
        """Match an entity's fingerprint against the gallery.

        Returns sorted list of matches (best first).
        """
        fp = self._gallery.get(entity_id)
        if fp is None:
            return []

        matches = []
        for gid, gfp in self._gallery.items():
            if gid == entity_id:
                continue
            if gfp.confidence < 0.3:
                continue

            sim = self._cosine_similarity(fp.fingerprint, gfp.fingerprint)

            if sim >= self._moderate_threshold:
                match_type = "strong" if sim >= self._strong_threshold else "moderate"
                matches.append(GaitMatch(
                    query_entity_id=entity_id,
                    matched_entity_id=gid,
                    similarity=round(sim, 3),
                    match_type=match_type,
                ))

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:10]  # top 10

    def identify_across_sessions(self, entity_id: str) -> GaitMatch | None:
        """Check if this entity matches anyone from a previous session
        (e.g., same person visiting on different days)."""
        matches = self.match(entity_id)
        return matches[0] if matches else None

    def get_gallery_size(self) -> int:
        return len(self._gallery)

    def get_gallery_summary(self) -> list[dict]:
        return [
            {
                "entity_id": fp.entity_id,
                "confidence": round(fp.confidence, 2),
                "observations": fp.observation_count,
                "first_seen": fp.first_seen,
                "last_updated": fp.last_updated,
            }
            for fp in self._gallery.values()
        ]

    # ── Feature Extraction ────────────────────────────────────────────────

    def _update_fingerprint(self, entity_id: str):
        """Extract features and update the fingerprint for an entity."""
        obs_list = list(self._observations[entity_id])
        if len(obs_list) < MIN_OBSERVATIONS:
            return

        features = np.zeros(FP_DIM, dtype=np.float32)

        # Extract trajectory features
        positions = np.array([(o["x"], o["y"]) for o in obs_list])
        times = np.array([o["t"] for o in obs_list])
        velocities = np.array([(o["vx"], o["vy"]) for o in obs_list])

        # Compute speeds
        speeds = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)
        # Filter out stationary periods for gait analysis
        moving_mask = speeds > 0.1  # m/s threshold

        if moving_mask.sum() < 10:
            return  # not enough movement data

        moving_speeds = speeds[moving_mask]

        # 0-1: Stride length (estimated from speed peaks)
        features[0], features[1] = self._stride_length(speeds, times)

        # 2-5: Speed distribution
        features[2] = np.mean(moving_speeds)
        features[3] = np.std(moving_speeds)
        features[4] = np.percentile(moving_speeds, 25)
        features[5] = np.percentile(moving_speeds, 75)

        # 6-7: Cadence (steps per second from FFT)
        features[6], features[7] = self._cadence(speeds[moving_mask],
                                                   times[moving_mask])

        # 8-9: Acceleration
        accels = np.diff(speeds) / np.maximum(np.diff(times), 1e-6)
        features[8] = np.mean(np.abs(accels))
        features[9] = np.std(accels)

        # 10-11: Turn rate
        headings = np.arctan2(velocities[:, 1], velocities[:, 0])
        heading_changes = np.diff(headings)
        # Wrap to [-pi, pi]
        heading_changes = (heading_changes + np.pi) % (2 * np.pi) - np.pi
        features[10] = np.mean(np.abs(heading_changes[moving_mask[1:]]))
        features[11] = np.std(heading_changes[moving_mask[1:]])

        # 12: Path straightness
        features[12] = self._path_straightness(positions)

        # 13: Direction entropy
        features[13] = self._direction_entropy(heading_changes)

        # 14-15: Pause patterns
        features[14], features[15] = self._pause_analysis(speeds, times)

        # 16: Micro-movement energy (when stationary)
        stationary_mask = ~moving_mask
        if stationary_mask.sum() > 5:
            stationary_pos = positions[stationary_mask]
            micro = np.std(stationary_pos, axis=0)
            features[16] = float(np.linalg.norm(micro))
        else:
            features[16] = 0.0

        # 17-18: Speed autocorrelation (rhythmicity)
        features[17] = self._autocorrelation(moving_speeds, lag=1)
        features[18] = self._autocorrelation(moving_speeds, lag=3)

        # 19: Heading change skew (left vs right preference)
        if len(heading_changes) > 5:
            features[19] = float(np.mean(heading_changes[moving_mask[1:]]))

        # 20: Speed CoV
        features[20] = features[3] / max(features[2], 1e-6)

        # 21: Step regularity (FFT peak prominence)
        features[21] = self._step_regularity(speeds, times)

        # 22-31: Pose-based features (if available)
        poses = [o["pose"] for o in obs_list if o["pose"] is not None]
        if len(poses) >= 10:
            pose_features = self._extract_pose_features(poses)
            features[22:32] = pose_features[:10]

        # Per-feature scaling then unit normalize
        features /= _FEATURE_SCALES
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        obs_count = len(obs_list)
        confidence = min(1.0, obs_count / 200)  # saturates at 200 obs

        self._gallery[entity_id] = GaitFingerprint(
            entity_id=entity_id,
            fingerprint=features,
            confidence=confidence,
            observation_count=obs_count,
            first_seen=obs_list[0]["t"],
            last_updated=obs_list[-1]["t"],
            feature_names=self.FEATURE_NAMES[:FP_DIM],
        )

    def _stride_length(self, speeds: np.ndarray, times: np.ndarray) -> tuple:
        """Estimate stride length from speed periodicity."""
        if len(speeds) < 20:
            return (0.0, 0.0)
        # Approximate: stride ≈ speed * cadence_period
        # Find peaks in speed signal
        mean_speed = np.mean(speeds[speeds > 0.1])
        # Rough stride estimate (average human ~0.7m stride)
        stride_est = mean_speed * 0.5  # rough heuristic
        stride_std = np.std(speeds[speeds > 0.1]) * 0.3
        return (float(stride_est), float(stride_std))

    def _cadence(self, speeds: np.ndarray, times: np.ndarray) -> tuple:
        """Estimate walking cadence using FFT on speed signal."""
        if len(speeds) < 20:
            return (0.0, 0.0)

        # Resample to uniform time grid
        dt = np.mean(np.diff(times))
        if dt <= 0:
            return (0.0, 0.0)

        fft = np.abs(np.fft.rfft(speeds - np.mean(speeds)))
        freqs = np.fft.rfftfreq(len(speeds), d=dt)

        # Look for cadence in 1.0–3.0 Hz range (walking = ~2 Hz)
        mask = (freqs >= 1.0) & (freqs <= 3.0)
        if not mask.any():
            return (0.0, 0.0)

        peak_idx = np.argmax(fft[mask])
        cadence = float(freqs[mask][peak_idx])
        # Variance: spectral spread around peak
        cadence_std = float(np.std(fft[mask])) / max(float(np.max(fft[mask])), 1e-6)

        return (cadence, cadence_std)

    def _path_straightness(self, positions: np.ndarray) -> float:
        """Ratio of displacement to total distance traveled."""
        if len(positions) < 2:
            return 1.0
        displacements = np.diff(positions, axis=0)
        distances = np.sqrt(np.sum(displacements ** 2, axis=1))
        total_distance = np.sum(distances)
        if total_distance < 0.01:
            return 1.0
        net_displacement = np.linalg.norm(positions[-1] - positions[0])
        return float(net_displacement / total_distance)

    def _direction_entropy(self, heading_changes: np.ndarray) -> float:
        """Shannon entropy of heading change distribution."""
        if len(heading_changes) < 5:
            return 0.0
        # Discretize into 8 bins
        bins = np.linspace(-np.pi, np.pi, 9)
        hist, _ = np.histogram(heading_changes, bins=bins)
        probs = hist / max(hist.sum(), 1)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def _pause_analysis(self, speeds: np.ndarray,
                        times: np.ndarray) -> tuple:
        """Analyze pause frequency and duration."""
        is_paused = speeds < 0.1
        if not is_paused.any():
            return (0.0, 0.0)

        # Count pause events (transitions from moving to paused)
        transitions = np.diff(is_paused.astype(int))
        pause_starts = np.where(transitions == 1)[0]
        pause_ends = np.where(transitions == -1)[0]

        total_time = times[-1] - times[0]
        if total_time <= 0:
            return (0.0, 0.0)

        freq = len(pause_starts) / max(total_time, 1)

        # Average pause duration
        durations = []
        for s in pause_starts:
            ends_after = pause_ends[pause_ends > s]
            if len(ends_after) > 0:
                durations.append(times[ends_after[0]] - times[s])

        avg_dur = float(np.mean(durations)) if durations else 0.0
        return (float(freq), avg_dur)

    def _autocorrelation(self, signal: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at given lag."""
        if len(signal) <= lag:
            return 0.0
        mean = np.mean(signal)
        var = np.var(signal)
        if var < 1e-10:
            return 0.0
        shifted = signal[lag:] - mean
        original = signal[:-lag] - mean
        return float(np.mean(shifted * original) / var)

    def _step_regularity(self, speeds: np.ndarray,
                         times: np.ndarray) -> float:
        """Measure regularity of gait from FFT peak prominence."""
        if len(speeds) < 20:
            return 0.0
        dt = np.mean(np.diff(times))
        if dt <= 0:
            return 0.0
        fft = np.abs(np.fft.rfft(speeds - np.mean(speeds)))
        if len(fft) < 3:
            return 0.0
        peak = np.max(fft[1:])  # skip DC
        mean_energy = np.mean(fft[1:])
        return float(peak / max(mean_energy, 1e-6))

    def _extract_pose_features(self, poses: list) -> np.ndarray:
        """Extract gait features from COCO-format pose keypoints.

        COCO keypoints: 17 points × (x, y, conf) = 51 values
        """
        features = np.zeros(10, dtype=np.float32)

        try:
            kps = np.array([p[:51] for p in poses if len(p) >= 51])
            if len(kps) < 10:
                return features
            kps = kps.reshape(-1, 17, 3)  # (T, 17, 3)

            # Keypoint indices (COCO):
            # 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow
            # 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip
            # 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
            # 0=nose

            # 22: Arm swing amplitude
            l_wrist = kps[:, 9, :2]
            r_wrist = kps[:, 10, :2]
            arm_swing_l = np.std(l_wrist[:, 1])  # vertical swing
            arm_swing_r = np.std(r_wrist[:, 1])
            features[0] = (arm_swing_l + arm_swing_r) / 2

            # 23: Shoulder sway
            l_shoulder = kps[:, 5, :2]
            r_shoulder = kps[:, 6, :2]
            shoulder_width = np.linalg.norm(l_shoulder - r_shoulder, axis=1)
            features[1] = float(np.std(shoulder_width))

            # 24: Head bob
            nose = kps[:, 0, :2]
            features[2] = float(np.std(nose[:, 1]))  # vertical oscillation

            # 25: Knee lift asymmetry
            l_knee_y = kps[:, 13, 1]
            r_knee_y = kps[:, 14, 1]
            features[3] = abs(float(np.std(l_knee_y) - np.std(r_knee_y)))

            # 26: Torso lean
            mid_shoulder = (l_shoulder + r_shoulder) / 2
            l_hip = kps[:, 11, :2]
            r_hip = kps[:, 12, :2]
            mid_hip = (l_hip + r_hip) / 2
            lean = mid_shoulder[:, 0] - mid_hip[:, 0]  # lateral lean
            features[4] = float(np.std(lean))

            # 27: Hip sway
            features[5] = float(np.std(np.linalg.norm(l_hip - r_hip, axis=1)))

            # 28: Elbow angle mean
            l_elbow = kps[:, 7, :2]
            r_elbow = kps[:, 8, :2]
            l_angle = self._compute_angle(l_shoulder, l_elbow, l_wrist)
            r_angle = self._compute_angle(r_shoulder, r_elbow, r_wrist)
            features[6] = (np.mean(l_angle) + np.mean(r_angle)) / 2

            # 29: Ankle clearance
            l_ankle = kps[:, 15, :2]
            r_ankle = kps[:, 16, :2]
            ankle_y = np.concatenate([l_ankle[:, 1], r_ankle[:, 1]])
            features[7] = float(np.std(ankle_y))

            # 30: Gait symmetry (correlation between left and right leg)
            if len(l_knee_y) > 5:
                corr = np.corrcoef(l_knee_y, r_knee_y)[0, 1]
                features[8] = float(corr) if not np.isnan(corr) else 0.0

            # 31: Overall rhythm (FFT of vertical nose movement)
            if len(nose) > 20:
                fft = np.abs(np.fft.rfft(nose[:, 1] - np.mean(nose[:, 1])))
                if len(fft) > 1:
                    features[9] = float(np.argmax(fft[1:]) + 1)

        except Exception as e:
            logger.debug("Pose feature extraction failed: %s", e)

        return features

    @staticmethod
    def _compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Compute angle at point b formed by segments a-b and b-c."""
        ba = a - b
        bc = c - b
        dot = np.sum(ba * bc, axis=1)
        mag_ba = np.linalg.norm(ba, axis=1)
        mag_bc = np.linalg.norm(bc, axis=1)
        cos_angle = dot / (mag_ba * mag_bc + 1e-8)
        return np.arccos(np.clip(cos_angle, -1, 1))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(dot / (na * nb))
