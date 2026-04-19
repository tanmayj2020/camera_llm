"""Pose-based activity recognition — classifies actions from pose keypoint sequences.

Detects: fighting, falling, running via geometric analysis of pose sequences.
No ML model needed — pure kinematic + pose geometry on sliding windows.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# COCO keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10
L_HIP, R_HIP = 11, 12

WINDOW = 15


@dataclass
class ActivityEvent:
    track_id: int
    activity: str
    confidence: float
    timestamp: float


def _kpt(kpts, idx):
    if idx >= len(kpts):
        return None
    kp = kpts[idx]
    if isinstance(kp, (list, tuple)) and len(kp) >= 3 and kp[2] > 0.3:
        return np.array([kp[0], kp[1]])
    return None


def _angle(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


class ActivityRecognizer:
    """Classifies human activities from sliding window of pose keypoints per track."""

    def __init__(self):
        self._kpt_buf: dict[int, deque] = defaultdict(lambda: deque(maxlen=WINDOW))
        self._bbox_buf: dict[int, deque] = defaultdict(lambda: deque(maxlen=WINDOW))
        self._ts_buf: dict[int, deque] = defaultdict(lambda: deque(maxlen=WINDOW))

    def update(self, track_id: int, keypoints: list, bbox: list, timestamp: float):
        self._kpt_buf[track_id].append(keypoints)
        self._bbox_buf[track_id].append(bbox)
        self._ts_buf[track_id].append(timestamp)

    def classify(self, track_id: int) -> ActivityEvent | None:
        buf = self._kpt_buf.get(track_id)
        bboxes = self._bbox_buf.get(track_id)
        ts = self._ts_buf.get(track_id)
        if not buf or len(buf) < 5:
            return None

        results = []
        for fn in (self._detect_fall, self._detect_running, self._detect_fighting):
            r = fn(buf, bboxes, ts)
            if r:
                results.append(r)

        if not results:
            return None
        best = max(results, key=lambda r: r.confidence)
        return ActivityEvent(track_id, best.activity, best.confidence, ts[-1])

    def _detect_fall(self, buf, bboxes, ts) -> ActivityEvent | None:
        recent = list(bboxes)[-5:]
        ratios = [(b[3] - b[1]) / (b[2] - b[0] + 1e-6) for b in recent]
        if ratios[0] > 1.2 and ratios[-1] < 1.0:
            return ActivityEvent(0, "falling", min(0.95, ratios[0] - ratios[-1]), ts[-1])
        kpts = buf[-1]
        hip, shoulder = _kpt(kpts, L_HIP), _kpt(kpts, L_SHOULDER)
        if hip is not None and shoulder is not None and abs(hip[1] - shoulder[1]) < 30:
            return ActivityEvent(0, "falling", 0.7, ts[-1])
        return None

    def _detect_running(self, buf, bboxes, ts) -> ActivityEvent | None:
        recent_b, recent_t = list(bboxes)[-5:], list(ts)[-5:]
        dt = recent_t[-1] - recent_t[0]
        if dt <= 0:
            return None
        c0 = ((recent_b[0][0] + recent_b[0][2]) / 2, (recent_b[0][1] + recent_b[0][3]) / 2)
        c1 = ((recent_b[-1][0] + recent_b[-1][2]) / 2, (recent_b[-1][1] + recent_b[-1][3]) / 2)
        speed = ((c1[0] - c0[0])**2 + (c1[1] - c0[1])**2)**0.5 / dt
        if speed > 150:
            return ActivityEvent(0, "running", min(0.9, speed / 300), ts[-1])
        return None

    def _detect_fighting(self, buf, ts) -> ActivityEvent | None:
        if len(buf) < 8:
            return None
        arm_ext, wrist_fast = 0, 0
        total_arm, total_wrist = 0, 0
        for i in range(max(0, len(buf) - 8), len(buf)):
            kpts = buf[i]
            for s_idx, e_idx, w_idx in [(L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)]:
                s, e, w = _kpt(kpts, s_idx), _kpt(kpts, e_idx), _kpt(kpts, w_idx)
                if s is not None and e is not None and w is not None:
                    total_arm += 1
                    if _angle(s, e, w) > 140:
                        arm_ext += 1
            if i > 0:
                lw_now, lw_prev = _kpt(kpts, L_WRIST), _kpt(buf[i - 1], L_WRIST)
                if lw_now is not None and lw_prev is not None:
                    total_wrist += 1
                    dt = max(ts[i] - ts[i - 1], 0.01) if i < len(ts) else 0.033
                    if float(np.linalg.norm(lw_now - lw_prev)) / dt > 200:
                        wrist_fast += 1

        ext_r = arm_ext / max(total_arm, 1)
        fast_r = wrist_fast / max(total_wrist, 1)
        if ext_r > 0.3 and fast_r > 0.3:
            return ActivityEvent(0, "fighting", min(0.85, (ext_r + fast_r) / 2), ts[-1])
        return None

    def classify_all(self, timestamp: float) -> list[ActivityEvent]:
        events = []
        for tid in list(self._kpt_buf.keys()):
            r = self.classify(tid)
            if r:
                r.track_id = tid
                events.append(r)
        return events

    def cleanup(self, max_age_s: float = 30.0):
        now = time.time()
        stale = [tid for tid, ts in self._ts_buf.items() if ts and now - ts[-1] > max_age_s]
        for tid in stale:
            del self._kpt_buf[tid]
            del self._bbox_buf[tid]
            del self._ts_buf[tid]
