"""Mock video source for development without a real camera."""

import time

import cv2


class MockSource:
    """Reads a video file and serves frames at native fps, optionally looping."""

    def __init__(self, video_file: str, loop: bool = True):
        self.video_file = video_file
        self.loop = loop
        self._cap = cv2.VideoCapture(video_file)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_file}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._interval = 1.0 / self.fps
        self._last_read = 0.0

    def get_frame(self):
        """Returns (frame, fps) or None if video ended and loop=False."""
        now = time.time()
        elapsed = now - self._last_read
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)

        ret, frame = self._cap.read()
        if not ret:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return None
            else:
                return None
        self._last_read = time.time()
        return frame

    def release(self):
        self._cap.release()
