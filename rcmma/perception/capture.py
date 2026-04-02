import time
import cv2
from typing import Iterator, Tuple


class VideoSource:
    """Simple video source wrapper.

    Usage:
        src = VideoSource(0)  # webcam
        src = VideoSource('video.mp4')  # file

    frames() yields tuples: (timestamp, frame)
    """

    def __init__(self, source=0):
        self.source = source
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.source)
        return self.cap.isOpened()

    def frames(self) -> Iterator[Tuple[float, any]]:
        if self.cap is None:
            opened = self.open()
            if not opened:
                return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            ts = time.time()
            yield ts, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
