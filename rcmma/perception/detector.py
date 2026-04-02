"""Lightweight detection module.

This file contains a MockDetector that uses simple background subtraction
and contour extraction to produce bounding boxes. It's a placeholder so the
knowledge graph and pipeline can be developed without heavy ML models.

Replace or extend this class with a real detector (YOLO/Detectron/etc.) later.
"""
from typing import List, Dict, Tuple
import cv2
import numpy as np


class MockDetector:
    """Detects moving objects using background subtraction and contours.

    Methods
    -------
    detect(frame) -> List[dict]
        Returns list of detections: {id,label,bbox,confidence}
    """

    def __init__(self, min_area: int = 500):
        self.bs = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25)
        self._next_id = 1
        self.min_area = min_area

    def detect(self, frame) -> List[Dict]:
        """Return detections in a frame.

        bbox is (x, y, w, h). confidence is a heuristic based on contour area.
        """
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg = self.bs.apply(gray)
        # Clean up
        fg = cv2.medianBlur(fg, 5)
        _, th = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            det = {
                "id": f"obj_{self._next_id}",
                "label": "object",
                "bbox": (int(x), int(y), int(w), int(h)),
                "confidence": float(min(0.99, area / (frame.shape[0] * frame.shape[1]))),
            }
            self._next_id += 1
            detections.append(det)

        return detections
