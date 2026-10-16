"""Wrist-camera YOLO detection and the permanent FIFO used in Phase 3."""

from __future__ import annotations

import math
import time

import cv2

CLASS_COLORS = {
    "red_circle": (0, 0, 255),
    "red_square": (0, 0, 255),
    "blue_circle": (255, 0, 0),
    "blue_square": (255, 0, 0),
    "blue_sqaure": (255, 0, 0),
    "green_circle": (0, 255, 0),
    "green_square": (0, 255, 0),
}


class WristDetector:
    def __init__(self, model_path, confidence: float = 0.75):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def detect(self, frame, station=None) -> list[dict]:
        """Detections whose center is inside the station ROI (if given)."""
        result = self.model(frame, conf=self.confidence, verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if station is not None and not station.inside_roi(cx, cy):
                continue
            detections.append({
                "class_name": self.model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": (x1, y1, x2, y2),
                "cx": cx,
                "cy": cy,
            })
        return detections


def draw_detections(frame, detections) -> None:
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS.get(det["class_name"], (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (det["cx"], det["cy"]), 4, color, -1)
        cv2.putText(frame, f"{det['class_name']} {det['confidence']:.2f}",
                    (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)


class SimpleFIFO:
    """Same rules as the original Script 18.

    A chip seen inside the ROI is appended once, its dwell timer starts once
    and never resets, only the front item may be picked, and it is removed
    only after a successful pick-and-place.
    """

    def __init__(self, ready_delay_sec: float = 4.0):
        self.ready_delay_sec = ready_delay_sec
        self.items: list[dict] = []
        self.next_id = 1

    def active_items(self) -> list[dict]:
        return [item for item in self.items if item["status"] == "waiting"]

    def clear(self) -> None:
        self.items.clear()

    @staticmethod
    def _distance(item, det) -> float:
        return math.hypot(item["cx"] - det["cx"], item["cy"] - det["cy"])

    def _add(self, det) -> None:
        now = time.monotonic()
        self.items.append({
            **det,
            "id": self.next_id,
            "registered_at": now,
            "ready_at": now + self.ready_delay_sec,
            "visible": True,
            "status": "waiting",
        })
        print(f"FIFO ADD #{self.next_id} {det['class_name']}")
        self.next_id += 1

    def update(self, detections: list[dict]) -> None:
        active = self.active_items()
        for item in active:
            item["visible"] = False
        classes = {det["class_name"] for det in detections}
        for class_name in classes:
            items = [item for item in active if item["class_name"] == class_name]
            dets = [det for det in detections if det["class_name"] == class_name]
            pairs = sorted(
                (self._distance(item, det), i, j)
                for i, item in enumerate(items)
                for j, det in enumerate(dets)
            )
            used_items, used_dets = set(), set()
            for _, i, j in pairs:
                if i in used_items or j in used_dets:
                    continue
                items[i].update({k: dets[j][k] for k in ("confidence", "bbox", "cx", "cy")})
                items[i]["visible"] = True
                used_items.add(i)
                used_dets.add(j)
            for j, det in enumerate(dets):
                if j not in used_dets:
                    self._add(det)

    def front(self) -> dict | None:
        active = self.active_items()
        return active[0] if active else None

    def ready(self) -> dict | None:
        front = self.front()
        if front and front["visible"] and time.monotonic() >= front["ready_at"]:
            return front
        return None

    @staticmethod
    def remaining(item) -> float:
        return max(0.0, item["ready_at"] - time.monotonic())

    @staticmethod
    def complete(item) -> None:
        item["status"] = "complete"
        print(f"FIFO COMPLETE #{item['id']} {item['class_name']}")

    @staticmethod
    def failed(item) -> None:
        item["visible"] = False
        print(f"FIFO PICK FAILED, retained #{item['id']} {item['class_name']}")


def draw_fifo(frame, fifo: SimpleFIFO) -> None:
    front = fifo.front()
    for item in fifo.active_items():
        if not item["visible"]:
            continue
        x1, y1, x2, y2 = item["bbox"]
        remaining = fifo.remaining(item)
        if item is front:
            color = (0, 165, 255) if remaining == 0 else (0, 255, 255)
            tag = "READY" if remaining == 0 else f"FRONT {remaining:.1f}s"
        else:
            color, tag = (255, 255, 255), f"QUEUED {remaining:.1f}s"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"#{item['id']} {tag}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
