import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


from common.settings import get as get_settings  # --robot / --conveyor

S = get_settings(need_conveyor=True)

ROBOT_NAME = S.ROBOT_NAME

ROBOT_DIR = S.ROBOT_DIR
PROJECT_ROOT = S.PROJECT_DIR

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
PHASE2_CONFIG_FILE = ROBOT_DIR / "phase2_config.json"

ROI_COLOR = (255, 255, 0)

CLASS_COLORS = {
    "red_circle": (0, 0, 255),
    "red_square": (0, 0, 255),
    "blue_circle": (255, 0, 0),
    "blue_square": (255, 0, 0),
    "blue_sqaure": (255, 0, 0),
    "green_circle": (0, 255, 0),
    "green_square": (0, 255, 0),
}

DEFAULT_COLOR = (255, 255, 255)


class Detector:
    def __init__(self, confidence=0.75):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

        if not PHASE2_CONFIG_FILE.exists():
            raise FileNotFoundError(
                f"Missing Phase 2 configuration: {PHASE2_CONFIG_FILE}"
            )

        self.confidence = confidence
        self.model = YOLO(str(MODEL_PATH))
        self.roi_polygon = self.load_roi()

    def load_roi(self):
        with PHASE2_CONFIG_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        saved_robot_name = data.get("robot_name")

        if saved_robot_name not in (None, ROBOT_NAME):
            raise ValueError(
                f"Phase 2 config belongs to {saved_robot_name}, "
                f"not {ROBOT_NAME}."
            )

        points = data.get("polygon_roi")

        if not isinstance(points, list) or len(points) < 3:
            raise ValueError(
                "phase2_config.json must contain polygon_roi "
                "with at least 3 vertices."
            )

        polygon = np.array(
            [
                [int(point[0]), int(point[1])]
                for point in points
            ],
            dtype=np.int32,
        )

        return polygon.reshape((-1, 1, 2))

    @staticmethod
    def point_inside(polygon, x, y):
        return (
            cv2.pointPolygonTest(
                polygon,
                (float(x), float(y)),
                False,
            )
            >= 0
        )

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False,
        )

        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].cpu().numpy(),
                )

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if not self.point_inside(
                    self.roi_polygon,
                    center_x,
                    center_y,
                ):
                    continue

                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "cx": center_x,
                    "cy": center_y,
                })

        return detections

    def draw(self, frame, detections):
        overlay = frame.copy()

        cv2.fillPoly(
            overlay,
            [self.roi_polygon],
            ROI_COLOR,
        )

        cv2.addWeighted(
            overlay,
            0.10,
            frame,
            0.90,
            0,
            frame,
        )

        cv2.polylines(
            frame,
            [self.roi_polygon],
            True,
            ROI_COLOR,
            3,
        )

        label_point = tuple(self.roi_polygon[0][0])

        cv2.putText(
            frame,
            "FIFO POLYGON ROI",
            (label_point[0], max(24, label_point[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            ROI_COLOR,
            2,
        )

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]

            class_name = detection["class_name"]
            confidence = detection["confidence"]

            color = CLASS_COLORS.get(
                class_name,
                DEFAULT_COLOR,
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.circle(
                frame,
                (detection["cx"], detection["cy"]),
                5,
                color,
                -1,
            )

            cv2.putText(
                frame,
                f"{class_name} {confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
            )

        return frame

