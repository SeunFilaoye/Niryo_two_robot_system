"""
03_test_yolo_detection.py

Purpose:
- Connect to Niryo robot camera
- Load existing YOLO chip model
- Run live chip detection
- Draw bounding boxes, labels, confidence, center point, ROI, and trigger line
- NO robot movement
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from pyniryo import NiryoRobot


ROBOT_IP = "192.168.0.199"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

CONF_THRESHOLD = 0.75

ROI_X1 = 100
ROI_Y1 = 100
ROI_X2 = 550
ROI_Y2 = 430

TRIGGER_X = 320
TRIGGER_TOLERANCE = 20


def get_center(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def inside_roi(cx, cy):
    return ROI_X1 <= cx <= ROI_X2 and ROI_Y1 <= cy <= ROI_Y2


def crossed_trigger(cx):
    return abs(cx - TRIGGER_X) <= TRIGGER_TOLERANCE


def draw_guides(frame):
    cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 2)
    cv2.line(frame, (TRIGGER_X, ROI_Y1), (TRIGGER_X, ROI_Y2), (0, 0, 255), 2)
    cv2.putText(frame, "ROI", (ROI_X1, ROI_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find model: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    robot = NiryoRobot(ROBOT_IP)
    print("Connected to robot camera. Press Q to quit.")

    try:
        while True:
            img_compressed = robot.get_img_compressed()
            np_arr = np.frombuffer(img_compressed, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Could not decode frame.")
                continue

            frame = draw_guides(frame)
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)

            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls_id]

                    x1, y1, x2, y2 = map(int, xyxy)
                    cx, cy = get_center(xyxy)

                    status = "OUTSIDE ROI"
                    if inside_roi(cx, cy):
                        status = "INSIDE ROI"
                    if inside_roi(cx, cy) and crossed_trigger(cx):
                        status = "TRIGGER"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

                    label = f"{name} {conf:.2f} ({cx},{cy}) {status}"
                    cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("YOLO Chip Detection Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        robot.close_connection()


if __name__ == "__main__":
    main()