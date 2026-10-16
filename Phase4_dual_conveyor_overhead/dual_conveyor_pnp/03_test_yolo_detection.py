"""
03_test_yolo_detection.py

Robot YOLO detection test.

Purpose:
- Connect to the robot
- Move the robot to its home/view joint position
- Load the shared YOLO chip model
- Run live chip detection
- Draw bounding boxes, labels, confidence, center point, ROI, and trigger line
- No picking or placing
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from pyniryo import NiryoRobot, JointsPosition


# ============================================================
# ROBOT SETTINGS
# ============================================================

from common.settings import get as get_settings  # --robot / --conveyor

S = get_settings(need_conveyor=True)

ROBOT_NAME = S.ROBOT_NAME
ROBOT_IP = S.ROBOT_IP

HOME_VIEW_JOINTS = S.HOME_VIEW_JOINTS


# ============================================================
# PROJECT PATHS
# ============================================================

ROBOT_DIR = S.ROBOT_DIR
PROJECT_DIR = S.PROJECT_DIR
MODEL_PATH = PROJECT_DIR / "models" / "best.pt"


# ============================================================
# YOLO AND DISPLAY SETTINGS
# ============================================================

CONF_THRESHOLD = 0.75

ROI_X1 = 100
ROI_Y1 = 100
ROI_X2 = 550
ROI_Y2 = 430

TRIGGER_X = 320
TRIGGER_TOLERANCE = 20

WINDOW_NAME = f"{ROBOT_NAME} - YOLO Chip Detection"


# ============================================================
# DETECTION HELPERS
# ============================================================

def get_center(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def inside_roi(cx, cy):
    return ROI_X1 <= cx <= ROI_X2 and ROI_Y1 <= cy <= ROI_Y2


def crossed_trigger(cx):
    return abs(cx - TRIGGER_X) <= TRIGGER_TOLERANCE


def draw_guides(frame):
    cv2.rectangle(
        frame,
        (ROI_X1, ROI_Y1),
        (ROI_X2, ROI_Y2),
        (255, 0, 0),
        2,
    )

    cv2.line(
        frame,
        (TRIGGER_X, ROI_Y1),
        (TRIGGER_X, ROI_Y2),
        (0, 0, 255),
        2,
    )

    cv2.putText(
        frame,
        "ROI",
        (ROI_X1, ROI_Y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():
    robot = None

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find YOLO model:\n{MODEL_PATH}"
        )

    if HOME_VIEW_JOINTS is None:
        raise ValueError(
            f"{ROBOT_NAME} HOME_VIEW_JOINTS has not been entered yet."
        )

    print("=" * 70)
    print(f"{ROBOT_NAME} YOLO DETECTION TEST")
    print(f"Robot IP: {ROBOT_IP}")
    print(f"Model: {MODEL_PATH}")
    print("=" * 70)

    print("Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))
    print("YOLO model loaded.")

    try:
        print(f"Connecting to {ROBOT_NAME}...")
        robot = NiryoRobot(ROBOT_IP)
        print(f"Connected to {ROBOT_NAME}.")

        print(f"Moving {ROBOT_NAME} to home/view position...")
        robot.move(JointsPosition(*HOME_VIEW_JOINTS))
        print(f"{ROBOT_NAME} reached home/view position.")

        print("Starting camera detection.")
        print("Press Q in the camera window to quit.")

        while True:
            img_compressed = robot.get_img_compressed()

            if img_compressed is None:
                print("No compressed camera image received.")
                continue

            np_arr = np.frombuffer(
                img_compressed,
                dtype=np.uint8,
            )

            frame = cv2.imdecode(
                np_arr,
                cv2.IMREAD_COLOR,
            )

            if frame is None:
                print("Could not decode camera frame.")
                continue

            frame = draw_guides(frame)

            results = model(
                frame,
                conf=CONF_THRESHOLD,
                verbose=False,
            )

            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()

                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[cls_id]

                    x1, y1, x2, y2 = map(int, xyxy)
                    cx, cy = get_center(xyxy)

                    status = "OUTSIDE ROI"

                    if inside_roi(cx, cy):
                        status = "INSIDE ROI"

                    if inside_roi(cx, cy) and crossed_trigger(cx):
                        status = "TRIGGER"

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.circle(
                        frame,
                        (cx, cy),
                        5,
                        (0, 255, 255),
                        -1,
                    )

                    label = (
                        f"{class_name} "
                        f"{confidence:.2f} "
                        f"({cx},{cy}) "
                        f"{status}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Q pressed. Closing detection test.")
                break

    except KeyboardInterrupt:
        print("\nDetection test stopped from the terminal.")

    except Exception as error:
        print(f"{ROBOT_NAME} detection test failed.")
        print(f"Error: {error}")

    finally:
        cv2.destroyAllWindows()

        if robot is not None:
            try:
                robot.close_connection()
                print(f"Disconnected from {ROBOT_NAME}.")
            except Exception as error:
                print(f"Error while disconnecting: {error}")


if __name__ == "__main__":
    main()