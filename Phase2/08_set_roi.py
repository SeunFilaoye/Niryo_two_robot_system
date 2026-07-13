"""
08_set_roi.py

Purpose:
- Opens Niryo wrist camera stream.
- Lets user drag/select a Phase 2 ROI.
- Saves ROI coordinates to phase2_config.json.

This does NOT recalibrate the robot.
It only changes the image region where the system looks for chips.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from pyniryo import NiryoRobot


ROBOT_IP = "192.168.0.199"

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "phase2_config.json"

roi_start = None
roi_end = None
drawing = False
current_mouse = (0, 0)


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)

    return {}


def save_config(data):
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)


def save_roi(x1, y1, x2, y2):
    data = load_config()

    data["roi"] = {
        "x1": int(min(x1, x2)),
        "y1": int(min(y1, y2)),
        "x2": int(max(x1, x2)),
        "y2": int(max(y1, y2)),
    }

    save_config(data)

    print("=" * 60)
    print("Saved Phase 2 ROI:")
    print(data["roi"])
    print(f"Saved to: {CONFIG_FILE}")
    print("=" * 60)


def mouse_callback(event, x, y, flags, param):
    global roi_start, roi_end, drawing, current_mouse

    current_mouse = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        roi_end = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)
        drawing = False

        x1, y1 = roi_start
        x2, y2 = roi_end

        save_roi(x1, y1, x2, y2)


def draw_interface(frame):
    display = frame.copy()

    mx, my = current_mouse

    cv2.putText(
        display,
        f"Mouse: ({mx}, {my})",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        display,
        "Drag mouse to select Phase 2 ROI",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    if roi_start is not None and roi_end is not None:
        x1, y1 = roi_start
        x2, y2 = roi_end

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            3,
        )

    cv2.putText(
        display,
        "LEFT DRAG = set ROI | Q = quit",
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    return display


def main():
    print("Connecting to robot...")
    robot = NiryoRobot(ROBOT_IP)
    print("Connected.")

    cv2.namedWindow("Set Phase 2 ROI")
    cv2.setMouseCallback("Set Phase 2 ROI", mouse_callback)

    try:
        while True:
            img_compressed = robot.get_img_compressed()
            np_arr = np.frombuffer(img_compressed, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Could not decode frame.")
                continue

            display = draw_interface(frame)
            cv2.imshow("Set Phase 2 ROI", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        robot.close_connection()
        print("Disconnected.")


if __name__ == "__main__":
    main()