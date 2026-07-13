"""
08_set_priority_point.py

Phase 2 helper script.

Purpose:
- Opens Niryo wrist camera stream
- Lets user click the priority pickup point
- Saves that point to phase2_config.json

During Phase 2 operation, the robot will choose the detected chip
closest to this saved point.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from pyniryo import NiryoRobot


ROBOT_IP = "192.168.0.199"

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "phase2_config.json"

priority_point = None
current_mouse = (0, 0)


def save_priority_point(x, y):
    data = {
        "priority_point": {
            "x": int(x),
            "y": int(y)
        }
    }

    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print("=" * 60)
    print(f"Saved priority point: ({x}, {y})")
    print(f"Saved to: {CONFIG_FILE}")
    print("=" * 60)


def mouse_callback(event, x, y, flags, param):
    global priority_point, current_mouse

    current_mouse = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        priority_point = (x, y)
        save_priority_point(x, y)


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

    if priority_point is not None:
        px, py = priority_point
        cv2.circle(display, (px, py), 8, (0, 255, 255), -1)
        cv2.putText(
            display,
            f"Priority Point: ({px}, {py})",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
    else:
        cv2.putText(
            display,
            "Click desired priority pickup point",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        display,
        "LEFT CLICK = save priority point | Q = quit",
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

    cv2.namedWindow("Set Phase 2 Priority Point")
    cv2.setMouseCallback("Set Phase 2 Priority Point", mouse_callback)

    try:
        while True:
            img_compressed = robot.get_img_compressed()
            np_arr = np.frombuffer(img_compressed, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Could not decode frame.")
                continue

            display = draw_interface(frame)
            cv2.imshow("Set Phase 2 Priority Point", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        robot.close_connection()
        print("Disconnected.")


if __name__ == "__main__":
    main()