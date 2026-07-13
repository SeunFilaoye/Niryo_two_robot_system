"""
05_test_pixel_to_robot.py

Purpose:
- Load saved homography calibration
- Show robot camera feed
- Click any image point
- Print estimated robot X,Y coordinate

Use this BEFORE automatic robot motion.
"""

import cv2
import numpy as np
from pathlib import Path
from pyniryo import NiryoRobot


ROBOT_IP = "192.168.0.199"

BASE_DIR = Path(__file__).resolve().parent
H_FILE = BASE_DIR / "calibration" / "homography_matrix.npy"

if not H_FILE.exists():
    raise FileNotFoundError(f"Missing calibration file: {H_FILE}")

H = np.load(H_FILE)


def pixel_to_robot(px, py):
    point = np.array([[[px, py]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, H)
    robot_x = float(mapped[0][0][0])
    robot_y = float(mapped[0][0][1])
    return robot_x, robot_y


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        robot_x, robot_y = pixel_to_robot(x, y)
        print(f"Pixel ({x}, {y}) -> Robot approx X={robot_x:.4f}, Y={robot_y:.4f}")


def main():
    robot = NiryoRobot(ROBOT_IP)

    cv2.namedWindow("Pixel to Robot Test")
    cv2.setMouseCallback("Pixel to Robot Test", mouse_callback)

    print("Click on the camera image to estimate robot X,Y.")
    print("Press Q to quit.")

    try:
        while True:
            img_compressed = robot.get_img_compressed()
            np_arr = np.frombuffer(img_compressed, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            cv2.putText(frame, "Click point to map pixel -> robot XY",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.imshow("Pixel to Robot Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        robot.close_connection()


if __name__ == "__main__":
    main()