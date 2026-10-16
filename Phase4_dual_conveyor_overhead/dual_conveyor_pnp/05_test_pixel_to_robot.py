"""
05_test_pixel_to_robot.py

Robot pixel-to-robot mapping test.

Purpose:
- Connect to the robot
- Move the robot to the calibrated home/view position
- Load the robot's saved homography matrix
- Show the wrist-camera feed
- Click any image point
- Print the estimated robot X and Y coordinates
- No automatic pickup or robot movement after positioning
"""

import cv2
import numpy as np
from pathlib import Path
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
# ROBOT CALIBRATION PATHS
# ============================================================

ROBOT_DIR = S.ROBOT_DIR
CALIBRATION_DIR = ROBOT_DIR / "calibration"

H_FILE = CALIBRATION_DIR / "homography_matrix.npy"

WINDOW_NAME = f"{ROBOT_NAME} - Pixel to Robot Test"


# ============================================================
# GLOBAL DISPLAY VALUES
# ============================================================

homography = None

last_clicked_pixel = None
last_robot_coordinate = None


# ============================================================
# CALIBRATION LOADING
# ============================================================

def load_homography():
    if not H_FILE.exists():
        raise FileNotFoundError(
            f"Missing {ROBOT_NAME} calibration file:\n{H_FILE}\n"
            "Run 04_workspace_calibration.py first."
        )

    loaded_homography = np.load(H_FILE)

    if loaded_homography.shape != (3, 3):
        raise ValueError(
            f"Invalid homography shape: {loaded_homography.shape}. "
            "Expected (3, 3)."
        )

    if not np.all(np.isfinite(loaded_homography)):
        raise ValueError(
            "The homography matrix contains invalid numeric values."
        )

    return loaded_homography


# ============================================================
# PIXEL-TO-ROBOT CONVERSION
# ============================================================

def pixel_to_robot(pixel_x, pixel_y):
    point = np.array(
        [[[pixel_x, pixel_y]]],
        dtype=np.float32,
    )

    mapped_point = cv2.perspectiveTransform(
        point,
        homography,
    )

    robot_x = float(mapped_point[0][0][0])
    robot_y = float(mapped_point[0][0][1])

    return robot_x, robot_y


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global last_clicked_pixel
    global last_robot_coordinate

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    robot_x, robot_y = pixel_to_robot(x, y)

    last_clicked_pixel = (x, y)
    last_robot_coordinate = (robot_x, robot_y)

    print("")
    print("=" * 60)
    print(f"{ROBOT_NAME} PIXEL-TO-ROBOT RESULT")
    print(f"Pixel coordinate: ({x}, {y})")
    print(f"Estimated robot X: {robot_x:.4f} m")
    print(f"Estimated robot Y: {robot_y:.4f} m")
    print("=" * 60)


# ============================================================
# DISPLAY
# ============================================================

def draw_overlay(frame):
    cv2.putText(
        frame,
        "Click a point to map pixel -> robot XY",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
    )

    if (
        last_clicked_pixel is not None
        and last_robot_coordinate is not None
    ):
        pixel_x, pixel_y = last_clicked_pixel
        robot_x, robot_y = last_robot_coordinate

        cv2.circle(
            frame,
            (pixel_x, pixel_y),
            7,
            (0, 255, 255),
            -1,
        )

        result_text = (
            f"Pixel ({pixel_x},{pixel_y}) -> "
            f"Robot ({robot_x:.4f}, {robot_y:.4f})"
        )

        cv2.putText(
            frame,
            result_text,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():
    global homography

    robot = None

    print("=" * 70)
    print(f"{ROBOT_NAME} PIXEL-TO-ROBOT TEST")
    print(f"Robot IP: {ROBOT_IP}")
    print(f"Homography file: {H_FILE}")
    print("=" * 70)

    homography = load_homography()

    print("Homography loaded successfully.")

    try:
        print(f"Connecting to {ROBOT_NAME}...")

        robot = NiryoRobot(ROBOT_IP)

        print(f"Connected to {ROBOT_NAME}.")

        print(f"Moving {ROBOT_NAME} to home/view position...")

        robot.move(JointsPosition(*HOME_VIEW_JOINTS))

        print(f"{ROBOT_NAME} reached home/view position.")
        print("")
        print("Click a point in the camera window.")
        print("The estimated robot X and Y will appear")
        print("in both the terminal and camera window.")
        print("Press Q in the camera window to quit.")

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(
            WINDOW_NAME,
            mouse_callback,
        )

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

            frame = draw_overlay(frame)

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Q pressed. Closing pixel-to-robot test.")
                break

    except KeyboardInterrupt:
        print("\nPixel-to-robot test interrupted from the terminal.")

    except Exception as error:
        print("")
        print(f"{ROBOT_NAME} pixel-to-robot test failed.")
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