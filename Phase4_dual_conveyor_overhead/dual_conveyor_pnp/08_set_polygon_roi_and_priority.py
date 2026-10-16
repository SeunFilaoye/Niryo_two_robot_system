"""
08_set_polygon_roi_and_priority.py

Purpose:
- Connect to the robot.
- Move the robot to its calibrated HOME/VIEW position.
- Display the wrist-camera feed.
- Let the user create a custom polygon ROI.
- Let the user select a priority point.
- Save both settings to phase2_config.json.

Controls:
- LEFT CLICK: add polygon vertex
- P: close/finalize polygon
- RIGHT CLICK: set priority point
- C: clear polygon
- R: clear priority point
- S: save configuration
- Q: quit

The polygon ROI defines where detections are valid.
The priority point defines which valid chip has the highest pickup priority.
"""

import cv2
import json
import numpy as np

from datetime import datetime
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
# PATHS
# ============================================================

ROBOT_DIR = S.ROBOT_DIR
CONFIG_FILE = ROBOT_DIR / "phase2_config.json"
CALIBRATION_FILE = ROBOT_DIR / "calibration" / "calibration_points.json"

WINDOW_NAME = f"{ROBOT_NAME} - Polygon ROI and Priority"


# ============================================================
# DISPLAY SETTINGS
# ============================================================

POLYGON_COLOR = (255, 255, 0)
POLYGON_FILL_COLOR = (255, 255, 0)
VERTEX_COLOR = (0, 255, 0)
PRIORITY_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)

POLYGON_ALPHA = 0.20

polygon_points = []
polygon_closed = False
priority_point = None
current_mouse = (0, 0)
status_message = "Add polygon points with LEFT CLICK."


# ============================================================
# CONFIGURATION HELPERS
# ============================================================

def load_existing_config():
    if not CONFIG_FILE.exists():
        return {}

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_saved_values():
    global polygon_points
    global polygon_closed
    global priority_point
    global status_message

    data = load_existing_config()

    saved_polygon = data.get("polygon_roi", [])

    if isinstance(saved_polygon, list) and len(saved_polygon) >= 3:
        polygon_points = [
            (int(point[0]), int(point[1]))
            for point in saved_polygon
        ]
        polygon_closed = True

    saved_priority = data.get("priority_point")

    if isinstance(saved_priority, dict):
        if "x" in saved_priority and "y" in saved_priority:
            priority_point = (
                int(saved_priority["x"]),
                int(saved_priority["y"]),
            )

    if polygon_closed or priority_point is not None:
        status_message = "Loaded existing configuration."


def validate_against_calibration():
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Missing calibration file:\n{CALIBRATION_FILE}\n"
            "Run 04_workspace_calibration.py first."
        )

    with CALIBRATION_FILE.open("r", encoding="utf-8") as file:
        calibration = json.load(file)

    saved_robot_name = calibration.get("robot_name")
    saved_robot_ip = calibration.get("robot_ip")
    saved_home = calibration.get("home_view_joints")

    if saved_robot_name not in (None, ROBOT_NAME):
        raise ValueError(
            f"Calibration belongs to {saved_robot_name}, "
            f"not {ROBOT_NAME}."
        )

    if saved_robot_ip not in (None, ROBOT_IP):
        raise ValueError(
            f"Calibration IP is {saved_robot_ip}, "
            f"not {ROBOT_IP}."
        )

    if saved_home is not None:
        saved = np.asarray(saved_home, dtype=float)
        current = np.asarray(HOME_VIEW_JOINTS, dtype=float)

        if saved.shape != (6,) or not np.allclose(
            saved,
            current,
            atol=1e-4,
            rtol=0.0,
        ):
            raise ValueError(
                "HOME_VIEW_JOINTS do not match the viewing "
                "position used during workspace calibration."
            )


def save_configuration():
    global status_message

    if len(polygon_points) < 3 or not polygon_closed:
        status_message = "Close a polygon with at least 3 points before saving."
        print(status_message)
        return False

    if priority_point is None:
        status_message = "Set a priority point with RIGHT CLICK before saving."
        print(status_message)
        return False

    existing = load_existing_config()

    existing.update({
        "config_version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "robot_name": ROBOT_NAME,
        "robot_ip": ROBOT_IP,
        "home_view_joints": list(HOME_VIEW_JOINTS),
        "selection_method": "closest_valid_chip_to_priority_point",
        "polygon_roi": [
            [int(x), int(y)]
            for x, y in polygon_points
        ],
        "priority_point": {
            "x": int(priority_point[0]),
            "y": int(priority_point[1]),
        },
        "notes": [
            "Only detections whose center is inside polygon_roi are valid.",
            "Among valid detections, choose the chip closest to priority_point.",
            "The robot camera must use the saved HOME/VIEW position.",
        ],
    })

    with CONFIG_FILE.open("w", encoding="utf-8") as file:
        json.dump(existing, file, indent=4)

    status_message = f"Configuration saved: {CONFIG_FILE}"

    print("=" * 70)
    print(f"{ROBOT_NAME} PHASE 2 CONFIGURATION SAVED")
    print(f"Polygon vertices: {len(polygon_points)}")
    print(f"Priority point: {priority_point}")
    print(f"Saved to: {CONFIG_FILE}")
    print("=" * 70)

    return True


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global current_mouse
    global polygon_closed
    global priority_point
    global status_message

    current_mouse = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if polygon_closed:
            status_message = (
                "Polygon is already closed. Press C to clear it."
            )
            return

        polygon_points.append((x, y))
        status_message = (
            f"Added polygon point {len(polygon_points)}: ({x}, {y})"
        )

        print(status_message)

    elif event == cv2.EVENT_RBUTTONDOWN:
        priority_point = (x, y)
        status_message = f"Priority point set: ({x}, {y})"

        print(status_message)


# ============================================================
# POLYGON HELPERS
# ============================================================

def close_polygon():
    global polygon_closed
    global status_message

    if polygon_closed:
        status_message = "Polygon is already closed."
        return

    if len(polygon_points) < 3:
        status_message = "At least 3 polygon points are required."
        print(status_message)
        return

    polygon_closed = True
    status_message = "Polygon closed. Set priority point and press S."

    print(status_message)


def clear_polygon():
    global polygon_closed
    global status_message

    polygon_points.clear()
    polygon_closed = False
    status_message = "Polygon cleared. Add new points with LEFT CLICK."

    print(status_message)


def clear_priority():
    global priority_point
    global status_message

    priority_point = None
    status_message = "Priority point cleared. Set it with RIGHT CLICK."

    print(status_message)


# ============================================================
# DISPLAY
# ============================================================

def draw_polygon_overlay(display):
    if not polygon_points:
        return display

    contour = np.array(
        polygon_points,
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    if polygon_closed and len(polygon_points) >= 3:
        overlay = display.copy()

        cv2.fillPoly(
            overlay,
            [contour],
            POLYGON_FILL_COLOR,
        )

        cv2.addWeighted(
            overlay,
            POLYGON_ALPHA,
            display,
            1.0 - POLYGON_ALPHA,
            0,
            display,
        )

        cv2.polylines(
            display,
            [contour],
            True,
            POLYGON_COLOR,
            3,
        )
    else:
        cv2.polylines(
            display,
            [contour],
            False,
            POLYGON_COLOR,
            2,
        )

    for index, (point_x, point_y) in enumerate(
        polygon_points,
        start=1,
    ):
        cv2.circle(
            display,
            (point_x, point_y),
            6,
            VERTEX_COLOR,
            -1,
        )

        cv2.putText(
            display,
            str(index),
            (point_x + 8, point_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            VERTEX_COLOR,
            2,
        )

    return display


def draw_priority_overlay(display):
    if priority_point is None:
        return display

    point_x, point_y = priority_point

    cv2.drawMarker(
        display,
        (point_x, point_y),
        PRIORITY_COLOR,
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=3,
    )

    cv2.circle(
        display,
        (point_x, point_y),
        10,
        PRIORITY_COLOR,
        2,
    )

    cv2.putText(
        display,
        "PRIORITY",
        (point_x + 14, point_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        PRIORITY_COLOR,
        2,
    )

    return display


def draw_interface(frame):
    display = frame.copy()

    draw_polygon_overlay(display)
    draw_priority_overlay(display)

    mouse_x, mouse_y = current_mouse

    cv2.putText(
        display,
        f"Mouse: ({mouse_x}, {mouse_y})",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        TEXT_COLOR,
        2,
    )

    polygon_state = (
        f"Polygon: {len(polygon_points)} points | "
        f"{'CLOSED' if polygon_closed else 'OPEN'}"
    )

    cv2.putText(
        display,
        polygon_state,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        POLYGON_COLOR,
        2,
    )

    priority_state = (
        f"Priority: {priority_point}"
        if priority_point is not None
        else "Priority: NOT SET"
    )

    cv2.putText(
        display,
        priority_state,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        PRIORITY_COLOR,
        2,
    )

    cv2.putText(
        display,
        status_message,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        TEXT_COLOR,
        2,
    )

    controls_line_1 = (
        "LEFT CLICK=vertex | P=close polygon | "
        "RIGHT CLICK=priority"
    )

    controls_line_2 = (
        "C=clear polygon | R=clear priority | "
        "S=save | Q=quit"
    )

    cv2.putText(
        display,
        controls_line_1,
        (20, display.shape[0] - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEXT_COLOR,
        2,
    )

    cv2.putText(
        display,
        controls_line_2,
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEXT_COLOR,
        2,
    )

    return display


# ============================================================
# MAIN
# ============================================================

def main():
    robot = None

    print("=" * 70)
    print(f"{ROBOT_NAME} POLYGON ROI AND PRIORITY CONFIGURATION")
    print(f"Robot IP: {ROBOT_IP}")
    print(f"Configuration file: {CONFIG_FILE}")
    print("=" * 70)

    validate_against_calibration()
    load_saved_values()

    try:
        print(f"Connecting to {ROBOT_NAME}...")
        robot = NiryoRobot(ROBOT_IP)
        print(f"Connected to {ROBOT_NAME}.")

        print("Moving to calibrated HOME/VIEW position...")
        robot.move(JointsPosition(*HOME_VIEW_JOINTS))
        print("Reached HOME/VIEW position.")

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(
            WINDOW_NAME,
            mouse_callback,
        )

        print("")
        print("Controls:")
        print("LEFT CLICK: add polygon point")
        print("P: close polygon")
        print("RIGHT CLICK: set priority point")
        print("C: clear polygon")
        print("R: clear priority point")
        print("S: save")
        print("Q: quit")
        print("")

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

            display = draw_interface(frame)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("p"):
                close_polygon()

            elif key == ord("c"):
                clear_polygon()

            elif key == ord("r"):
                clear_priority()

            elif key == ord("s"):
                save_configuration()

            elif key == ord("q"):
                print("Q pressed. Closing configuration tool.")
                break

    except KeyboardInterrupt:
        print("\nConfiguration tool interrupted from terminal.")

    except Exception as error:
        print("")
        print(f"{ROBOT_NAME} Script 08 failed.")
        print(f"Error: {error}")

    finally:
        cv2.destroyAllWindows()

        if robot is not None:
            try:
                robot.close_connection()
                print(f"Disconnected from {ROBOT_NAME}.")
            except Exception as error:
                print(f"Disconnect error: {error}")


if __name__ == "__main__":
    main()

