"""
04_workspace_calibration.py

Robot full conveyor workspace calibration.

This script:

1. Moves the robot to its fixed home/view joint position.
2. Records four image points from the wrist camera.
3. Records four matching physical robot points.
4. Calculates the image-pixel-to-robot-XY homography matrix.
5. Records the hover and contact/pick poses.
6. Saves all calibration information inside robots/<robot>/<conveyor>/calibration/.

Generated files:

- calibration/image_points.npy
- calibration/robot_points_xy.npy
- calibration/homography_matrix.npy
- calibration/pick_reference.json
- calibration/calibration_points.json
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
# ROBOT PATHS
# ============================================================

ROBOT_DIR = S.ROBOT_DIR

CAL_DIR = ROBOT_DIR / "calibration"
CAL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_POINTS_FILE = CAL_DIR / "image_points.npy"
ROBOT_POINTS_FILE = CAL_DIR / "robot_points_xy.npy"
H_FILE = CAL_DIR / "homography_matrix.npy"
PICK_REF_FILE = CAL_DIR / "pick_reference.json"
JSON_FILE = CAL_DIR / "calibration_points.json"


# ============================================================
# CALIBRATION POINT SETTINGS
# ============================================================

POINT_NAMES = [
    "top-left",
    "top-right",
    "bottom-right",
    "bottom-left",
]

image_points = []
robot_points_xy = []

current_mouse = (0, 0)

camera_frame_width = None
camera_frame_height = None


# ============================================================
# CAMERA CLICK CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global current_mouse

    current_mouse = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(image_points) >= 4:
            print(
                "Already recorded four image points. "
                "Press C to clear them if needed."
            )
            return

        image_points.append([x, y])

        point_name = POINT_NAMES[len(image_points) - 1]

        print("")
        print(
            f"Image point {len(image_points)} recorded: "
            f"{point_name}"
        )
        print(f"Pixel coordinate: ({x}, {y})")


# ============================================================
# CAMERA DISPLAY
# ============================================================

PHASE2_CONFIG_FILE = ROBOT_DIR / "phase2_config.json"


def draw_saved_roi(display):
    """Display only: shows the pick ROI from 08 while clicking corners.

    Corners spread across the ROI give the most accurate mapping there.
    """
    if not PHASE2_CONFIG_FILE.exists():
        return
    with PHASE2_CONFIG_FILE.open("r", encoding="utf-8") as file:
        points = json.load(file).get("polygon_roi", [])
    if len(points) < 3:
        return
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(display, [contour], True, (255, 255, 0), 2)
    cv2.putText(
        display,
        "PICK ROI (from 08)",
        (int(points[0][0]), max(20, int(points[0][1]) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 0),
        1,
    )


def draw_overlay(frame, mode_text):
    display = frame.copy()

    draw_saved_roi(display)

    for index, (x, y) in enumerate(image_points):
        cv2.circle(
            display,
            (x, y),
            6,
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            display,
            f"{index + 1}: {POINT_NAMES[index]}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    mouse_x, mouse_y = current_mouse

    cv2.putText(
        display,
        f"Mouse pixel: ({mouse_x}, {mouse_y})",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        display,
        f"Mode: {mode_text}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        display,
        (
            f"Image points: {len(image_points)}/4 | "
            f"Robot XY points: {len(robot_points_xy)}/4"
        ),
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        display,
        (
            "LEFT CLICK = record point | "
            "C = clear points | Q = continue"
        ),
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
    )

    return display


def collect_image_points(robot):
    global camera_frame_width
    global camera_frame_height

    image_points.clear()

    print("")
    print("=" * 70)
    print("STEP 1: IMAGE POINT COLLECTION")
    print("=" * 70)
    print(f"{ROBOT_NAME} is at its fixed home/view position.")
    print("")
    print("Click the four conveyor workspace corners in this order:")
    print("")
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")
    print("")
    print("Press Q only after all four image points are selected.")
    print("=" * 70)

    window_name = f"{ROBOT_NAME} - Workspace Calibration"

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

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

        camera_frame_height, camera_frame_width = frame.shape[:2]

        display = draw_overlay(
            frame,
            "Click four workspace corners",
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            image_points.clear()
            print("Cleared all image points.")

        elif key == ord("q"):
            if len(image_points) != 4:
                print("")
                print("Exactly four image points are required.")
                print(f"Current count: {len(image_points)}/4")
                continue

            break

    cv2.destroyWindow(window_name)


# ============================================================
# ROBOT POSE COLLECTION
# ============================================================

def get_current_pose_dict(robot):
    pose = robot.get_pose()

    return {
        "x": float(pose.x),
        "y": float(pose.y),
        "z": float(pose.z),
        "roll": float(pose.roll),
        "pitch": float(pose.pitch),
        "yaw": float(pose.yaw),
    }


def collect_robot_workspace_points(robot):
    robot_points_xy.clear()

    full_robot_poses = []

    print("")
    print("=" * 70)
    print("STEP 2: ROBOT PHYSICAL POINT COLLECTION")
    print("=" * 70)
    print("Move or jog the vacuum nozzle to each matching")
    print("physical conveyor point.")
    print("")
    print("Use the exact same order as the image points:")
    print("")
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")
    print("")
    print("Press ENTER in the Ubuntu terminal to record each pose.")
    print("=" * 70)

    for index, point_name in enumerate(POINT_NAMES, start=1):
        input(
            f"\nMove the nozzle to point {index} "
            f"({point_name}), then press ENTER..."
        )

        pose_dict = get_current_pose_dict(robot)

        robot_points_xy.append(
            [
                pose_dict["x"],
                pose_dict["y"],
            ]
        )

        full_robot_poses.append(pose_dict)

        print("")
        print(f"Recorded point {index}: {point_name}")
        print(
            f"X={pose_dict['x']:.4f}, "
            f"Y={pose_dict['y']:.4f}, "
            f"Z={pose_dict['z']:.4f}"
        )
        print(
            f"Roll={pose_dict['roll']:.4f}, "
            f"Pitch={pose_dict['pitch']:.4f}, "
            f"Yaw={pose_dict['yaw']:.4f}"
        )

    return full_robot_poses


# ============================================================
# PICK HEIGHT AND ORIENTATION
# ============================================================

def collect_pick_references(robot):
    print("")
    print("=" * 70)
    print("STEP 3: PICK HEIGHT AND ORIENTATION")
    print("=" * 70)
    print("Two poses will now be recorded:")
    print("")
    print("1. HOVER pose")
    print("   Safely above the conveyor and chip.")
    print("")
    print("2. CONTACT/PICK pose")
    print("   At the correct vacuum pickup height.")
    print("")
    print("These values will be used by future pickup scripts.")
    print("=" * 70)

    input(
        "\nMove the nozzle to the safe HOVER pose, "
        "then press ENTER..."
    )

    hover_pose = get_current_pose_dict(robot)

    print("")
    print("Recorded hover pose:")
    print(json.dumps(hover_pose, indent=4))

    input(
        "\nMove the nozzle to the CONTACT/PICK pose, "
        "then press ENTER..."
    )

    contact_pose = get_current_pose_dict(robot)

    print("")
    print("Recorded contact/pick pose:")
    print(json.dumps(contact_pose, indent=4))

    pick_reference = {
        "robot_name": ROBOT_NAME,
        "robot_ip": ROBOT_IP,
        "hover_pose": hover_pose,
        "contact_pose": contact_pose,
        "hover_z": hover_pose["z"],
        "contact_z": contact_pose["z"],
        "roll": contact_pose["roll"],
        "pitch": contact_pose["pitch"],
        "yaw": contact_pose["yaw"],
    }

    return pick_reference


# ============================================================
# OVERWRITE CHECK
# ============================================================

def calibration_files_exist():
    calibration_files = [
        IMAGE_POINTS_FILE,
        ROBOT_POINTS_FILE,
        H_FILE,
        PICK_REF_FILE,
        JSON_FILE,
    ]

    return any(path.exists() for path in calibration_files)


def confirm_overwrite():
    if not calibration_files_exist():
        return True

    print("")
    print("=" * 70)
    print("EXISTING CALIBRATION DETECTED")
    print("=" * 70)
    print(f"Calibration directory: {CAL_DIR}")
    print("")
    print("Running this script will replace the existing calibration.")
    print("=" * 70)

    response = input(
        "Type YES to replace the existing calibration: "
    ).strip()

    return response == "YES"


# ============================================================
# SAVE CALIBRATION
# ============================================================

def save_calibration(full_robot_poses, pick_reference):
    image_np = np.array(
        image_points,
        dtype=np.float32,
    )

    robot_np = np.array(
        robot_points_xy,
        dtype=np.float32,
    )

    if image_np.shape != (4, 2):
        raise ValueError(
            f"Invalid image-point shape: {image_np.shape}. "
            "Expected (4, 2)."
        )

    if robot_np.shape != (4, 2):
        raise ValueError(
            f"Invalid robot-point shape: {robot_np.shape}. "
            "Expected (4, 2)."
        )

    homography, mask = cv2.findHomography(
        image_np,
        robot_np,
        method=0,
    )

    if homography is None:
        raise RuntimeError(
            "Homography calculation failed. "
            "Recheck all four calibration points."
        )

    if not np.all(np.isfinite(homography)):
        raise RuntimeError(
            "Homography contains invalid numeric values."
        )

    calibration_time = datetime.now().isoformat(
        timespec="seconds"
    )

    calibration_data = {
        "calibration_version": 1,
        "created_at": calibration_time,
        "description": (
            f"{ROBOT_NAME} wrist-camera-to-conveyor workspace calibration"
        ),
        "robot_name": ROBOT_NAME,
        "robot_ip": ROBOT_IP,
        "robot_folder": str(ROBOT_DIR),
        "calibration_directory": str(CAL_DIR),
        "home_view_joints": list(HOME_VIEW_JOINTS),
        "camera_frame_width": camera_frame_width,
        "camera_frame_height": camera_frame_height,
        "calibration_order": POINT_NAMES,
        "image_points_pixels": image_points,
        "robot_points_xy_meters": robot_points_xy,
        "robot_full_poses": full_robot_poses,
        "homography_matrix": homography.tolist(),
        "pick_reference": pick_reference,
        "notes": [
            (
                "Image points are wrist-camera pixel coordinates."
            ),
            (
                "Robot points are physical X and Y coordinates "
                "in meters."
            ),
            (
                "The homography maps detected image centers "
                "to robot pickup X and Y coordinates."
            ),
            (
                "The camera must return to the saved home/view "
                "joint position before detection."
            ),
            (
                "Hover and contact references provide the pickup "
                "Z height and tool orientation."
            ),
        ],
    }

    np.save(
        IMAGE_POINTS_FILE,
        image_np,
    )

    np.save(
        ROBOT_POINTS_FILE,
        robot_np,
    )

    np.save(
        H_FILE,
        homography,
    )

    with PICK_REF_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            pick_reference,
            file,
            indent=4,
        )

    with JSON_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            calibration_data,
            file,
            indent=4,
        )

    print("")
    print("=" * 70)
    print(f"{ROBOT_NAME} CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"Image points:      {IMAGE_POINTS_FILE}")
    print(f"Robot XY points:   {ROBOT_POINTS_FILE}")
    print(f"Homography matrix: {H_FILE}")
    print(f"Pick reference:    {PICK_REF_FILE}")
    print(f"JSON reference:    {JSON_FILE}")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    robot = None

    print("=" * 70)
    print(f"{ROBOT_NAME} FULL WORKSPACE CALIBRATION")
    print(f"Robot IP: {ROBOT_IP}")
    print(f"Calibration directory: {CAL_DIR}")
    print("=" * 70)

    if not confirm_overwrite():
        print("Calibration cancelled.")
        return

    try:
        print(f"Connecting to {ROBOT_NAME}...")

        robot = NiryoRobot(ROBOT_IP)

        print(f"Connected to {ROBOT_NAME}.")

        robot.update_tool()

        print("")
        print(f"Moving {ROBOT_NAME} to home/view position...")

        robot.move(JointsPosition(*HOME_VIEW_JOINTS))

        print(f"{ROBOT_NAME} reached home/view position.")

        collect_image_points(robot)

        print("")
        print("Recorded image points:")

        for index, point in enumerate(image_points, start=1):
            print(
                f"{index}. {POINT_NAMES[index - 1]} "
                f"pixel={point}"
            )

        full_robot_poses = collect_robot_workspace_points(robot)

        pick_reference = collect_pick_references(robot)

        save_calibration(
            full_robot_poses,
            pick_reference,
        )

    except KeyboardInterrupt:
        print("\nCalibration interrupted from the terminal.")

    except Exception as error:
        print("")
        print(f"{ROBOT_NAME} calibration failed.")
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