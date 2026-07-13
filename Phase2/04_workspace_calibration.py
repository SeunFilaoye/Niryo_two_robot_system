"""
04_workspace_calibration.py

One-robot conveyor workspace calibration.

This script does 3 things:

1. Records 4 IMAGE points from the wrist camera.
2. Records 4 matching ROBOT physical points using robot.get_pose().
3. Records pickup height/orientation references:
   - hover pose
   - contact/pick pose

This creates:
- calibration/image_points.npy
- calibration/robot_points_xy.npy
- calibration/homography_matrix.npy
- calibration/pick_reference.json
- calibration/calibration_points.json
"""

import cv2
import json
import numpy as np
from pathlib import Path
from pyniryo import NiryoRobot


# ============================================================
# SETTINGS
# ============================================================

ROBOT_IP = "192.168.0.199"

BASE_DIR = Path(__file__).resolve().parent
CAL_DIR = BASE_DIR / "calibration"
CAL_DIR.mkdir(exist_ok=True)

IMAGE_POINTS_FILE = CAL_DIR / "image_points.npy"
ROBOT_POINTS_FILE = CAL_DIR / "robot_points_xy.npy"
H_FILE = CAL_DIR / "homography_matrix.npy"
PICK_REF_FILE = CAL_DIR / "pick_reference.json"
JSON_FILE = CAL_DIR / "calibration_points.json"

POINT_NAMES = [
    "top-left",
    "top-right",
    "bottom-right",
    "bottom-left",
]

image_points = []
robot_points_xy = []
current_mouse = (0, 0)


# ============================================================
# CAMERA CLICK CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global current_mouse

    current_mouse = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(image_points) >= 4:
            print("Already recorded 4 image points. Press C to clear if needed.")
            return

        image_points.append([x, y])
        point_name = POINT_NAMES[len(image_points) - 1]

        print(f"Image point {len(image_points)} recorded: {point_name}")
        print(f"Pixel: ({x}, {y})")


# ============================================================
# DISPLAY HELPERS
# ============================================================

def draw_overlay(frame, mode_text):
    display = frame.copy()

    for i, (x, y) in enumerate(image_points):
        cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            display,
            f"{i + 1}:{POINT_NAMES[i]}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    mx, my = current_mouse

    cv2.putText(
        display,
        f"Mouse pixel: ({mx}, {my})",
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
        f"Image points: {len(image_points)}/4 | Robot XY points: {len(robot_points_xy)}/4",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        display,
        "Controls: LEFT CLICK=image point | C=clear image points | Q=quit",
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
    )

    return display


def show_camera_and_collect_image_points(robot):
    print("=" * 70)
    print("STEP 1: IMAGE POINT COLLECTION")
    print("=" * 70)
    print("Put the robot in the SAME camera/observe pose you will use for detection.")
    print("Click 4 conveyor workspace corners in this order:")
    print("1. top-left")
    print("2. top-right")
    print("3. bottom-right")
    print("4. bottom-left")
    print("")
    print("Press Q after all 4 image points are selected.")
    print("=" * 70)

    cv2.namedWindow("Workspace Calibration - Image Points")
    cv2.setMouseCallback("Workspace Calibration - Image Points", mouse_callback)

    while True:
        img_compressed = robot.get_img_compressed()
        np_arr = np.frombuffer(img_compressed, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print("Could not decode camera frame.")
            continue

        display = draw_overlay(frame, "Click 4 image points")
        cv2.imshow("Workspace Calibration - Image Points", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            image_points.clear()
            print("Cleared image points.")

        elif key == ord("q"):
            if len(image_points) != 4:
                print("You need exactly 4 image points before continuing.")
                print(f"Current count: {len(image_points)}/4")
                continue
            break

    cv2.destroyWindow("Workspace Calibration - Image Points")


# ============================================================
# ROBOT POINT COLLECTION
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


def collect_robot_xy_points(robot):
    print("=" * 70)
    print("STEP 2: ROBOT PHYSICAL POINT COLLECTION")
    print("=" * 70)
    print("Now move/jog the vacuum nozzle to each matching physical conveyor point.")
    print("Use the SAME order as the image points:")
    print("1. top-left")
    print("2. top-right")
    print("3. bottom-right")
    print("4. bottom-left")
    print("")
    print("For each point:")
    print("- Jog the vacuum nozzle to the real physical point.")
    print("- Press ENTER here in PowerShell to record robot.get_pose().")
    print("=" * 70)

    robot_points_xy.clear()

    full_robot_poses = []

    for i, point_name in enumerate(POINT_NAMES, start=1):
        input(f"\nMove robot nozzle to physical point {i} ({point_name}), then press ENTER...")

        pose_dict = get_current_pose_dict(robot)

        robot_points_xy.append([pose_dict["x"], pose_dict["y"]])
        full_robot_poses.append(pose_dict)

        print(f"Recorded point {i} ({point_name}):")
        print(f"X={pose_dict['x']:.4f}, Y={pose_dict['y']:.4f}, Z={pose_dict['z']:.4f}")
        print(f"Roll={pose_dict['roll']:.4f}, Pitch={pose_dict['pitch']:.4f}, Yaw={pose_dict['yaw']:.4f}")

    return full_robot_poses


# ============================================================
# PICK HEIGHT / ORIENTATION COLLECTION
# ============================================================

def collect_pick_references(robot):
    print("=" * 70)
    print("STEP 3: PICK HEIGHT / HOVER REFERENCE COLLECTION")
    print("=" * 70)
    print("Now record two important poses:")
    print("")
    print("1. HOVER pose:")
    print("   Move the vacuum nozzle safely above the conveyor/chip.")
    print("   This is the approach height.")
    print("")
    print("2. CONTACT pose:")
    print("   Move the vacuum nozzle down to the chip pickup/contact height.")
    print("   This is where vacuum pickup happens.")
    print("")
    print("These provide Z, roll, pitch, yaw values for automatic pickup.")
    print("=" * 70)

    input("\nMove robot to safe HOVER pose above conveyor/chip, then press ENTER...")
    hover_pose = get_current_pose_dict(robot)

    print("Recorded hover pose:")
    print(hover_pose)

    input("\nMove robot to CONTACT/PICK pose at chip height, then press ENTER...")
    contact_pose = get_current_pose_dict(robot)

    print("Recorded contact pose:")
    print(contact_pose)

    pick_reference = {
        "hover_pose": hover_pose,
        "contact_pose": contact_pose,
        "hover_z": hover_pose["z"],
        "contact_z": contact_pose["z"],
        "roll": contact_pose["roll"],
        "pitch": contact_pose["pitch"],
        "yaw": contact_pose["yaw"],
    }

    with open(PICK_REF_FILE, "w") as f:
        json.dump(pick_reference, f, indent=4)

    print(f"Pick reference saved to: {PICK_REF_FILE}")

    return pick_reference


# ============================================================
# SAVE CALIBRATION
# ============================================================

def save_calibration(full_robot_poses, pick_reference):
    image_np = np.array(image_points, dtype=np.float32)
    robot_np = np.array(robot_points_xy, dtype=np.float32)

    H, _ = cv2.findHomography(image_np, robot_np)

    np.save(IMAGE_POINTS_FILE, image_np)
    np.save(ROBOT_POINTS_FILE, robot_np)
    np.save(H_FILE, H)

    calibration_data = {
        "description": "One-robot conveyor workspace calibration",
        "calibration_order": POINT_NAMES,
        "image_points_pixels": image_points,
        "robot_points_xy_meters": robot_points_xy,
        "robot_full_poses": full_robot_poses,
        "homography_matrix": H.tolist(),
        "pick_reference": pick_reference,
        "notes": [
            "Image points are camera pixel coordinates.",
            "Robot points are physical robot X,Y coordinates in meters.",
            "Homography maps image pixel centers to robot X,Y pickup coordinates.",
            "Hover/contact references provide Z and orientation for pickup motion.",
            "Camera must remain in the same observe pose used during calibration.",
        ],
    }

    with open(JSON_FILE, "w") as f:
        json.dump(calibration_data, f, indent=4)

    print("=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"Saved image points:      {IMAGE_POINTS_FILE}")
    print(f"Saved robot XY points:   {ROBOT_POINTS_FILE}")
    print(f"Saved homography matrix: {H_FILE}")
    print(f"Saved pick reference:    {PICK_REF_FILE}")
    print(f"Saved JSON reference:    {JSON_FILE}")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Connecting to robot...")
    robot = NiryoRobot(ROBOT_IP)
    print("Connected.")

    try:
        show_camera_and_collect_image_points(robot)

        print("\nImage points recorded:")
        for i, pt in enumerate(image_points, start=1):
            print(f"{i}. {POINT_NAMES[i-1]} pixel={pt}")

        full_robot_poses = collect_robot_xy_points(robot)

        pick_reference = collect_pick_references(robot)

        save_calibration(full_robot_poses, pick_reference)

    finally:
        cv2.destroyAllWindows()
        robot.close_connection()
        print("Disconnected from robot.")


if __name__ == "__main__":
    main()