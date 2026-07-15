"""
07_run_pick_place.py

Full one-robot vision-guided chip pick-and-place.

Flow:
1. Start conveyor
2. Camera sees chip inside ROI
3. Stop conveyor immediately
4. Wait 2.5 sec for camera/stream to settle
5. Re-detect chip
6. Convert pixel center to robot X/Y
7. Robot picks chip using vacuum
8. Robot moves through safe midpoint
9. Robot drops chip
10. Robot returns home
11. Conveyor restarts
12. Repeat

Press Q to quit.
"""

import csv
import cv2
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from pyniryo import NiryoRobot, ConveyorDirection


# ============================================================
# SETTINGS
# ============================================================

ROBOT_IP = "192.168.0.199"

ENABLE_ROBOT_MOVEMENT = True
ENABLE_CONVEYOR = True

CONF_THRESHOLD = 0.75

CONVEYOR_SPEED = 50
CONVEYOR_DIRECTION = ConveyorDirection.FORWARD

CAMERA_SETTLE_TIME_SEC = 2.5

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"
H_FILE = BASE_DIR / "calibration" / "homography_matrix.npy"
PICK_REF_FILE = BASE_DIR / "calibration" / "pick_reference.json"
LOG_FILE = BASE_DIR / "logs" / "pick_place_log.csv"


# ============================================================
# ROI SETTINGS
# Bright blue ROI
# ============================================================

ROI_X1 = 185
ROI_Y1 = 177
ROI_X2 = 562
ROI_Y2 = 431

ROI_COLOR = (255, 255, 0)  # bright blue/cyan in OpenCV BGR
ROI_THICKNESS = 3


# ============================================================
# ROBOT JOINT POSITIONS
# ============================================================

HOME_JOINTS = (
    -0.060,
    0.293,
    -0.203,
    0.033,
    -1.886,
    -1.751
)

MIDPOINT_JOINTS = (
    1.3005,
    -0.0217,
    -0.3825,
    -0.052,
    -1.2104,
    -1.744
)

DROP_JOINTS = (
    2.286,
    -0.426,
    -0.559,
    0.099,
    -0.489,
    -1.655
)


# ============================================================
# CLASS COLORS
# OpenCV uses BGR, not RGB.
# ============================================================

CLASS_COLORS = {
    "red_circle": (0, 0, 255),
    "red_square": (0, 0, 255),

    "blue_circle": (255, 0, 0),
    "blue_square": (255, 0, 0),

    "green_circle": (0, 255, 0),
    "green_square": (0, 255, 0),
}

DEFAULT_BOX_COLOR = (255, 255, 255)


# ============================================================
# LOGGING
# ============================================================

def init_log():
    LOG_FILE.parent.mkdir(exist_ok=True)

    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "class",
                "confidence",
                "pixel_x",
                "pixel_y",
                "robot_x",
                "robot_y",
                "action",
            ])


def log_event(class_name, confidence, px, py, rx, ry, action):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            class_name,
            f"{confidence:.4f}",
            px,
            py,
            f"{rx:.4f}",
            f"{ry:.4f}",
            action,
        ])


# ============================================================
# CALIBRATION / PICK REFERENCE
# ============================================================

def load_pick_reference():
    if not PICK_REF_FILE.exists():
        raise FileNotFoundError(f"Missing pick reference file: {PICK_REF_FILE}")

    with open(PICK_REF_FILE, "r") as f:
        data = json.load(f)

    return data


def pixel_to_robot(px, py, H):
    point = np.array([[[px, py]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, H)

    robot_x = float(mapped[0][0][0])
    robot_y = float(mapped[0][0][1])

    return robot_x, robot_y


# ============================================================
# IMAGE / DETECTION HELPERS
# ============================================================

def get_camera_frame(robot):
    img_compressed = robot.get_img_compressed()
    np_arr = np.frombuffer(img_compressed, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def get_center(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def inside_roi(cx, cy):
    return ROI_X1 <= cx <= ROI_X2 and ROI_Y1 <= cy <= ROI_Y2


def draw_roi(frame):
    cv2.rectangle(
        frame,
        (ROI_X1, ROI_Y1),
        (ROI_X2, ROI_Y2),
        ROI_COLOR,
        ROI_THICKNESS,
    )

    cv2.putText(
        frame,
        "PICK ROI",
        (ROI_X1, ROI_Y1 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        ROI_COLOR,
        2,
    )

    return frame


def draw_detection(frame, det):
    x1, y1, x2, y2 = map(int, det["xyxy"])
    cx = det["cx"]
    cy = det["cy"]
    class_name = det["class_name"]
    confidence = det["confidence"]

    color = CLASS_COLORS.get(class_name, DEFAULT_BOX_COLOR)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

    label = f"{class_name} | {confidence:.2f} | px=({cx},{cy})"

    cv2.rectangle(
        frame,
        (x1, max(y1 - 32, 0)),
        (x1 + min(430, len(label) * 12), y1),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x1 + 5, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
    )

    return frame


def find_first_chip_in_roi(model, frame):
    """
    First come, first pick:
    - scan detections in YOLO output order
    - return the first valid chip inside ROI above confidence threshold
    """

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    all_detections = []

    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls_id]

            cx, cy = get_center(xyxy)

            det = {
                "xyxy": xyxy,
                "class_name": class_name,
                "confidence": confidence,
                "cx": cx,
                "cy": cy,
                "inside_roi": inside_roi(cx, cy),
            }

            all_detections.append(det)

            if det["inside_roi"]:
                return det, all_detections

    return None, all_detections


# ============================================================
# CONVEYOR CONTROL
# ============================================================

def setup_conveyor(robot):
    if not ENABLE_CONVEYOR:
        return None

    try:
        conveyor_id = robot.set_conveyor()
        print(f"Conveyor connected: {conveyor_id}")
        return conveyor_id

    except Exception as e:
        print("Could not set conveyor automatically.")
        print(e)
        return None


def start_conveyor(robot, conveyor_id):
    if not ENABLE_CONVEYOR:
        return

    try:
        robot.run_conveyor(
            conveyor_id,
            speed=CONVEYOR_SPEED,
            direction=CONVEYOR_DIRECTION,
        )
        print("Conveyor running.")

    except Exception as e:
        print("Could not start conveyor.")
        print(e)


def stop_conveyor(robot, conveyor_id):
    if not ENABLE_CONVEYOR:
        return

    try:
        robot.stop_conveyor(conveyor_id)
        print("Conveyor stopped.")

    except Exception as e:
        print("Could not stop conveyor.")
        print(e)


# ============================================================
# ROBOT PICK AND PLACE
# ============================================================

def build_pick_poses(robot_x, robot_y, pick_ref):
    hover_pose = (
        robot_x,
        robot_y,
        float(pick_ref["hover_z"]),
        float(pick_ref["roll"]),
        float(pick_ref["pitch"]),
        float(pick_ref["yaw"]),
    )

    contact_pose = (
        robot_x,
        robot_y,
        float(pick_ref["contact_z"]),
        float(pick_ref["roll"]),
        float(pick_ref["pitch"]),
        float(pick_ref["yaw"]),
    )

    return hover_pose, contact_pose


def pick_and_place(robot, robot_x, robot_y, pick_ref):
    hover_pose, contact_pose = build_pick_poses(robot_x, robot_y, pick_ref)

    print("Updating tool...")
    robot.update_tool()

    print("Moving to HOME...")
    robot.move_joints(HOME_JOINTS)

    print("Moving to PICK HOVER...")
    robot.move_pose(*hover_pose)

    print("Descending to PICK CONTACT...")
    robot.move_pose(*contact_pose)

    print("Vacuum ON...")
    robot.grasp_with_tool()
    time.sleep(0.7)

    print("Lifting to PICK HOVER...")
    robot.move_pose(*hover_pose)

    print("Moving through MIDPOINT...")
    robot.move_joints(MIDPOINT_JOINTS)

    print("Moving to DROP...")
    robot.move_joints(DROP_JOINTS)

    print("Vacuum OFF...")
    robot.release_with_tool()
    time.sleep(0.7)

    print("Returning through MIDPOINT...")
    robot.move_joints(MIDPOINT_JOINTS)

    print("Returning HOME...")
    robot.move_joints(HOME_JOINTS)

    print("Robot is back HOME.")


# ============================================================
# SAFETY CHECKS
# ============================================================

def safety_check():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing YOLO model: {MODEL_PATH}")

    if not H_FILE.exists():
        raise FileNotFoundError(f"Missing homography file: {H_FILE}")

    if not PICK_REF_FILE.exists():
        raise FileNotFoundError(f"Missing pick reference file: {PICK_REF_FILE}")


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    safety_check()
    init_log()

    print("Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))

    print("Loading calibration...")
    H = np.load(H_FILE)

    print("Loading pick reference...")
    pick_ref = load_pick_reference()

    print("Connecting to robot...")
    robot = NiryoRobot(ROBOT_IP)
    print("Connected.")

    conveyor_id = setup_conveyor(robot)

    system_busy = False

    try:
        print("Moving robot HOME before starting...")
        if ENABLE_ROBOT_MOVEMENT:
            robot.move_joints(HOME_JOINTS)

        start_conveyor(robot, conveyor_id)

        print("=" * 60)
        print("SYSTEM RUNNING")
        print("Place chips on conveyor.")
        print("Detection inside ROI will stop conveyor and trigger pickup.")
        print("Press Q to quit.")
        print("=" * 60)

        while True:
            frame = get_camera_frame(robot)

            if frame is None:
                print("Could not decode frame.")
                continue

            frame = draw_roi(frame)

            detection, all_detections = find_first_chip_in_roi(model, frame)

            for det in all_detections:
                frame = draw_detection(frame, det)

            if detection is not None and not system_busy:
                system_busy = True

                print("=" * 60)
                print("CHIP DETECTED INSIDE ROI")
                print(f"Class: {detection['class_name']}")
                print(f"Confidence: {detection['confidence']:.2f}")
                print(f"Pixel center: ({detection['cx']}, {detection['cy']})")
                print("=" * 60)

                stop_conveyor(robot, conveyor_id)

                print(f"Waiting {CAMERA_SETTLE_TIME_SEC} seconds for camera/stream to settle...")
                time.sleep(CAMERA_SETTLE_TIME_SEC)

                print("Re-detecting chip after conveyor stop...")

                stable_frame = get_camera_frame(robot)

                if stable_frame is None:
                    print("Could not get stable frame. Restarting conveyor.")
                    start_conveyor(robot, conveyor_id)
                    system_busy = False
                    continue

                stable_detection, stable_all = find_first_chip_in_roi(model, stable_frame)

                if stable_detection is None:
                    print("No chip found after settling. Restarting conveyor.")
                    start_conveyor(robot, conveyor_id)
                    system_busy = False
                    continue

                sx = stable_detection["cx"]
                sy = stable_detection["cy"]

                robot_x, robot_y = pixel_to_robot(sx, sy, H)

                print("=" * 60)
                print("STABLE DETECTION")
                print(f"Class: {stable_detection['class_name']}")
                print(f"Confidence: {stable_detection['confidence']:.2f}")
                print(f"Pixel center: ({sx}, {sy})")
                print(f"Robot target: X={robot_x:.4f}, Y={robot_y:.4f}")
                print("=" * 60)

                log_event(
                    stable_detection["class_name"],
                    stable_detection["confidence"],
                    sx,
                    sy,
                    robot_x,
                    robot_y,
                    "pickup_started",
                )

                if ENABLE_ROBOT_MOVEMENT:
                    pick_and_place(robot, robot_x, robot_y, pick_ref)
                else:
                    print("Robot movement disabled. Set ENABLE_ROBOT_MOVEMENT=True to pick.")

                log_event(
                    stable_detection["class_name"],
                    stable_detection["confidence"],
                    sx,
                    sy,
                    robot_x,
                    robot_y,
                    "pickup_complete",
                )

                print("Restarting conveyor after robot returned home...")
                start_conveyor(robot, conveyor_id)

                system_busy = False

            cv2.imshow("Single Robot Vision Pick-and-Place", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        print("Shutting down...")

        try:
            stop_conveyor(robot, conveyor_id)
        except Exception:
            pass

        cv2.destroyAllWindows()
        robot.close_connection()
        print("Disconnected.")


if __name__ == "__main__":
    main()