"""
09_run_pick_place_phase2.py

Phase 2: Conveyor-always-on accumulation wall pick-and-place.

Behavior:
- Conveyor starts once and stays ON during operation.
- Conveyor does NOT stop for pickup.
- Conveyor only stops when the program exits.
- Chip must stay near the ROI wall for 2 seconds before target lock.
- Wall is shifted slightly right using LEFT_WALL_TRIGGER_OFFSET_PX.
- Once locked, robot immediately fetches that locked target.
- Robot drops it off.
- Robot returns to Phase 2 viewing pose.
- System continues scanning.
"""

import csv
import cv2
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
from pyniryo import NiryoRobot, ConveyorDirection


# ============================================================
# BASIC SETTINGS
# ============================================================

ROBOT_IP = "192.168.0.199"

ENABLE_ROBOT_MOVEMENT = True
ENABLE_CONVEYOR = True

CONF_THRESHOLD = 0.75

CONVEYOR_SPEED = 50
CONVEYOR_DIRECTION = ConveyorDirection.FORWARD

POST_PICK_DELAY_SEC = 0.75

# Wall moved more to the right.
# Wall x-position = ROI x1 + this value.
LEFT_WALL_TRIGGER_OFFSET_PX = 55

WALL_DWELL_TIME_SEC = 2.0
TARGET_MATCH_DISTANCE_PX = 45

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"
CONFIG_FILE = BASE_DIR / "phase2_config.json"
H_FILE = BASE_DIR / "calibration" / "homography_matrix.npy"
PICK_REF_FILE = BASE_DIR / "calibration" / "pick_reference.json"
LOG_FILE = BASE_DIR / "logs" / "phase2_pick_place_log.csv"


# ============================================================
# ROBOT JOINTS
# ============================================================

HOME_VIEW_JOINTS = (
    0.3112,
    0.61,
    -0.2931,
    -0.0459,
    -1.8991,
    -1.5737
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
# UI COLORS
# OpenCV uses BGR
# ============================================================

ROI_COLOR = (255, 255, 0)
TRIGGER_WALL_COLOR = (255, 255, 255)
TARGET_COLOR = (0, 255, 255)
LOCKED_COLOR = (0, 165, 255)

TEXT_WHITE = (245, 245, 245)
TEXT_DARK = (0, 0, 0)
PANEL_BG = (35, 35, 35)

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
# CONFIG / FILE HELPERS
# ============================================================

def safety_check():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing Phase 2 config: {CONFIG_FILE}")
    if not H_FILE.exists():
        raise FileNotFoundError(f"Missing homography calibration: {H_FILE}")
    if not PICK_REF_FILE.exists():
        raise FileNotFoundError(f"Missing pick reference: {PICK_REF_FILE}")


def load_phase2_config():
    with open(CONFIG_FILE, "r") as f:
        data = json.load(f)

    return data["roi"], data["priority_point"]


def load_pick_reference():
    with open(PICK_REF_FILE, "r") as f:
        return json.load(f)


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
                "distance_to_priority",
                "status",
            ])


def log_event(det, robot_x, robot_y, status):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            det["class_name"],
            f"{det['confidence']:.4f}",
            det["cx"],
            det["cy"],
            f"{robot_x:.4f}",
            f"{robot_y:.4f}",
            f"{det['priority_distance']:.2f}",
            status,
        ])


# ============================================================
# DETECTION HELPERS
# ============================================================

def get_camera_frame(robot):
    img_compressed = robot.get_img_compressed()
    np_arr = np.frombuffer(img_compressed, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def get_center(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def inside_roi(cx, cy, roi):
    return roi["x1"] <= cx <= roi["x2"] and roi["y1"] <= cy <= roi["y2"]


def wall_x_position(roi):
    return roi["x1"] + LEFT_WALL_TRIGGER_OFFSET_PX


def distance_to_priority(cx, cy, priority):
    dx = cx - priority["x"]
    dy = cy - priority["y"]
    return float((dx * dx + dy * dy) ** 0.5)


def reached_wall(cx, cy, roi):
    if not inside_roi(cx, cy, roi):
        return False

    return cx <= wall_x_position(roi)


def pixel_distance(a, b):
    dx = a["cx"] - b["cx"]
    dy = a["cy"] - b["cy"]
    return float((dx * dx + dy * dy) ** 0.5)


def pixel_to_robot(px, py, H):
    point = np.array([[[px, py]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, H)
    return float(mapped[0][0][0]), float(mapped[0][0][1])


def detect_chips(model, frame, roi, priority):
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    all_detections = []
    valid_roi_detections = []
    wall_detections = []

    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls_id]

            cx, cy = get_center(xyxy)
            in_roi = inside_roi(cx, cy, roi)
            dist = distance_to_priority(cx, cy, priority)
            at_wall = reached_wall(cx, cy, roi)

            det = {
                "xyxy": xyxy,
                "class_name": class_name,
                "confidence": confidence,
                "cx": cx,
                "cy": cy,
                "inside_roi": in_roi,
                "at_wall": at_wall,
                "priority_distance": dist,
            }

            all_detections.append(det)

            if in_roi:
                valid_roi_detections.append(det)

            if at_wall:
                wall_detections.append(det)

    target = None

    if valid_roi_detections:
        target = min(valid_roi_detections, key=lambda d: d["priority_distance"])

    return target, all_detections, valid_roi_detections, wall_detections


# ============================================================
# UI DRAWING
# ============================================================

def draw_rounded_box(frame, x1, y1, x2, y2, color, thickness=2):
    radius = 10

    cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)


def draw_roi(frame, roi):
    cv2.rectangle(
        frame,
        (roi["x1"], roi["y1"]),
        (roi["x2"], roi["y2"]),
        ROI_COLOR,
        3,
    )

    cv2.putText(
        frame,
        "ACCUMULATION ROI",
        (roi["x1"], max(roi["y1"] - 12, 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        ROI_COLOR,
        2,
    )

    wx = wall_x_position(roi)

    cv2.line(
        frame,
        (wx, roi["y1"]),
        (wx, roi["y2"]),
        TRIGGER_WALL_COLOR,
        2,
    )

    cv2.putText(
        frame,
        "DWELL WALL",
        (wx + 4, roi["y1"] + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TRIGGER_WALL_COLOR,
        1,
    )


def draw_detection(frame, det, is_target=False, is_candidate=False, is_locked=False):
    x1, y1, x2, y2 = map(int, det["xyxy"])

    class_name = det["class_name"]
    confidence = det["confidence"]

    if is_locked:
        color = LOCKED_COLOR
        thickness = 4
        tag = "LOCKED"
    elif is_candidate:
        color = TARGET_COLOR
        thickness = 4
        tag = "DWELLING"
    elif is_target:
        color = TARGET_COLOR
        thickness = 3
        tag = "TARGET"
    else:
        color = CLASS_COLORS.get(class_name, DEFAULT_BOX_COLOR)
        thickness = 2
        tag = "CHIP"

    if det["at_wall"] and not is_locked and not is_candidate:
        tag = "AT WALL"

    draw_rounded_box(frame, x1, y1, x2, y2, color, thickness)

    cv2.circle(frame, (det["cx"], det["cy"]), 5, color, -1)

    label = f"{tag} | {class_name} | {confidence:.2f}"

    label_w = max(180, min(430, len(label) * 10))

    cv2.rectangle(
        frame,
        (x1, max(y1 - 30, 0)),
        (x1 + label_w, y1),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x1 + 5, max(y1 - 9, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEXT_DARK,
        2,
    )


def draw_status_panel(
    frame,
    status,
    target,
    chip_count,
    wall_count,
    conveyor_status,
    dwell_time,
):
    h, w = frame.shape[:2]

    panel_h = 106
    cv2.rectangle(frame, (0, 0), (w, panel_h), PANEL_BG, -1)

    cv2.putText(
        frame,
        "PHASE 2: CONVEYOR-ON WALL LOCK PICK",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        TEXT_WHITE,
        2,
    )

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        ROI_COLOR,
        2,
    )

    cv2.putText(
        frame,
        (
            f"CONVEYOR: {conveyor_status} | ROI CHIPS: {chip_count} | "
            f"WALL CHIPS: {wall_count} | DWELL: {dwell_time:.1f}/{WALL_DWELL_TIME_SEC:.1f}s"
        ),
        (16, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        TEXT_WHITE,
        2,
    )

    if target is not None:
        target_text = (
            f"TARGET: {target['class_name']} "
            f"conf={target['confidence']:.2f} "
            f"px=({target['cx']},{target['cy']}) "
            f"d={target['priority_distance']:.1f}"
        )

        cv2.putText(
            frame,
            target_text,
            (w - 620, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            TARGET_COLOR,
            2,
        )


# ============================================================
# CONVEYOR
# ============================================================

def setup_conveyor(robot):
    if not ENABLE_CONVEYOR:
        return None

    try:
        conveyor_id = robot.set_conveyor()
        print(f"Conveyor connected: {conveyor_id}")
        return conveyor_id
    except Exception as e:
        print("Could not connect conveyor automatically.")
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
# ROBOT MOTION
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

    print("STATUS: going to pickup hover...")
    robot.move_pose(*hover_pose)

    print("STATUS: fetching chip...")
    robot.move_pose(*contact_pose)

    print("STATUS: vacuum on...")
    robot.grasp_with_tool()
    time.sleep(0.5)

    print("STATUS: lifting chip...")
    robot.move_pose(*hover_pose)

    print("STATUS: going to midpoint...")
    robot.move_joints(MIDPOINT_JOINTS)

    print("STATUS: transferring to dropoff...")
    robot.move_joints(DROP_JOINTS)

    print("STATUS: releasing chip...")
    robot.release_with_tool()
    time.sleep(0.5)

    print("STATUS: returning to midpoint...")
    robot.move_joints(MIDPOINT_JOINTS)

    print("STATUS: going home / viewing pose...")
    robot.move_joints(HOME_VIEW_JOINTS)


# ============================================================
# TARGET LOCKING
# ============================================================

def choose_wall_candidate(wall_detections):
    if not wall_detections:
        return None

    return min(wall_detections, key=lambda d: d["priority_distance"])


def same_target(candidate, previous_candidate):
    if candidate is None or previous_candidate is None:
        return False

    return pixel_distance(candidate, previous_candidate) <= TARGET_MATCH_DISTANCE_PX


def pixel_distance(a, b):
    dx = a["cx"] - b["cx"]
    dy = a["cy"] - b["cy"]
    return float((dx * dx + dy * dy) ** 0.5)


# ============================================================
# MAIN
# ============================================================

def main():
    safety_check()
    init_log()

    roi, priority = load_phase2_config()

    print("Phase 2 config loaded:")
    print("ROI:", roi)
    print("Hidden priority point:", priority)
    print(f"Wall x-position: {roi['x1']} + {LEFT_WALL_TRIGGER_OFFSET_PX} = {wall_x_position(roi)}")

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
    status = "READY"
    conveyor_status = "ON"

    wall_candidate = None
    wall_candidate_start_time = None
    locked_target = None

    try:
        robot.update_tool()

        print("STATUS: moving to Phase 2 viewing pose...")
        robot.move_joints(HOME_VIEW_JOINTS)

        start_conveyor(robot, conveyor_id)

        print("=" * 60)
        print("PHASE 2 SYSTEM RUNNING")
        print("Conveyor stays ON during operation.")
        print("Chip must stay near dwell wall for 2 seconds before lock.")
        print("Press Q to quit.")
        print("=" * 60)

        while True:
            frame = get_camera_frame(robot)

            if frame is None:
                print("Could not decode camera frame.")
                continue

            target, all_detections, valid_roi_detections, wall_detections = detect_chips(
                model,
                frame,
                roi,
                priority,
            )

            chips_in_roi = len(valid_roi_detections)
            wall_count = len(wall_detections)

            dwell_time = 0.0

            if wall_candidate is not None and wall_candidate_start_time is not None:
                dwell_time = time.time() - wall_candidate_start_time

            if not system_busy:
                current_wall_candidate = choose_wall_candidate(wall_detections)

                if current_wall_candidate is None:
                    wall_candidate = None
                    wall_candidate_start_time = None
                    locked_target = None

                    if chips_in_roi > 0:
                        status = "ACCUMULATING"
                    else:
                        status = "LOCATING"

                else:
                    if wall_candidate is None:
                        wall_candidate = current_wall_candidate
                        wall_candidate_start_time = time.time()
                        status = "WALL CANDIDATE"

                        print(
                            f"STATUS: wall candidate started | "
                            f"{wall_candidate['class_name']} "
                            f"px=({wall_candidate['cx']},{wall_candidate['cy']})"
                        )

                    elif same_target(current_wall_candidate, wall_candidate):
                        dwell_time = time.time() - wall_candidate_start_time
                        status = f"WALL DWELL {dwell_time:.1f}s"

                        wall_candidate = current_wall_candidate

                        if dwell_time >= WALL_DWELL_TIME_SEC:
                            locked_target = current_wall_candidate
                            system_busy = True

                            print("=" * 60)
                            print("STATUS: locked wall target")
                            print(f"Class: {locked_target['class_name']}")
                            print(f"Confidence: {locked_target['confidence']:.2f}")
                            print(f"Pixel center: ({locked_target['cx']}, {locked_target['cy']})")
                            print(f"Dwell time: {dwell_time:.2f}s")
                            print("=" * 60)

                    else:
                        wall_candidate = current_wall_candidate
                        wall_candidate_start_time = time.time()
                        status = "WALL CANDIDATE RESET"

                        print("STATUS: wall candidate reset; new chip near wall.")

            draw_roi(frame, roi)

            for det in all_detections:
                is_candidate = (
                    wall_candidate is not None
                    and pixel_distance(det, wall_candidate) <= TARGET_MATCH_DISTANCE_PX
                )

                is_locked = (
                    locked_target is not None
                    and pixel_distance(det, locked_target) <= TARGET_MATCH_DISTANCE_PX
                )

                draw_detection(
                    frame,
                    det,
                    is_target=(target is not None and det is target),
                    is_candidate=is_candidate,
                    is_locked=is_locked,
                )

            draw_status_panel(
                frame,
                status,
                target,
                chips_in_roi,
                wall_count,
                conveyor_status,
                dwell_time,
            )

            cv2.imshow("Phase 2 Conveyor-On Wall Lock", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if locked_target is not None and system_busy:
                print("=" * 60)
                print("STATUS: locked target confirmed")
                print("Conveyor remains ON. Fetching locked target immediately.")
                print("=" * 60)

                robot_x, robot_y = pixel_to_robot(
                    locked_target["cx"],
                    locked_target["cy"],
                    H,
                )

                print(f"STATUS: mapped locked target to robot X={robot_x:.4f}, Y={robot_y:.4f}")

                log_event(
                    locked_target,
                    robot_x,
                    robot_y,
                    "locked_target_fetch_conveyor_on",
                )

                if ENABLE_ROBOT_MOVEMENT:
                    pick_and_place(robot, robot_x, robot_y, pick_ref)
                    log_event(locked_target, robot_x, robot_y, "pick_complete")
                else:
                    print("Robot movement disabled.")
                    log_event(locked_target, robot_x, robot_y, "movement_disabled")

                print("STATUS: robot home. Conveyor stayed ON.")
                time.sleep(POST_PICK_DELAY_SEC)

                system_busy = False
                wall_candidate = None
                wall_candidate_start_time = None
                locked_target = None

    finally:
        print("Shutting down Phase 2...")

        # Conveyor stops only when exiting program for safety.
        try:
            stop_conveyor(robot, conveyor_id)
        except Exception:
            pass

        cv2.destroyAllWindows()
        robot.close_connection()
        print("Disconnected.")


if __name__ == "__main__":
    main()