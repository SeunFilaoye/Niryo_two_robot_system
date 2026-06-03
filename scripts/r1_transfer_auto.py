import cv2
import json
import numpy as np
from pathlib import Path
from pyniryo import NiryoRobot, ConveyorDirection
from ultralytics import YOLO
import time

# -------------------------
# SETTINGS
# -------------------------
ROBOT_IP = "192.168.0.229"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

CALIB_DIR = PROJECT_ROOT / "calibration_r1"
CAMERA_CALIB_JSON = CALIB_DIR / "workspace_calibration.json"
ROBOT_POINTS_JSON = CALIB_DIR / "robot_workspace_points.json"
HOVER_REF_JSON = CALIB_DIR / "workspace_hover_reference.json"

CONFIDENCE = 0.35
PANEL_WIDTH = 420

# -------------------------
# ROBOT 1 POSES
# -------------------------
OBSERVATION_JOINTS = (0.5243, -0.2777, -0.0568, -0.0643, -1.3944, -0.9509)
HOME_SAFE_JOINTS = (0.5243, 0.3282, -1.34, -0.0643, -0.6888, -0.9494)
SAFE_TRANSFER_JOINTS = (1.824, 0.1252, -0.4052, -0.075, -1.2088, -0.954)
CONVEYOR2_HOVER_JOINTS = (2.8635, 0.0252, -0.6416, -0.0413, -1.0447, -0.9509)
CONVEYOR2_DROP_JOINTS = (2.9, -0.4247, -0.6537, -0.072, -0.6888, -0.9402)

# -------------------------
# ROI / DISPLAY
# -------------------------
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 180, 100, 500, 380
SHOW_ROI = True
SHOW_CALIBRATION_POLYGON = True

# -------------------------
# AUTO STOP / SETTLE
# -------------------------
SETTLE_TIME_SEC = 0.45
FINAL_MATCH_MAX_DIST_PX = 140

# -------------------------
# PICK SETTINGS
# -------------------------
DESCEND_OFFSET_M = 0.05
PAUSE_AT_PICK_SEC = 0.5
GRIPPER_WAIT_SEC = 0.8
RELEASE_WAIT_SEC = 0.6

# -------------------------
# CONVEYOR SETTINGS
# -------------------------
CONVEYOR_SPEED = 40
CURRENT_DIRECTION = ConveyorDirection.FORWARD

# -------------------------
# VALID CLASSES
# -------------------------
VALID_CLASSES = {
    "blue_square",
    "blue_circle",
    "red_square",
    "red_circle",
    "green_square",
    "green_circle",
}

# -------------------------
# COLORS
# -------------------------
CLASS_COLORS = {
    "red_square": (203, 192, 255),
    "red_circle": (0, 0, 139),
    "blue_square": (255, 191, 0),
    "blue_circle": (211, 0, 148),
    "green_square": (144, 238, 144),
    "green_circle": (0, 100, 0),
}

# -------------------------
# GLOBALS
# -------------------------
conveyor_running = False
conveyor_id = None
auto_mode = False
locked_target = None


def decode_robot_image(img_compressed):
    np_arr = np.frombuffer(img_compressed, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def load_camera_calibration():
    with open(CAMERA_CALIB_JSON, "r") as f:
        data = json.load(f)

    workspace_width_mm = float(data["workspace_width_mm"])
    workspace_height_mm = float(data["workspace_height_mm"])

    image_points = np.array([
        data["image_points"]["top_left"],
        data["image_points"]["top_right"],
        data["image_points"]["bottom_right"],
        data["image_points"]["bottom_left"],
    ], dtype=np.float32)

    world_points = np.array([
        [0.0, 0.0],
        [workspace_width_mm, 0.0],
        [workspace_width_mm, workspace_height_mm],
        [0.0, workspace_height_mm],
    ], dtype=np.float32)

    H_img_to_ws = cv2.getPerspectiveTransform(image_points, world_points)

    return {
        "workspace_width_mm": workspace_width_mm,
        "workspace_height_mm": workspace_height_mm,
        "image_points": image_points,
        "H_img_to_ws": H_img_to_ws,
    }


def load_robot_workspace_mapping():
    with open(ROBOT_POINTS_JSON, "r") as f:
        data = json.load(f)

    pts = data["points"]
    return np.array([
        [pts["top_left"]["x_m"], pts["top_left"]["y_m"]],
        [pts["top_right"]["x_m"], pts["top_right"]["y_m"]],
        [pts["bottom_right"]["x_m"], pts["bottom_right"]["y_m"]],
        [pts["bottom_left"]["x_m"], pts["bottom_left"]["y_m"]],
    ], dtype=np.float32)


def load_hover_reference():
    with open(HOVER_REF_JSON, "r") as f:
        data = json.load(f)
    return data["hover_reference"]


def pixel_to_workspace_homography(cx, cy, H_img_to_ws):
    src = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H_img_to_ws)
    return float(dst[0][0][0]), float(dst[0][0][1])


def build_workspace_to_robot_transform(workspace_width_mm, workspace_height_mm, robot_xy):
    workspace_pts = np.array([
        [0.0, 0.0],
        [workspace_width_mm, 0.0],
        [workspace_width_mm, workspace_height_mm],
        [0.0, workspace_height_mm],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(workspace_pts, robot_xy)


def workspace_to_robot_xy(x_mm, y_mm, H_ws_to_robot):
    src = np.array([[[float(x_mm), float(y_mm)]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H_ws_to_robot)
    return float(dst[0][0][0]), float(dst[0][0][1])


def build_target(class_name, x1, y1, x2, y2, cx, cy, H_img_to_ws, H_ws_to_robot, hover_ref):
    x_mm, y_mm = pixel_to_workspace_homography(cx, cy, H_img_to_ws)
    robot_x_m, robot_y_m = workspace_to_robot_xy(x_mm, y_mm, H_ws_to_robot)

    hover_z = float(hover_ref["z_m"])
    pick_z = hover_z - DESCEND_OFFSET_M

    return {
        "class_name": class_name,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx": cx,
        "cy": cy,
        "x_mm": x_mm,
        "y_mm": y_mm,
        "robot_x_m": robot_x_m,
        "robot_y_m": robot_y_m,
        "robot_z_m": hover_z,
        "pick_z_m": pick_z,
        "roll_rad": float(hover_ref["roll_rad"]),
        "pitch_rad": float(hover_ref["pitch_rad"]),
        "yaw_rad": float(hover_ref["yaw_rad"]),
    }


def extract_roi_detections(results, model):
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        class_name = model.names[cls_id]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if class_name in VALID_CLASSES and ROI_X1 <= cx <= ROI_X2 and ROI_Y1 <= cy <= ROI_Y2:
            detections.append((class_name, x1, y1, x2, y2, cx, cy))
    return detections


def choose_first_target_in_roi(detections, H_img_to_ws, H_ws_to_robot, hover_ref):
    if not detections:
        return None

    # "first in" = closest to the conveyor stop section / ROI front
    # For now, use smallest x distance to ROI center-left stop area.
    stop_x = ROI_X1 + 20
    candidates = []
    for det in detections:
        class_name, x1, y1, x2, y2, cx, cy = det
        target = build_target(class_name, x1, y1, x2, y2, cx, cy, H_img_to_ws, H_ws_to_robot, hover_ref)
        score = abs(cx - stop_x)
        candidates.append((score, target))

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def find_final_target_after_stop(robot, model, prev_target, H_img_to_ws, H_ws_to_robot, hover_ref):
    time.sleep(SETTLE_TIME_SEC)

    img_compressed = robot.get_img_compressed()
    if img_compressed is None:
        return None, None

    frame = decode_robot_image(img_compressed)
    if frame is None:
        return None, None

    results = model(frame, conf=CONFIDENCE, verbose=False)[0]
    detections = extract_roi_detections(results, model)

    candidates = []
    for det in detections:
        class_name, x1, y1, x2, y2, cx, cy = det
        if class_name != prev_target["class_name"]:
            continue

        dist = ((cx - prev_target["cx"]) ** 2 + (cy - prev_target["cy"]) ** 2) ** 0.5
        if dist <= FINAL_MATCH_MAX_DIST_PX:
            target = build_target(class_name, x1, y1, x2, y2, cx, cy, H_img_to_ws, H_ws_to_robot, hover_ref)
            candidates.append((dist, target))

    if not candidates:
        return frame, None

    candidates.sort(key=lambda item: item[0])
    return frame, candidates[0][1]


def start_conveyor(robot):
    global conveyor_running
    robot.run_conveyor(conveyor_id, speed=CONVEYOR_SPEED, direction=CURRENT_DIRECTION)
    conveyor_running = True


def stop_conveyor(robot):
    global conveyor_running
    robot.stop_conveyor(conveyor_id)
    conveyor_running = False


def move_to_hover_reference(robot, hover_ref):
    robot.move_pose(
        float(hover_ref["x_m"]),
        float(hover_ref["y_m"]),
        float(hover_ref["z_m"]),
        float(hover_ref["roll_rad"]),
        float(hover_ref["pitch_rad"]),
        float(hover_ref["yaw_rad"]),
    )


def move_above_target(robot, target):
    robot.move_pose(
        target["robot_x_m"],
        target["robot_y_m"],
        target["robot_z_m"],
        target["roll_rad"],
        target["pitch_rad"],
        target["yaw_rad"],
    )


def move_to_pick_height(robot, target):
    robot.move_pose(
        target["robot_x_m"],
        target["robot_y_m"],
        target["pick_z_m"],
        target["roll_rad"],
        target["pitch_rad"],
        target["yaw_rad"],
    )


def run_pick_and_transfer(robot, hover_ref, target):
    robot.update_tool()

    # pick
    robot.release_with_tool()
    time.sleep(0.4)

    move_to_hover_reference(robot, hover_ref)
    move_above_target(robot, target)
    move_to_pick_height(robot, target)
    time.sleep(PAUSE_AT_PICK_SEC)

    robot.grasp_with_tool()
    time.sleep(GRIPPER_WAIT_SEC)

    move_above_target(robot, target)

    # transfer
    robot.move_joints(*SAFE_TRANSFER_JOINTS)
    robot.move_joints(*CONVEYOR2_HOVER_JOINTS)
    robot.move_joints(*CONVEYOR2_DROP_JOINTS)

    robot.release_with_tool()
    time.sleep(RELEASE_WAIT_SEC)

    robot.move_joints(*CONVEYOR2_HOVER_JOINTS)
    robot.move_joints(*SAFE_TRANSFER_JOINTS)
    robot.move_joints(*OBSERVATION_JOINTS)


def build_panel(frame_height, live_target, locked_target):
    panel = np.zeros((frame_height, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (35, 35, 35)

    cv2.putText(panel, "Robot 1 Transfer Auto", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(panel, (15, 50), (PANEL_WIDTH - 15, 50), (180, 180, 180), 1)

    y = 82
    lines = [
        f"Auto mode: {auto_mode}",
        f"Conveyor running: {conveyor_running}",
        f"Speed: {CONVEYOR_SPEED}",
        f"Settle sec: {SETTLE_TIME_SEC}",
    ]
    for line in lines:
        cv2.putText(panel, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 2)
        y += 24

    active = locked_target if locked_target is not None else live_target
    y += 12
    if active is not None:
        cv2.putText(panel, f"Target: {active['class_name']}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        y += 28
        cv2.putText(panel, f"Workspace X: {active['x_mm']:.1f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 2)
        y += 22
        cv2.putText(panel, f"Workspace Y: {active['y_mm']:.1f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 2)
        y += 22
        cv2.putText(panel, f"Robot X: {active['robot_x_m']:.4f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 2)
        y += 22
        cv2.putText(panel, f"Robot Y: {active['robot_y_m']:.4f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 2)

    y += 24
    controls = [
        "Keys:",
        "m = auto on/off",
        "g = start conveyor",
        "x = stop conveyor",
        "p = manual pick/transfer",
        "u = clear locked target",
        "q = quit",
    ]
    for line in controls:
        cv2.putText(panel, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 2)
        y += 21

    return panel


def main():
    global locked_target, conveyor_running, conveyor_id, auto_mode

    cam_calib = load_camera_calibration()
    hover_ref = load_hover_reference()
    robot_xy = load_robot_workspace_mapping()
    H_ws_to_robot = build_workspace_to_robot_transform(
        cam_calib["workspace_width_mm"],
        cam_calib["workspace_height_mm"],
        robot_xy
    )
    H_img_to_ws = cam_calib["H_img_to_ws"]
    image_points = cam_calib["image_points"].astype(int)

    model = YOLO(str(MODEL_PATH))
    robot = None
    window_name = "Robot 1 Transfer Auto"

    try:
        robot = NiryoRobot(ROBOT_IP)
        robot.clear_collision_detected()
        robot.update_tool()

        conveyors = robot.get_connected_conveyors_id()
        if not conveyors:
            print("No conveyor found for robot 1.")
            return

        conveyor_id = conveyors[0]
        robot.move_joints(*OBSERVATION_JOINTS)

        cv2.namedWindow(window_name)

        while True:
            img_compressed = robot.get_img_compressed()
            if img_compressed is None:
                continue

            frame = decode_robot_image(img_compressed)
            if frame is None:
                continue

            annotated = frame.copy()
            results = model(frame, conf=CONFIDENCE, verbose=False)[0]
            detections = extract_roi_detections(results, model)
            live_target = choose_first_target_in_roi(detections, H_img_to_ws, H_ws_to_robot, hover_ref)

            # AUTO: stop as soon as chip is detected in ROI
            if auto_mode and conveyor_running and locked_target is None and live_target is not None:
                print(f"[R1 AUTO] detected {live_target['class_name']} in ROI -> stopping conveyor")
                stop_conveyor(robot)
                locked_target = dict(live_target)

                settled_frame, final_target = find_final_target_after_stop(
                    robot, model, locked_target, H_img_to_ws, H_ws_to_robot, hover_ref
                )

                if final_target is not None:
                    locked_target = dict(final_target)
                    if settled_frame is not None:
                        annotated = settled_frame.copy()

                print(f"[R1 AUTO] picking {locked_target['class_name']} and transferring to conveyor 2")
                run_pick_and_transfer(robot, hover_ref, locked_target)

                locked_target = None

                if auto_mode:
                    print("[R1 AUTO] restarting conveyor 1")
                    start_conveyor(robot)

            # draw detections
            for det in detections:
                class_name, x1, y1, x2, y2, cx, cy = det
                color = CLASS_COLORS.get(class_name, (255, 255, 255))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.circle(annotated, (cx, cy), 4, color, -1)
                cv2.putText(annotated, class_name, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if SHOW_ROI:
                cv2.rectangle(annotated, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 255, 255), 2)
                cv2.putText(annotated, "ROI", (ROI_X1, max(ROI_Y1 - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if SHOW_CALIBRATION_POLYGON:
                pts = image_points.reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
                labels = ["TL", "TR", "BR", "BL"]
                for i, pt in enumerate(image_points):
                    cv2.circle(annotated, tuple(pt), 5, (0, 255, 255), -1)
                    cv2.putText(annotated, labels[i], (pt[0] + 8, pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if locked_target is not None:
                cv2.circle(annotated, (locked_target["cx"], locked_target["cy"]), 10, (255, 255, 255), 2)
                cv2.putText(annotated, f"LOCKED: {locked_target['class_name']}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            elif live_target is not None:
                cv2.circle(annotated, (live_target["cx"], live_target["cy"]), 10, (255, 255, 255), 2)
                cv2.putText(annotated, f"LIVE: {live_target['class_name']}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            panel = build_panel(annotated.shape[0], live_target, locked_target)
            combined = cv2.hconcat([annotated, panel])
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("g"):
                start_conveyor(robot)
            elif key == ord("x"):
                stop_conveyor(robot)
            elif key == ord("m"):
                auto_mode = not auto_mode
                print(f"Robot 1 auto mode: {auto_mode}")
                if auto_mode and not conveyor_running:
                    start_conveyor(robot)
                elif not auto_mode and conveyor_running:
                    stop_conveyor(robot)
            elif key == ord("u"):
                locked_target = None
            elif key == ord("p"):
                if live_target is not None:
                    stop_conveyor(robot)
                    locked_target = dict(live_target)
                    _, final_target = find_final_target_after_stop(
                        robot, model, locked_target, H_img_to_ws, H_ws_to_robot, hover_ref
                    )
                    if final_target is not None:
                        locked_target = dict(final_target)
                    run_pick_and_transfer(robot, hover_ref, locked_target)
                    locked_target = None
            elif key == ord("h"):
                robot.move_joints(*HOME_SAFE_JOINTS)
            elif key == ord("o"):
                robot.move_joints(*OBSERVATION_JOINTS)
            elif key == ord("s"):
                robot.move_joints(*SAFE_TRANSFER_JOINTS)

    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        cv2.destroyAllWindows()
        if robot is not None:
            try:
                if conveyor_running and conveyor_id is not None:
                    robot.stop_conveyor(conveyor_id)
            except Exception:
                pass
            try:
                robot.close_connection()
            except Exception:
                pass
        print("Connection closed.")


if __name__ == "__main__":
    main()