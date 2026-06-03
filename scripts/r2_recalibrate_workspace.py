import cv2
import json
import numpy as np
from pathlib import Path
from pyniryo import NiryoRobot

ROBOT_IP = "192.168.0.201"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = PROJECT_ROOT / "calibration_r2"
CALIB_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_CALIB_JSON = CALIB_DIR / "workspace_calibration.json"
ROBOT_POINTS_JSON = CALIB_DIR / "robot_workspace_points.json"
HOVER_REF_JSON = CALIB_DIR / "workspace_hover_reference.json"

OBSERVATION_JOINTS = (-2.9121, 0.7115, -0.687, -0.0121, -1.6445, -0.1057)

# same workspace size for now
WORKSPACE_WIDTH_MM = 165.1
WORKSPACE_HEIGHT_MM = 203.2

WINDOW_NAME = "Robot 2 Recalibration"

clicked_points = []
robot_points = {}
hover_reference = None


def decode_robot_image(img_compressed):
    np_arr = np.frombuffer(img_compressed, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def pose_to_dict(pose):
    return {
        "x_m": float(pose.x),
        "y_m": float(pose.y),
        "z_m": float(pose.z),
        "roll_rad": float(pose.roll),
        "pitch_rad": float(pose.pitch),
        "yaw_rad": float(pose.yaw),
    }


def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"Saved image point {len(clicked_points)}: ({x}, {y})")


def draw_overlay(frame):
    display = frame.copy()

    instructions = [
        "Click corners in order: TL, TR, BR, BL",
        "Record robot poses: t=TL  y=TR  u=BR  i=BL",
        "Record hover reference: h",
        "Save all: s   Reset clicks: r   Quit: q",
    ]

    y = 28
    for line in instructions:
        cv2.putText(display, line, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        y += 24

    labels = ["TL", "TR", "BR", "BL"]
    for idx, pt in enumerate(clicked_points):
        cv2.circle(display, pt, 6, (0, 255, 255), -1)
        cv2.putText(display, labels[idx], (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if len(clicked_points) > 1:
        for i in range(len(clicked_points) - 1):
            cv2.line(display, clicked_points[i], clicked_points[i + 1], (0, 255, 255), 2)
    if len(clicked_points) == 4:
        cv2.line(display, clicked_points[3], clicked_points[0], (0, 255, 255), 2)

    return display


def main():
    global hover_reference, clicked_points, robot_points
    robot = None

    try:
        print(f"Connecting to robot at {ROBOT_IP} ...")
        robot = NiryoRobot(ROBOT_IP)
        robot.clear_collision_detected()
        print("Connected.")

        print("Moving to observation pose...")
        robot.move_joints(*OBSERVATION_JOINTS)

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

        while True:
            frame = decode_robot_image(robot.get_img_compressed())
            if frame is None:
                continue

            display = draw_overlay(frame)
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                clicked_points = []
                print("Reset clicked points.")
            elif key == ord("t"):
                robot_points["top_left"] = pose_to_dict(robot.get_pose())
                print("Saved top_left")
            elif key == ord("y"):
                robot_points["top_right"] = pose_to_dict(robot.get_pose())
                print("Saved top_right")
            elif key == ord("u"):
                robot_points["bottom_right"] = pose_to_dict(robot.get_pose())
                print("Saved bottom_right")
            elif key == ord("i"):
                robot_points["bottom_left"] = pose_to_dict(robot.get_pose())
                print("Saved bottom_left")
            elif key == ord("h"):
                hover_reference = pose_to_dict(robot.get_pose())
                print("Saved hover reference")
            elif key == ord("s"):
                if len(clicked_points) != 4:
                    print("Need 4 image points.")
                    continue
                if set(robot_points.keys()) != {"top_left", "top_right", "bottom_right", "bottom_left"}:
                    print("Need all 4 robot points.")
                    continue
                if hover_reference is None:
                    print("Need hover reference.")
                    continue

                with open(CAMERA_CALIB_JSON, "w") as f:
                    json.dump({
                        "robot_ip": ROBOT_IP,
                        "workspace_width_mm": WORKSPACE_WIDTH_MM,
                        "workspace_height_mm": WORKSPACE_HEIGHT_MM,
                        "image_points": {
                            "top_left": list(clicked_points[0]),
                            "top_right": list(clicked_points[1]),
                            "bottom_right": list(clicked_points[2]),
                            "bottom_left": list(clicked_points[3]),
                        }
                    }, f, indent=4)

                with open(ROBOT_POINTS_JSON, "w") as f:
                    json.dump({
                        "robot_ip": ROBOT_IP,
                        "points": robot_points
                    }, f, indent=4)

                with open(HOVER_REF_JSON, "w") as f:
                    json.dump({
                        "robot_ip": ROBOT_IP,
                        "hover_reference": hover_reference
                    }, f, indent=4)

                print("Saved all robot 2 calibration files.")

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        cv2.destroyAllWindows()
        if robot is not None:
            try:
                robot.close_connection()
            except Exception:
                pass
        print("Connection closed.")


if __name__ == "__main__":
    main()