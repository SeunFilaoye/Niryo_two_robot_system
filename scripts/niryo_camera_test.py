import cv2
import numpy as np
from pyniryo import NiryoRobot
from pyniryo.vision import uncompress_image, undistort_image

ROBOT_IP = "192.168.0.223"

# Your saved poses in joint radians
HOME_JOINTS = (-0.002, 0.532, -1.338, 0.042, -0.391, -0.208)
OBSERVATION_JOINTS = (0.0266, 0.5721, -0.8476, -0.0229, -1.5616, 0.0046)
HOVER_JOINTS = (0.0266, -0.1277, -1.0203, -0.0397, -0.5063, 0.1274)
SAFE_RETREAT_JOINTS = (1.474, 0.4569, -0.9143, 0.0813, -0.5047, 0.1182)

WINDOW_NAME = "Niryo Wrist Camera Test"


def draw_overlay(frame, message_lines):
    y = 30
    for line in message_lines:
        cv2.putText(
            frame,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        y += 28
    return frame


def main():
    robot = None
    try:
        print(f"Connecting to robot at {ROBOT_IP} ...")
        robot = NiryoRobot(ROBOT_IP)
        print("Connected.")

        # Camera intrinsics for optional undistortion
        print("Getting camera intrinsics...")
        mtx, dist = robot.get_camera_intrinsics()
        print("Camera intrinsics received.")

        print("Moving to observation pose...")
        robot.move_joints(*OBSERVATION_JOINTS)
        print("Robot is at observation pose.")

        print("Controls:")
        print("  q = quit")
        print("  h = move to Home")
        print("  o = move to Observation")
        print("  v = move to Hover")
        print("  s = move to Safe Retreat")
        print("  u = toggle undistort on/off")

        show_undistort = True

        while True:
            img_compressed = robot.get_img_compressed()
            if img_compressed is None:
                print("No image received from robot camera.")
                continue

            frame = uncompress_image(img_compressed)
            if frame is None:
                print("Failed to decode robot camera frame.")
                continue

            if show_undistort:
                display_frame = undistort_image(frame, mtx, dist)
            else:
                display_frame = frame.copy()

            status_lines = [
                f"Robot IP: {ROBOT_IP}",
                f"Undistort: {show_undistort}",
                "q quit | h home | o observation | v hover | s safe retreat | u undistort",
            ]

            display_frame = draw_overlay(display_frame, status_lines)
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("h"):
                print("Moving to Home pose...")
                robot.move_joints(*HOME_JOINTS)
            elif key == ord("o"):
                print("Moving to Observation pose...")
                robot.move_joints(*OBSERVATION_JOINTS)
            elif key == ord("v"):
                print("Moving to Hover pose...")
                robot.move_joints(*HOVER_JOINTS)
            elif key == ord("s"):
                print("Moving to Safe Retreat pose...")
                robot.move_joints(*SAFE_RETREAT_JOINTS)
            elif key == ord("u"):
                show_undistort = not show_undistort
                print(f"Undistort set to: {show_undistort}")

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