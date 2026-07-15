import json
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

H_FILE = PROJECT_ROOT / "calibration" / "homography_matrix.npy"
PICK_REF_FILE = PROJECT_ROOT / "calibration" / "pick_reference.json"


HOME_VIEW_JOINTS = (
    0.3112,
    0.61,
    -0.2931,
    -0.0459,
    -1.8991,
    -1.5737,
)

MIDPOINT_JOINTS = (
    1.3005,
    -0.0217,
    -0.3825,
    -0.052,
    -1.2104,
    -1.744,
)

DROP_JOINTS = (
    2.286,
    -0.426,
    -0.559,
    0.099,
    -0.489,
    -1.655,
)


SAFE_X_MIN = 0.080
SAFE_X_MAX = 0.300
SAFE_Y_MIN = -0.250
SAFE_Y_MAX = 0.240


class Picker:
    def __init__(self):
        self.H = np.load(H_FILE)

        with open(PICK_REF_FILE, "r", encoding="utf-8") as file:
            self.pick_ref = json.load(file)

    def pixel_to_robot(self, px, py):
        point = np.array(
            [[[float(px), float(py)]]],
            dtype=np.float32,
        )

        mapped = cv2.perspectiveTransform(point, self.H)

        return (
            float(mapped[0][0][0]),
            float(mapped[0][0][1]),
        )

    @staticmethod
    def is_safe(x, y):
        return (
            SAFE_X_MIN <= x <= SAFE_X_MAX
            and SAFE_Y_MIN <= y <= SAFE_Y_MAX
        )

    def build_poses(self, robot_x, robot_y):
        hover_reference = self.pick_ref["hover_pose"]
        contact_reference = self.pick_ref["contact_pose"]

        hover_pose = (
            robot_x,
            robot_y,
            float(hover_reference["z"]),
            float(hover_reference["roll"]),
            float(hover_reference["pitch"]),
            float(hover_reference["yaw"]),
        )

        contact_pose = (
            robot_x,
            robot_y,
            float(contact_reference["z"]),
            float(contact_reference["roll"]),
            float(contact_reference["pitch"]),
            float(contact_reference["yaw"]),
        )

        return hover_pose, contact_pose

    @staticmethod
    def go_home(robot):
        print("ROBOT: moving to viewing pose...")
        robot.move_joints(HOME_VIEW_JOINTS)

    def pick_and_place(self, robot, chip):
        robot_x, robot_y = self.pixel_to_robot(
            chip["cx"],
            chip["cy"],
        )

        print(
            f"TARGET PIXEL: ({chip['cx']}, {chip['cy']})"
        )
        print(
            f"TARGET ROBOT: X={robot_x:.4f}, Y={robot_y:.4f}"
        )

        if not self.is_safe(robot_x, robot_y):
            print("UNSAFE TARGET")
            return False

        hover_pose, contact_pose = self.build_poses(
            robot_x,
            robot_y,
        )

        print(
            f"HOVER Z={hover_pose[2]:.4f}, "
            f"CONTACT Z={contact_pose[2]:.4f}"
        )

        print("ROBOT: moving to pickup hover...")
        robot.move_pose(*hover_pose)

        print("ROBOT: descending to contact...")
        robot.move_pose(*contact_pose)

        print("ROBOT: vacuum on...")
        robot.grasp_with_tool()
        time.sleep(0.75)

        print("ROBOT: lifting...")
        robot.move_pose(*hover_pose)

        print("ROBOT: moving through midpoint...")
        robot.move_joints(MIDPOINT_JOINTS)

        print("ROBOT: moving to drop-off...")
        robot.move_joints(DROP_JOINTS)

        print("ROBOT: releasing chip...")
        robot.release_with_tool()
        time.sleep(0.5)

        print("ROBOT: returning through midpoint...")
        robot.move_joints(MIDPOINT_JOINTS)

        print("ROBOT: returning to viewing pose...")
        robot.move_joints(HOME_VIEW_JOINTS)

        print("ROBOT: pick-and-place complete.")
        return True