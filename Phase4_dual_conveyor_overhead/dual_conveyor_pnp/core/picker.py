import json
import time
from pathlib import Path

import cv2
import numpy as np
from pyniryo import JointsPosition, PoseObject


from common.settings import get as get_settings  # --robot / --conveyor

S = get_settings(need_conveyor=True)

ROBOT_NAME = S.ROBOT_NAME
ROBOT_IP = S.ROBOT_IP

ROBOT_DIR = S.ROBOT_DIR

H_FILE = ROBOT_DIR / "calibration" / "homography_matrix.npy"
PICK_REF_FILE = ROBOT_DIR / "calibration" / "pick_reference.json"
CALIBRATION_FILE = ROBOT_DIR / "calibration" / "calibration_points.json"


HOME_VIEW_JOINTS = S.HOME_VIEW_JOINTS

MIDPOINT_JOINTS = S.MIDPOINT_JOINTS

DROP_JOINTS = S.DROP_JOINTS


# The homography covers the whole conveyor plane, so any chip in the 08 ROI
# may be picked. Robot 1 used a hard-coded box; here the only limit is the
# arm's reach (distance of the target from the robot base, config/system.json).
MIN_REACH_M = S.SYSTEM["robot_runtime"]["min_reach_m"]
MAX_REACH_M = S.SYSTEM["robot_runtime"]["max_reach_m"]


class Picker:
    def __init__(self):
        from common.settings import require_midpoint_drop
        require_midpoint_drop(S)
        self._validate_files()
        self._validate_view_calibration()

        self.H = np.load(H_FILE)

        if self.H.shape != (3, 3):
            raise ValueError(
                f"Invalid homography shape: {self.H.shape}"
            )

        if not np.all(np.isfinite(self.H)):
            raise ValueError("Homography contains invalid values.")

        with PICK_REF_FILE.open("r", encoding="utf-8") as file:
            self.pick_ref = json.load(file)

        self._validate_pick_reference()

    @staticmethod
    def _validate_files():
        required = (
            H_FILE,
            PICK_REF_FILE,
            CALIBRATION_FILE,
        )

        missing = [path for path in required if not path.exists()]

        if missing:
            text = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"Missing picker files:\n{text}"
            )

    def _validate_view_calibration(self):
        with CALIBRATION_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        saved_robot_name = data.get("robot_name")
        saved_robot_ip = data.get("robot_ip")
        saved_home = data.get("home_view_joints")

        if saved_robot_name not in (None, ROBOT_NAME):
            raise ValueError(
                f"Calibration belongs to {saved_robot_name}."
            )

        if saved_robot_ip not in (None, ROBOT_IP):
            raise ValueError(
                f"Calibration IP is {saved_robot_ip}."
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
                    "HOME_VIEW_JOINTS do not match the "
                    "workspace calibration viewing pose."
                )

    def _validate_pick_reference(self):
        saved_robot_name = self.pick_ref.get("robot_name")
        saved_robot_ip = self.pick_ref.get("robot_ip")

        if saved_robot_name not in (None, ROBOT_NAME):
            raise ValueError(
                f"Pick reference belongs to {saved_robot_name}."
            )

        if saved_robot_ip not in (None, ROBOT_IP):
            raise ValueError(
                f"Pick reference IP is {saved_robot_ip}."
            )

        required = (
            "hover_z",
            "contact_z",
            "roll",
            "pitch",
            "yaw",
        )

        missing = [
            key for key in required
            if key not in self.pick_ref
        ]

        if missing:
            raise KeyError(
                f"Pick reference missing fields: {missing}"
            )

        if (
            float(self.pick_ref["hover_z"])
            <= float(self.pick_ref["contact_z"])
        ):
            raise ValueError(
                "Hover Z must be above contact Z."
            )

    def pixel_to_robot(self, px, py):
        point = np.array(
            [[[float(px), float(py)]]],
            dtype=np.float32,
        )

        mapped = cv2.perspectiveTransform(
            point,
            self.H,
        )

        # Same correction as PICK_X/Y_OFFSET_M on Robots 2/3/5.
        # Set "pick_offset_m": [dx, dy] in pick_reference.json (default 0, 0).
        offset_x, offset_y = self.pick_ref.get("pick_offset_m", [0.0, 0.0])

        return (
            float(mapped[0][0][0]) + float(offset_x),
            float(mapped[0][0][1]) + float(offset_y),
        )

    @staticmethod
    def is_safe(x, y):
        reach = float(np.hypot(x, y))
        return MIN_REACH_M <= reach <= MAX_REACH_M

    def build_poses(self, robot_x, robot_y):
        hover_pose = PoseObject(
            robot_x,
            robot_y,
            float(self.pick_ref["hover_z"]),
            float(self.pick_ref["roll"]),
            float(self.pick_ref["pitch"]),
            float(self.pick_ref["yaw"]),
        )

        contact_pose = PoseObject(
            robot_x,
            robot_y,
            float(self.pick_ref["contact_z"]),
            float(self.pick_ref["roll"]),
            float(self.pick_ref["pitch"]),
            float(self.pick_ref["yaw"]),
        )

        return hover_pose, contact_pose

    @staticmethod
    def go_home(robot):
        print("ROBOT: moving to viewing pose...")
        robot.move(JointsPosition(*HOME_VIEW_JOINTS))

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

        vacuum_active = False

        try:
            print("ROBOT: moving to pickup hover...")
            robot.move(hover_pose)

            print("ROBOT: descending to contact...")
            robot.move(contact_pose)

            print("ROBOT: vacuum on...")
            robot.grasp_with_tool()
            vacuum_active = True
            time.sleep(0.75)

            print("ROBOT: lifting...")
            robot.move(hover_pose)

            print("ROBOT: moving through midpoint...")
            robot.move(JointsPosition(*MIDPOINT_JOINTS))

            print("ROBOT: moving to drop-off...")
            robot.move(JointsPosition(*DROP_JOINTS))

            print("ROBOT: releasing chip...")
            robot.release_with_tool()
            vacuum_active = False
            time.sleep(0.5)

            print("ROBOT: returning through midpoint...")
            robot.move(JointsPosition(*MIDPOINT_JOINTS))

            print("ROBOT: returning to viewing pose...")
            robot.move(JointsPosition(*HOME_VIEW_JOINTS))

            print("ROBOT: pick-and-place complete.")
            return True

        except Exception as error:
            print(f"ROBOT PICK ERROR: {error}")
            return False

        finally:
            if vacuum_active:
                try:
                    robot.release_with_tool()
                except Exception:
                    pass