"""One robot + one conveyor = one station.

A station bundles everything trained for that conveyor's view pose:
view / midpoint / drop poses, the pixel -> robot homography, hover and
contact TCP references, and the wrist-camera polygon ROI.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from pyniryo import JointsPosition, PoseObject

from common.config import StationPaths, read_json

REQUIRED_POSES = ("view", "midpoint", "drop")


class StationNotReady(RuntimeError):
    pass


def missing_station_files(paths: StationPaths, need_roi: bool = True) -> list[str]:
    required = [paths.poses, paths.homography, paths.pick_reference, paths.calibration]
    if need_roi:
        required.append(paths.roi)
    return [str(path) for path in required if not path.exists()]


def load_poses(paths: StationPaths, required=REQUIRED_POSES) -> dict:
    if not paths.poses.exists():
        raise StationNotReady(
            f"Missing {paths.poses}. Save the view pose, then run 04b_record_midpoint_drop.py."
        )
    data = read_json(paths.poses)
    missing = [name for name in required if name not in data.get("poses", {})]
    if missing:
        raise StationNotReady(
            f"{paths.poses} is missing poses: {missing}. "
            "Run 04b_record_midpoint_drop.py first."
        )
    return data["poses"]


def load_view_joints(paths: StationPaths) -> list[float]:
    """The station's VIEW pose; the only pose most training steps need."""
    return load_poses(paths, required=("view",))["view"]["joints"]


def load_polygon(paths: StationPaths) -> np.ndarray:
    if not paths.roi.exists():
        raise StationNotReady(
            f"Missing {paths.roi}. Run 08_set_polygon_roi_and_priority.py first."
        )
    points = read_json(paths.roi).get("polygon_roi")
    if not isinstance(points, list) or len(points) < 3:
        raise StationNotReady(f"{paths.roi} needs polygon_roi with 3+ points.")
    return np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))


class Station:
    def __init__(self, robot: str, conveyor: str, need_roi: bool = True):
        self.robot = robot
        self.conveyor = conveyor
        self.paths = StationPaths(robot, conveyor)

        missing = missing_station_files(self.paths, need_roi)
        if missing:
            raise StationNotReady(
                f"{robot}/{conveyor} is not trained yet. Missing:\n  "
                + "\n  ".join(missing)
            )

        self.poses = load_poses(self.paths)
        self.view_joints = self.poses["view"]["joints"]
        self.midpoint_joints = self.poses["midpoint"]["joints"]
        self.drop_joints = self.poses["drop"]["joints"]

        calibration = read_json(self.paths.calibration)
        saved_view = calibration.get("home_view_joints")
        if saved_view is not None and not np.allclose(
            saved_view, self.view_joints, atol=1e-3, rtol=0.0
        ):
            raise StationNotReady(
                f"{robot}/{conveyor}: the view pose changed after calibration. "
                "Re-run 04_workspace_calibration.py."
            )

        self.H = np.load(self.paths.homography)
        if self.H.shape != (3, 3) or not np.all(np.isfinite(self.H)):
            raise StationNotReady(f"Invalid homography in {self.paths.homography}")

        self.pick_ref = read_json(self.paths.pick_reference)
        if float(self.pick_ref["hover_z"]) <= float(self.pick_ref["contact_z"]):
            raise StationNotReady("Hover Z must be above contact Z.")

        offset = self.pick_ref.get("pick_offset_m", [0.0, 0.0])
        self.offset_x, self.offset_y = float(offset[0]), float(offset[1])


        self.polygon = load_polygon(self.paths) if need_roi else None

    # ---------------------------------------------------------- mapping

    def pixel_to_robot(self, px: float, py: float) -> tuple[float, float]:
        point = np.array([[[float(px), float(py)]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.H)[0][0]
        x = float(mapped[0]) + self.offset_x
        y = float(mapped[1]) + self.offset_y
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError("Pixel mapping returned invalid values.")
        return x, y

    @staticmethod
    def is_safe(x: float, y: float, runtime: dict) -> bool:
        """Same rule as core/picker.py: target within the arm's reach."""
        reach = float(np.hypot(x, y))
        return runtime["min_reach_m"] <= reach <= runtime["max_reach_m"]

    def inside_roi(self, x: float, y: float) -> bool:
        if self.polygon is None:
            return True
        return cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0

    def build_poses(self, x: float, y: float) -> tuple[PoseObject, PoseObject]:
        ref = self.pick_ref
        orientation = (float(ref["roll"]), float(ref["pitch"]), float(ref["yaw"]))
        hover = PoseObject(x, y, float(ref["hover_z"]), *orientation)
        contact = PoseObject(x, y, float(ref["contact_z"]), *orientation)
        return hover, contact

    # ---------------------------------------------------------- motion

    def go_view(self, niryo) -> None:
        niryo.set_learning_mode(False)
        niryo.move(JointsPosition(*self.view_joints))

    def pick_and_place(self, niryo, x: float, y: float, log=print) -> bool:
        hover, contact = self.build_poses(x, y)
        vacuum = False
        try:
            log(f"{self.robot}/{self.conveyor}: hover X={x:.4f} Y={y:.4f}")
            niryo.move(hover)
            log("descending to contact")
            niryo.move(contact)
            niryo.grasp_with_tool()
            vacuum = True
            time.sleep(0.75)
            niryo.move(hover)
            log("midpoint -> drop")
            niryo.move(JointsPosition(*self.midpoint_joints))
            niryo.move(JointsPosition(*self.drop_joints))
            niryo.release_with_tool()
            vacuum = False
            time.sleep(0.5)
            niryo.move(JointsPosition(*self.midpoint_joints))
            niryo.move(JointsPosition(*self.view_joints))
            log("pick-and-place complete, back at view pose")
            return True
        except Exception as error:
            log(f"PICK ERROR: {error}")
            return False
        finally:
            if vacuum:
                try:
                    niryo.release_with_tool()
                except Exception:
                    pass


def draw_station_roi(frame, polygon, label: str, color=(255, 255, 0)) -> None:
    if polygon is None:
        return
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
    cv2.polylines(frame, [polygon], True, color, 2)
    x, y = polygon[0][0]
    cv2.putText(frame, label, (int(x), max(20, int(y) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
