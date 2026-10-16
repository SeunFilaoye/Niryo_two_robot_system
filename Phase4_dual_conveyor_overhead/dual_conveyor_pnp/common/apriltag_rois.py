"""AprilTag-anchored conveyor ROIs for the overhead cameras.

Adapted from robo_confirm/test_all_apriltag_rois.py and
robo_confirm/overhead/apriltag_priority_supervisor.py, restricted to the tags
this system uses (0 and 1) and extended with a third, per-conveyor
ALERT ROI.

Every ROI is stored in AprilTag-local inches. At runtime each tag's corners
are averaged once (session lock), and a tag-local -> pixel homography places
the ROIs on that camera's image.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from common.config import OVERHEAD_ROI_FILE, read_json, write_json

TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11

COLOR_CONVEYOR = (0, 255, 0)
COLOR_ENTRY = (255, 0, 255)
COLOR_ALERT = (0, 0, 255)
COLOR_PREVIEW = (0, 220, 255)


def make_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_tags(frame, detector, valid_ids) -> dict[int, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {
        int(tag): corners[i].reshape(4, 2).astype(np.float32)
        for i, tag in enumerate(ids.flatten())
        if int(tag) in valid_ids
    }


def tag_local_corners(tag_size_in: float) -> np.ndarray:
    s = float(tag_size_in)
    return np.asarray([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)


MAPPING_MODES = ("homography", "similarity", "affine")


def tag_transform(src, dst, mode: str = "homography"):
    """3x3 matrix mapping src points onto dst points.

    homography  full perspective (8 DOF). Exact on the tag, but tag-corner
                noise or tag tilt adds a perspective term that grows with
                distance, so far-away ROIs widen.
    similarity  rotation + uniform scale + shift (4 DOF). No perspective or
                shear: the ROI keeps its real shape at any distance.
    affine      similarity + shear / non-uniform scale (6 DOF), no perspective.
    """
    src = np.asarray(src, np.float32).reshape(-1, 2)
    dst = np.asarray(dst, np.float32).reshape(-1, 2)
    if mode == "similarity":
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    elif mode == "affine":
        M, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
    else:
        H, _ = cv2.findHomography(src, dst)
        return H
    if M is None:
        return None
    return np.vstack([M, [0.0, 0.0, 1.0]])


def local_to_pixels(local_points, tag_corners, local_corners,
                    mode: str = "homography") -> list[list[int]]:
    if local_points is None or len(local_points) < 3:
        return []
    H = tag_transform(local_corners, tag_corners, mode)
    if H is None:
        return []
    pts = np.asarray(local_points, dtype=np.float32).reshape((-1, 1, 2))
    out = cv2.perspectiveTransform(pts, H).reshape((-1, 2))
    return [[int(round(x)), int(round(y))] for x, y in out]


def pixels_to_local(pixel_points, tag_corners, local_corners,
                    mode: str = "homography") -> list[list[float]]:
    if not pixel_points:
        return []
    H = tag_transform(tag_corners, local_corners, mode)
    if H is None:
        return []
    pts = np.asarray(pixel_points, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.perspectiveTransform(pts, H).reshape((-1, 2)).tolist()


def inside_local(polygon, x: float, y: float) -> bool:
    if polygon is None or len(polygon) < 3:
        return False
    contour = np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


# ============================================================
# CONFIG
# ============================================================

@dataclass
class OverheadROIs:
    tag_size_in: float
    conveyor: list
    entry: list
    alerts: dict  # conveyor name -> polygon or None
    quarter_turns: dict  # conveyor name -> 0..3, how the tag is mounted
    mappings: dict  # conveyor name -> "homography" | "similarity" | "affine"

    def mode(self, conveyor: str) -> str:
        mode = self.mappings.get(conveyor, "homography")
        return mode if mode in MAPPING_MODES else "homography"

    def oriented(self, conveyor: str, corners):
        """Tag corners re-ordered for how this tag is mounted on its conveyor.

        All tags share one ROI template. A tag mounted turned by 90/180/270
        degrees relative to the others would swing that template the wrong
        way; rolling the corner order turns the tag frame back by that many
        quarter turns about the tag center.
        """
        turns = int(self.quarter_turns.get(conveyor, 0)) % 4
        return np.roll(np.asarray(corners, np.float32).reshape(4, 2), -turns, axis=0)


def load_overhead_rois() -> OverheadROIs:
    data = read_json(OVERHEAD_ROI_FILE)
    template = data["shared_roi_template"]
    if template.get("coordinate_frame") != "apriltag_local":
        raise ValueError("shared ROI coordinate_frame must be apriltag_local")
    return OverheadROIs(
        tag_size_in=float(data["tag_size_in"]),
        conveyor=template["polygon_roi_local"],
        entry=template["entry_roi_local"],
        alerts=data.get("alert_rois_local", {}),
        quarter_turns=data.get("tag_quarter_turns", {}),
        mappings=data.get("roi_mapping", {}),
    )


def save_mapping(conveyor: str, mode: str) -> None:
    data = read_json(OVERHEAD_ROI_FILE)
    data.setdefault("roi_mapping", {})[conveyor] = mode
    write_json(OVERHEAD_ROI_FILE, data)


def save_quarter_turns(conveyor: str, turns: int) -> None:
    data = read_json(OVERHEAD_ROI_FILE)
    data.setdefault("tag_quarter_turns", {})[conveyor] = int(turns) % 4
    write_json(OVERHEAD_ROI_FILE, data)


def save_alert_roi(conveyor: str, polygon_local: list[list[float]] | None) -> None:
    data = read_json(OVERHEAD_ROI_FILE)
    data.setdefault("alert_rois_local", {})[conveyor] = polygon_local
    write_json(OVERHEAD_ROI_FILE, data)


# ============================================================
# SESSION LOCK + CAMERA OWNERSHIP
# ============================================================

class SessionTagReference:
    """Average the first N corner observations, then lock for the session."""

    def __init__(self, samples: int = 30):
        self.sample_count = samples
        self.samples = deque(maxlen=samples)
        self.locked = None
        self.preview = None

    def observe(self, corners) -> bool:
        """Return True on the frame the reference becomes locked."""
        if corners is None or self.locked is not None:
            return False
        corners = np.asarray(corners, np.float32).reshape(4, 2)
        if not np.isfinite(corners).all() or not cv2.isContourConvex(corners):
            return False
        if abs(cv2.contourArea(corners)) < 1:
            return False
        self.samples.append(corners.copy())
        self.preview = np.mean(np.stack(self.samples), axis=0).astype(np.float32)
        if len(self.samples) >= self.sample_count:
            self.locked = self.preview.copy()
            self.samples.clear()
            return True
        return False

    @property
    def ready(self) -> bool:
        return self.locked is not None

    def displayed(self):
        return self.locked if self.locked is not None else self.preview


class TagOwnership:
    """The camera with the largest view of a locked tag owns it for the run."""

    def __init__(self):
        self.owners: dict[int, int] = {}

    def update(self, locked_by_camera: dict[int, dict[int, np.ndarray]]) -> None:
        seen: dict[int, list[tuple[float, int]]] = {}
        for camera, tags in locked_by_camera.items():
            for tag, corners in tags.items():
                area = abs(float(cv2.contourArea(corners.reshape(-1, 1, 2))))
                seen.setdefault(tag, []).append((area, -camera))
        for tag, views in seen.items():
            if tag not in self.owners:
                self.owners[tag] = -max(views)[1]

    def owner(self, tag: int) -> int | None:
        return self.owners.get(tag)


class ZoneGate:
    """Fire once when a zone is stably occupied; re-arm after it empties."""

    def __init__(self, stable_frames: int, rearm_empty_frames: int):
        self.stable_required = stable_frames
        self.rearm_required = rearm_empty_frames
        self.armed = True
        self.stable = 0
        self.empty = 0

    def update(self, candidates: list[dict], observable: bool):
        """Returns (fired_detection_or_None, rearmed_bool)."""
        if not observable:
            self.stable = 0
            return None, False
        if not candidates:
            self.stable = 0
            self.empty += 1
            if not self.armed and self.empty >= self.rearm_required:
                self.armed, self.empty = True, 0
                return None, True
            return None, False
        self.empty = 0
        self.stable += 1
        if self.armed and self.stable >= self.stable_required:
            self.armed = False
            return max(candidates, key=lambda d: d["confidence"]), False
        return None, False


# ============================================================
# DRAWING
# ============================================================

def draw_polygon(frame, points, color, fill_alpha=0.06, thickness=2, label=None):
    if len(points) < 3:
        return
    contour = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    if fill_alpha > 0:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [contour], color)
        cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
    cv2.polylines(frame, [contour], True, color, thickness, cv2.LINE_AA)
    if label:
        x, y = points[0]
        cv2.putText(frame, label, (x + 3, max(14, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_tag(frame, corners, tag_id, owner: bool) -> None:
    color = (0, 255, 0) if owner else (110, 110, 110)
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, color, 2 if owner else 1, cv2.LINE_AA)
    x, y = corners[0]
    cv2.putText(frame, f"tag {tag_id}", (int(x), int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


# ============================================================
# CAMERAS
# ============================================================

def open_cameras(indices, width=640, height=480) -> dict[int, cv2.VideoCapture]:
    captures = {}
    for index in indices:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            continue
        captures[index] = cap
    return captures
