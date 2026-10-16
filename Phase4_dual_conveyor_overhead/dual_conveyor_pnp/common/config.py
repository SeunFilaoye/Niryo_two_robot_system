"""Paths and configuration loading for the dual-conveyor system.

Folder layout
-------------
dual_conveyor_pnp/
├── config/system.json          robots, conveyors, tags, ports, runtime tuning
├── config/overhead_rois.json   AprilTag-local conveyor / entry / alert ROIs
└── robots/robot_150/
    ├── robot.json              conveyor hardware IDs, park pose
    └── C1/                     one folder per conveyor station
        ├── poses.json          view / midpoint / drop  (joints + TCP, 04b)
        ├── phase2_config.json  wrist-camera polygon ROI + priority point (08)
        ├── calibration/        homography + hover/contact (TCP)
        └── logs/
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

SYSTEM_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = SYSTEM_DIR / "config"
ROBOTS_DIR = SYSTEM_DIR / "robots"
LOGS_DIR = SYSTEM_DIR / "logs"

SYSTEM_FILE = CONFIG_DIR / "system.json"
OVERHEAD_ROI_FILE = CONFIG_DIR / "overhead_rois.json"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
    tmp.replace(path)


def load_system() -> dict:
    return read_json(SYSTEM_FILE)


def model_path(kind: str, system: dict | None = None) -> Path:
    system = system or load_system()
    return (SYSTEM_DIR / system["models"][kind]).resolve()


def robot_key(value: str | int) -> str:
    """Accept 150, '150', 'robot_150' or '192.168.0.150'."""
    text = str(value).strip().lower()
    text = text.split(".")[-1]
    if not text.startswith("robot_"):
        text = f"robot_{text}"
    system = load_system()
    if text not in system["robots"]:
        raise ValueError(
            f"Unknown robot '{value}'. Known: {sorted(system['robots'])}"
        )
    return text


def conveyor_key(value: str | int) -> str:
    text = str(value).strip().upper()
    if not text.startswith("C"):
        text = f"C{text}"
    system = load_system()
    if text not in system["conveyors"]:
        raise ValueError(
            f"Unknown conveyor '{value}'. Known: {sorted(system['conveyors'])}"
        )
    return text


def alert_conveyors(system: dict | None = None) -> list[str]:
    """Conveyors owned by a robot. Only these get an ALERT ROI and alerts;
    the others (robot null) are watched for position only."""
    system = system or load_system()
    return [name for name, c in system["conveyors"].items() if c.get("robot")]


def robot_info(robot: str, system: dict | None = None) -> dict:
    system = system or load_system()
    return system["robots"][robot]


def check_robot_owns_conveyor(robot: str, conveyor: str) -> None:
    owned = robot_info(robot)["conveyors"]
    if conveyor not in owned:
        raise ValueError(
            f"{conveyor} is not connected to {robot}. "
            f"{robot} handles {owned}."
        )


def robot_dir(robot: str) -> Path:
    return ROBOTS_DIR / robot


def robot_file(robot: str) -> Path:
    return robot_dir(robot) / "robot.json"


def load_robot_file(robot: str) -> dict:
    path = robot_file(robot)
    if path.exists():
        return read_json(path)
    return {"robot": robot, "ip": robot_info(robot)["ip"]}


def station_dir(robot: str, conveyor: str) -> Path:
    return robot_dir(robot) / conveyor


class StationPaths:
    def __init__(self, robot: str, conveyor: str):
        self.robot = robot
        self.conveyor = conveyor
        self.dir = station_dir(robot, conveyor)
        self.poses = self.dir / "poses.json"
        self.roi = self.dir / "phase2_config.json"
        self.cal_dir = self.dir / "calibration"
        self.image_points = self.cal_dir / "image_points.npy"
        self.robot_points = self.cal_dir / "robot_points_xy.npy"
        self.homography = self.cal_dir / "homography_matrix.npy"
        self.pick_reference = self.cal_dir / "pick_reference.json"
        self.calibration = self.cal_dir / "calibration_points.json"
        self.logs = self.dir / "logs"
        self.pick_log = self.logs / "pick_log.csv"
