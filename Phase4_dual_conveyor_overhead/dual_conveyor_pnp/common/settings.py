"""Per-run robot/station settings for the original training scripts.

The original scripts (copied from dynamic_pnp/Robot 1) kept ROBOT_NAME,
ROBOT_IP, HOME_VIEW_JOINTS, MIDPOINT_JOINTS, DROP_JOINTS and ROBOT_DIR as
constants at the top of every file. Here those same constants come from the
command line instead:

    python 04_workspace_calibration.py --robot 150 --conveyor C1

  ROBOT_NAME        "robot_150"
  ROBOT_IP          from config/system.json
  HOME_VIEW_JOINTS  this conveyor's VIEW pose   (robots/<robot>/<C>/poses.json)
  MIDPOINT_JOINTS   this conveyor's MIDPOINT    (04b_record_midpoint_drop.py)
  DROP_JOINTS       this conveyor's DROP        (04b_record_midpoint_drop.py)
  ROBOT_DIR         robots/<robot>/<C>/  -> calibration/, phase2_config.json, logs/
  PROJECT_DIR       dynamic_pnp/         -> models/best.pt (same model path as before)
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

from common.config import (
    SYSTEM_DIR, check_robot_owns_conveyor, conveyor_key, load_robot_file,
    load_system, read_json, robot_dir, robot_key, station_dir,
)

_cache: dict = {}


def _pose_joints(poses: dict, name: str):
    pose = poses.get(name)
    return tuple(pose["joints"]) if pose else None


def get(need_conveyor: bool = True) -> SimpleNamespace:
    key = need_conveyor
    if key in _cache:
        return _cache[key]

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--robot")
    parser.add_argument("--conveyor")
    args, _ = parser.parse_known_args()
    script = sys.argv[0].rsplit("/", 1)[-1]
    usage = f"python {script} --robot 150" + (" --conveyor C1" if need_conveyor else "")
    if not args.robot or (need_conveyor and not args.conveyor):
        raise SystemExit(f"Usage: {usage}")

    robot = robot_key(args.robot)
    system = load_system()
    settings = SimpleNamespace(
        ROBOT_KEY=robot,
        ROBOT_NAME=robot,
        ROBOT_IP=system["robots"][robot]["ip"],
        CONVEYOR=None,
        CONVEYOR_HARDWARE=None,
        HOME_VIEW_JOINTS=None,
        MIDPOINT_JOINTS=None,
        DROP_JOINTS=None,
        ROBOT_DIR=robot_dir(robot),
        PROJECT_DIR=SYSTEM_DIR.parent,
        SYSTEM=system,
    )

    if args.conveyor:
        conveyor = conveyor_key(args.conveyor)
        check_robot_owns_conveyor(robot, conveyor)
        directory = station_dir(robot, conveyor)
        poses_file = directory / "poses.json"
        poses = read_json(poses_file).get("poses", {}) if poses_file.exists() else {}
        settings.CONVEYOR = conveyor
        settings.ROBOT_NAME = f"{robot} {conveyor}"
        settings.ROBOT_DIR = directory
        settings.HOME_VIEW_JOINTS = _pose_joints(poses, "view")
        settings.MIDPOINT_JOINTS = _pose_joints(poses, "midpoint")
        settings.DROP_JOINTS = _pose_joints(poses, "drop")
        settings.CONVEYOR_HARDWARE = load_robot_file(robot).get(
            "conveyor_hardware", {}
        ).get(conveyor)

        if settings.HOME_VIEW_JOINTS is None:
            raise SystemExit(
                f"No VIEW pose saved for {robot} {conveyor} in {poses_file}."
            )

    _cache[key] = settings
    return settings


def require_midpoint_drop(settings) -> None:
    if settings.MIDPOINT_JOINTS is None or settings.DROP_JOINTS is None:
        raise SystemExit(
            f"{settings.ROBOT_NAME}: MIDPOINT/DROP not recorded yet. Run:\n"
            f"  python 04b_record_midpoint_drop.py --robot {settings.ROBOT_KEY.split('_')[1]} "
            f"--conveyor {settings.CONVEYOR}"
        )
