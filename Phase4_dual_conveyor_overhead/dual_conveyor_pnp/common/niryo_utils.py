"""Small pyniryo helpers shared by every script."""

from __future__ import annotations

import time

import cv2
import numpy as np
from pyniryo import ConveyorDirection, ConveyorID, JointsPosition, NiryoRobot

from common.config import load_robot_file, robot_info


def connect(robot: str) -> NiryoRobot:
    ip = robot_info(robot)["ip"]
    print(f"Connecting to {robot} ({ip})...")
    niryo = NiryoRobot(ip)
    print(f"Connected to {robot}.")
    return niryo


def disconnect(niryo: NiryoRobot | None, robot: str) -> None:
    if niryo is None:
        return
    try:
        niryo.close_connection()
        print(f"Disconnected from {robot}.")
    except Exception as error:
        print(f"Disconnect warning: {error}")


def get_frame(niryo: NiryoRobot):
    compressed = niryo.get_img_compressed()
    if compressed is None:
        return None
    return cv2.imdecode(np.frombuffer(compressed, dtype=np.uint8), cv2.IMREAD_COLOR)


def move_joints(niryo: NiryoRobot, joints, label: str = "") -> None:
    if label:
        print(f"ROBOT: moving to {label}...")
    niryo.set_learning_mode(False)
    niryo.move(JointsPosition(*[float(value) for value in joints]))


# ============================================================
# CONVEYORS
# ============================================================

def conveyor_id_from_name(name: str) -> ConveyorID:
    return ConveyorID[name]


def conveyor_hardware_map(robot: str) -> dict[str, ConveyorID]:
    """Map 'C1' -> ConveyorID from robot.json (written by 01b_identify_conveyors)."""
    data = load_robot_file(robot).get("conveyor_hardware", {})
    return {conveyor: conveyor_id_from_name(name) for conveyor, name in data.items()}


def run_conveyor(niryo: NiryoRobot, conveyor_id, speed: int) -> None:
    niryo.run_conveyor(conveyor_id, speed=int(speed), direction=ConveyorDirection.FORWARD)


def stop_all_conveyors(niryo: NiryoRobot) -> None:
    try:
        ids = niryo.get_connected_conveyors_id()
    except Exception as error:
        print(f"Could not query conveyors: {error}")
        return
    for conveyor_id in ids or []:
        try:
            niryo.stop_conveyor(conveyor_id)
            print(f"Stopped conveyor {conveyor_id}.")
        except Exception as error:
            print(f"Could not stop {conveyor_id}: {error}")
