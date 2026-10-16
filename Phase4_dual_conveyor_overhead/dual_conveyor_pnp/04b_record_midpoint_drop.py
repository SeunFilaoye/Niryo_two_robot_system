"""
04b_record_midpoint_drop.py

Record this conveyor's MIDPOINT and DROP joint positions.

On the original robots these were typed into MIDPOINT_JOINTS / DROP_JOINTS
at the top of 06, 07, 09 and core/picker.py. Here they are recorded once per
conveyor (joints + TCP) into robots/<robot>/<conveyor>/poses.json, and every
script reads them from there.

This script:

1. Moves the robot to this conveyor's home/view joint position.
2. You move the arm to the SAFE MIDPOINT, press ENTER in the terminal.
3. You move the arm to the DROP-OFF position, press ENTER in the terminal.

    python 04b_record_midpoint_drop.py --robot 150 --conveyor C1
    python 04b_record_midpoint_drop.py --robot 150 --conveyor C1 --only drop
    python 04b_record_midpoint_drop.py --robot 150 --park
"""

import argparse
import json

from pyniryo import NiryoRobot, JointsPosition

from common.config import (
    StationPaths, check_robot_owns_conveyor, conveyor_key, load_robot_file,
    load_system, now_iso, read_json, robot_file, robot_key, write_json,
)


POSE_TEXT = {
    "midpoint": "the SAFE MIDPOINT between the conveyor and the drop-off",
    "drop": "the DROP-OFF position (where the chip is released)",
}


def get_current_pose(robot):
    pose = robot.get_pose()
    return {
        "joints": [round(float(value), 4) for value in robot.get_joints()],
        "tcp": {
            "x": float(pose.x),
            "y": float(pose.y),
            "z": float(pose.z),
            "roll": float(pose.roll),
            "pitch": float(pose.pitch),
            "yaw": float(pose.yaw),
        },
        "recorded_at": now_iso(),
    }


def record(robot, text):
    input(f"\nMove the robot to {text}, then press ENTER...")
    pose = get_current_pose(robot)
    print(f"Joints: {pose['joints']}")
    print(f"TCP:    {json.dumps(pose['tcp'])}")
    return pose


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--conveyor")
    parser.add_argument("--only", choices=sorted(POSE_TEXT))
    parser.add_argument("--park", action="store_true",
                        help="record where the robot waits between supervisor checks")
    args = parser.parse_args()
    robot_name = robot_key(args.robot)
    robot_ip = load_system()["robots"][robot_name]["ip"]

    if not args.park:
        if not args.conveyor:
            parser.error("--conveyor is required unless --park is used")
        conveyor = conveyor_key(args.conveyor)
        check_robot_owns_conveyor(robot_name, conveyor)
        paths = StationPaths(robot_name, conveyor)
        data = read_json(paths.poses)
        home_view = data["poses"]["view"]["joints"]

    robot = None
    try:
        print(f"Connecting to {robot_name} ({robot_ip})...")
        robot = NiryoRobot(robot_ip)
        print(f"Connected to {robot_name}.")

        if args.park:
            pose = record(robot, "the PARK position (where it waits between checks)")
            info = load_robot_file(robot_name)
            info["park"] = pose
            write_json(robot_file(robot_name), info)
            print(f"\nSaved PARK to {robot_file(robot_name)}")
            return

        print(f"Moving {robot_name} to {conveyor} home/view position...")
        robot.move(JointsPosition(*home_view))
        print("Reached home/view position.")

        names = [args.only] if args.only else ["midpoint", "drop"]
        for name in names:
            data["poses"][name] = record(robot, POSE_TEXT[name])
            data["updated_at"] = now_iso()
            write_json(paths.poses, data)
            print(f"Saved {name.upper()} to {paths.poses}")

        print("")
        print("=" * 70)
        print(f"{robot_name} {conveyor} MIDPOINT/DROP RECORDED")
        print("Next: 06_manual_pick_test.py")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\nRecording interrupted. Poses saved so far are kept.")

    except Exception as error:
        print(f"Recording failed: {error}")

    finally:
        if robot is not None:
            try:
                robot.close_connection()
                print(f"Disconnected from {robot_name}.")
            except Exception as error:
                print(f"Error while disconnecting: {error}")


if __name__ == "__main__":
    main()
