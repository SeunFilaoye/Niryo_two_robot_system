"""
01b_identify_conveyors.py

Each robot drives two conveyors. pyniryo only knows them as ConveyorID.ID_1 /
ID_2, so this script runs each one briefly and asks which physical conveyor
(and AprilTag) moved. The answer is saved in robots/<robot>/robot.json.

    python 01b_identify_conveyors.py --robot 150
"""

import time

from common.cli import confirm, parse_robot, robot_parser
from common.config import load_robot_file, load_system, now_iso, robot_file, write_json
from common.niryo_utils import connect, disconnect, run_conveyor

TEST_SPEED = 40
TEST_SECONDS = 2.5


def main():
    args = parse_robot(robot_parser(__doc__))
    system = load_system()
    owned = system["robots"][args.robot]["conveyors"]

    print(f"{args.robot} should drive {owned}.")
    if not confirm("Each conveyor will run for a few seconds."):
        return

    niryo = None
    mapping = {}
    try:
        niryo = connect(args.robot)
        ids = niryo.get_connected_conveyors_id()
        print(f"Connected conveyors: {ids}")
        if not ids:
            print("No conveyor reported. Check cables / Niryo Studio conveyor setup.")
            return

        for conveyor_id in ids:
            print(f"\nRunning {conveyor_id.name} for {TEST_SECONDS}s...")
            run_conveyor(niryo, conveyor_id, TEST_SPEED)
            time.sleep(TEST_SECONDS)
            niryo.stop_conveyor(conveyor_id)

            options = ", ".join(
                f"{name} (tag {system['conveyors'][name]['tag_id']})" for name in owned
            )
            while True:
                answer = input(f"Which conveyor moved? [{options}] or 'skip': ").strip().upper()
                if answer == "SKIP":
                    break
                if not answer.startswith("C"):
                    answer = f"C{answer}"
                if answer in owned and answer not in mapping:
                    mapping[answer] = conveyor_id.name
                    break
                print("Not a valid / unused choice.")

        data = load_robot_file(args.robot)
        data.update({
            "robot": args.robot,
            "ip": system["robots"][args.robot]["ip"],
            "conveyor_hardware": mapping,
            "conveyors_identified_at": now_iso(),
        })
        write_json(robot_file(args.robot), data)
        print(f"\nSaved {robot_file(args.robot)}")
        print(f"Mapping: {mapping}")
        missing = [name for name in owned if name not in mapping]
        if missing:
            print(f"WARNING: not identified yet: {missing}")

    except Exception as error:
        print(f"Conveyor identification failed: {error}")
    finally:
        if niryo is not None:
            for conveyor_id in niryo.get_connected_conveyors_id() or []:
                try:
                    niryo.stop_conveyor(conveyor_id)
                except Exception:
                    pass
        disconnect(niryo, args.robot)


if __name__ == "__main__":
    main()
