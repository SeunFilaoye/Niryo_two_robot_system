"""
06_manual_pick_test.py

Manual vacuum pick-and-place test.

Purpose:
- Uses saved values from calibration.
- Does NOT use YOLO.
- Does NOT use camera.
- Tests physical robot motion, vacuum pickup, safe midpoint, and dropoff.

Sequence:
HOME
→ PICK_HOVER_POSE
→ PICK_CONTACT_POSE
→ vacuum ON
→ PICK_HOVER_POSE
→ MIDPOINT_JOINTS
→ DROP_JOINTS
→ vacuum OFF
→ MIDPOINT_JOINTS
→ HOME

IMPORTANT:
Place a chip at the physical location where the contact pose was recorded.
Keep your hand near emergency stop.
"""

from pyniryo import NiryoRobot
import time


# ============================================================
# ROBOT CONNECTION
# ============================================================

ROBOT_IP = "192.168.0.199"


# ============================================================
# JOINT POSITIONS
# Format: j1, j2, j3, j4, j5, j6
# ============================================================

HOME_JOINTS = (
    -0.060,
    0.293,
    -0.203,
    0.033,
    -1.886,
    -1.751
)

MIDPOINT_JOINTS = (
    1.3005,
    -0.0217,
    -0.3825,
    -0.052,
    -1.2104,
    -1.744
)

DROP_JOINTS = (
    2.286,
    -0.426,
    -0.559,
    0.099,
    -0.489,
    -1.655
)


# ============================================================
# PICKUP POSES
# Format: x, y, z, roll, pitch, yaw
# These came from your 04_workspace_calibration.py output.
# ============================================================

PICK_HOVER_POSE = (
    0.18606280074238177,
    -0.02289445834263062,
    0.11044280042333063,
    1.6558313801797941,
    1.334049821506203,
    -3.0089002695092293
)

PICK_CONTACT_POSE = (
    0.19914561513071916,
    -0.020204871136436955,
    0.0752505289998351,
    1.3044557040133955,
    1.5430313697396774,
    2.842307988731357
)


# ============================================================
# MAIN TEST
# ============================================================

def main():
    print("Manual vacuum pick-and-place test")
    print("=" * 60)
    print("Make sure:")
    print("1. Vacuum pump is attached.")
    print("2. Chip is placed at the contact point.")
    print("3. Drop area is clear.")
    print("4. Emergency stop is reachable.")
    print("=" * 60)

    confirm = input("Type YES to start robot motion: ")

    if confirm != "YES":
        print("Cancelled.")
        return

    robot = NiryoRobot(ROBOT_IP)
    print("Connected to robot.")

    try:
        robot.update_tool()

        print("Moving to HOME...")
        robot.move_joints(HOME_JOINTS)

        print("Moving to PICK HOVER...")
        robot.move_pose(*PICK_HOVER_POSE)

        print("Descending to PICK CONTACT...")
        robot.move_pose(*PICK_CONTACT_POSE)

        print("Activating vacuum pump...")
        robot.grasp_with_tool()
        time.sleep(1.0)

        print("Lifting back to PICK HOVER...")
        robot.move_pose(*PICK_HOVER_POSE)

        print("Moving through SAFE MIDPOINT...")
        robot.move_joints(MIDPOINT_JOINTS)

        print("Moving to DROP...")
        robot.move_joints(DROP_JOINTS)

        print("Releasing vacuum pump...")
        robot.release_with_tool()
        time.sleep(1.0)

        print("Returning through SAFE MIDPOINT...")
        robot.move_joints(MIDPOINT_JOINTS)

        print("Returning HOME...")
        robot.move_joints(HOME_JOINTS)

        print("=" * 60)
        print("Manual pick-and-place test complete.")
        print("=" * 60)

    finally:
        robot.close_connection()
        print("Disconnected from robot.")


if __name__ == "__main__":
    main()