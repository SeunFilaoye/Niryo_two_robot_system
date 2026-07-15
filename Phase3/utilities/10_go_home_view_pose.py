"""
10_go_home_view_pose.py

Moves the robot to the calibrated Phase 2 home/viewing pose.

Use this before:
- setting ROI
- setting priority point
- running Phase 2
- checking camera alignment

This does not run YOLO or conveyor.
"""

from pyniryo import NiryoRobot


ROBOT_IP = "192.168.0.199"

HOME_VIEW_JOINTS = (
    0.3112,
    0.61,
    -0.2931,
    -0.0459,
    -1.8991,
    -1.5737
)


def main():
    print("Connecting to robot...")
    robot = NiryoRobot(ROBOT_IP)

    try:
        print("Connected.")
        print("Moving to calibrated Phase 2 home/viewing pose...")
        robot.move_joints(HOME_VIEW_JOINTS)
        print("Robot is now in calibrated viewing pose.")

    finally:
        robot.close_connection()
        print("Disconnected.")


if __name__ == "__main__":
    main()