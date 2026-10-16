from pyniryo import NiryoRobot


# ============================================================
# ROBOT CONNECTION SETTINGS
# ============================================================

from common.settings import get as get_settings  # --robot / --conveyor

S = get_settings(need_conveyor=False)

ROBOT_NAME = S.ROBOT_NAME
ROBOT_IP = S.ROBOT_IP


# ============================================================
# MAIN
# ============================================================

def main():
    robot = None

    print("=" * 60)
    print(f"{ROBOT_NAME} CONNECTION TEST")
    print(f"IP address: {ROBOT_IP}")
    print("=" * 60)

    try:
        print(f"Connecting to {ROBOT_NAME}...")

        robot = NiryoRobot(ROBOT_IP)

        print(f"Connected successfully to {ROBOT_NAME}.")
        print("No movement command was sent.")

        # Do not uncomment until the robot's home pose is confirmed safe.
        # HOME_JOINTS = (...)
        # robot.move_joints(HOME_JOINTS)

    except Exception as error:
        print(f"Could not connect to {ROBOT_NAME}.")
        print(f"Error: {error}")

    finally:
        if robot is not None:
            try:
                robot.close_connection()
                print(f"Disconnected from {ROBOT_NAME}.")
            except Exception as error:
                print(f"Error while disconnecting: {error}")


if __name__ == "__main__":
    main()