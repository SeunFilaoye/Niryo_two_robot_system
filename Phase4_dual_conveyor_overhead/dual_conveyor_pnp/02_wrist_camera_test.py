import cv2
import numpy as np
from pyniryo import NiryoRobot


# ============================================================
# ROBOT SETTINGS
# ============================================================

from common.settings import get as get_settings  # --robot / --conveyor

S = get_settings(need_conveyor=False)

ROBOT_NAME = S.ROBOT_NAME
ROBOT_IP = S.ROBOT_IP

WINDOW_NAME = f"{ROBOT_NAME} - Niryo Wrist Camera"


# ============================================================
# MAIN
# ============================================================

def main():
    robot = None

    print("=" * 60)
    print(f"{ROBOT_NAME} WRIST CAMERA TEST")
    print(f"IP address: {ROBOT_IP}")
    print("=" * 60)

    try:
        print(f"Connecting to {ROBOT_NAME}...")
        robot = NiryoRobot(ROBOT_IP)

        print(f"Connected successfully to {ROBOT_NAME}.")
        print("Camera window opening.")
        print("Press Q in the camera window to quit.")

        while True:
            img_compressed = robot.get_img_compressed()

            if img_compressed is None:
                print("No compressed image received.")
                continue

            np_arr = np.frombuffer(img_compressed, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Could not decode camera frame.")
                continue

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Q pressed. Closing camera test.")
                break

    except KeyboardInterrupt:
        print("\nCamera test stopped from the terminal.")

    except Exception as error:
        print(f"Camera test failed for {ROBOT_NAME}.")
        print(f"Error: {error}")

    finally:
        cv2.destroyAllWindows()

        if robot is not None:
            try:
                robot.close_connection()
                print(f"Disconnected from {ROBOT_NAME}.")
            except Exception as error:
                print(f"Error while disconnecting: {error}")


if __name__ == "__main__":
    main()