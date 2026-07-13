import cv2
import numpy as np
from pyniryo import NiryoRobot

ROBOT_IP = "192.168.0.199"

robot = NiryoRobot(ROBOT_IP)

print("Connected. Press Q to quit.")

try:
    while True:
        img_compressed = robot.get_img_compressed()

        np_arr = np.frombuffer(img_compressed, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print("Could not decode frame.")
            continue

        cv2.imshow("Niryo Wrist Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cv2.destroyAllWindows()
    robot.close_connection()