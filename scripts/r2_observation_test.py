import cv2
import numpy as np
from pyniryo import NiryoRobot
from ultralytics import YOLO
from pathlib import Path

ROBOT_IP = "192.168.0.201"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"

# REPLACE THIS with your new robot 2 observation pose
OBSERVATION_JOINTS = (-2.9121, 0.7115, -0.687, -0.0121, -1.6445, -0.1057)

ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 180, 100, 500, 380

CLASS_COLORS = {
    "red_square": (203, 192, 255),
    "red_circle": (0, 0, 139),
    "blue_square": (255, 191, 0),
    "blue_circle": (211, 0, 148),
    "green_square": (144, 238, 144),
    "green_circle": (0, 100, 0),
}

def decode_robot_image(img_compressed):
    np_arr = np.frombuffer(img_compressed, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def main():
    robot = None
    model = YOLO(str(MODEL_PATH))

    try:
        print(f"Connecting to robot at {ROBOT_IP} ...")
        robot = NiryoRobot(ROBOT_IP)
        robot.clear_collision_detected()
        print("Connected.")

        print("Moving to observation pose...")
        robot.move_joints(*OBSERVATION_JOINTS)

        while True:
            img_compressed = robot.get_img_compressed()
            if img_compressed is None:
                continue

            frame = decode_robot_image(img_compressed)
            if frame is None:
                continue

            annotated = frame.copy()
            results = model(frame, conf=0.35, verbose=False)[0]

            cv2.rectangle(annotated, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 255, 255), 2)
            cv2.putText(annotated, "ROI", (ROI_X1, max(ROI_Y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                class_name = model.names[cls_id]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                color = CLASS_COLORS.get(class_name, (255, 255, 255))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.circle(annotated, (cx, cy), 4, color, -1)
                cv2.putText(annotated, class_name, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            cv2.imshow("Robot 2 Observation Test", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        cv2.destroyAllWindows()
        if robot is not None:
            try:
                robot.close_connection()
            except Exception:
                pass
        print("Connection closed.")

if __name__ == "__main__":
    main()