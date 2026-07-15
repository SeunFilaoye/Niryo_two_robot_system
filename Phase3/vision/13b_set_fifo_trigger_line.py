import cv2
import json
import numpy as np
from pathlib import Path
from pyniryo import NiryoRobot

ROBOT_IP = "192.168.0.199"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_FILE = PROJECT_ROOT / "workspace_config.json"

points = []


def load_workspace():
    if not WORKSPACE_FILE.exists():
        raise FileNotFoundError("workspace_config.json not found. Run 14 first.")

    with open(WORKSPACE_FILE, "r") as f:
        return json.load(f)


def save_workspace(data):
    data["fifo_trigger_gate"] = [
        {"x": int(x), "y": int(y)}
        for x, y in points
    ]

    # Remove old 2-point line if it exists
    if "fifo_trigger_line" in data:
        del data["fifo_trigger_line"]

    with open(WORKSPACE_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print("Saved 3-point FIFO trigger gate to workspace_config.json")


def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 3:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
        else:
            print("Already have 3 points. Press C to clear.")


def draw(frame, data):
    img = frame.copy()

    if "detection_polygon" in data:
        det = np.array(
            [(p["x"], p["y"]) for p in data["detection_polygon"]],
            np.int32
        )
        cv2.polylines(img, [det], True, (255, 255, 0), 2)

    if "collector_polygon" in data:
        col = np.array(
            [(p["x"], p["y"]) for p in data["collector_polygon"]],
            np.int32
        )
        cv2.polylines(img, [col], True, (0, 255, 255), 3)

    for i, p in enumerate(points):
        cv2.circle(img, p, 6, (255, 0, 255), -1)
        cv2.putText(
            img,
            str(i + 1),
            (p[0] + 6, p[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    if len(points) >= 2:
        cv2.polylines(
            img,
            [np.array(points, np.int32)],
            False,
            (255, 0, 255),
            2,
        )

    if len(points) == 3:
        cv2.polylines(
            img,
            [np.array(points, np.int32)],
            True,
            (255, 0, 255),
            3,
        )

    cv2.putText(
        img,
        "Click 3 points for FIFO trigger gate | S=Save C=Clear Q=Quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    return img


def main():
    data = load_workspace()

    robot = NiryoRobot(ROBOT_IP)

    cv2.namedWindow("Set FIFO Trigger Gate")
    cv2.setMouseCallback("Set FIFO Trigger Gate", mouse)

    try:
        while True:
            img = robot.get_img_compressed()
            frame = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

            cv2.imshow("Set FIFO Trigger Gate", draw(frame, data))

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                points.clear()
                print("Cleared points.")

            elif key == ord("s"):
                if len(points) != 3:
                    print("Need exactly 3 points.")
                    continue

                save_workspace(data)

            elif key == ord("q"):
                break

    finally:
        robot.close_connection()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()