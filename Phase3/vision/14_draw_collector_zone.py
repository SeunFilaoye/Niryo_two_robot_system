import cv2
import json
import numpy as np
from pathlib import Path
from pyniryo import NiryoRobot

ROBOT_IP = "192.168.0.199"

# Project root (one folder above /vision)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files
ROI_FILE = PROJECT_ROOT / "polygon_roi.json"
WORKSPACE_FILE = PROJECT_ROOT / "workspace_config.json"

points = []


def load_detection_polygon():
    if not ROI_FILE.exists():
        raise FileNotFoundError(
            "Run 13_set_polygon_roi.py first."
        )

    with open(ROI_FILE, "r") as f:
        data = json.load(f)

    return [(p["x"], p["y"]) for p in data["polygon_roi"]]


def save_workspace(detection_polygon, collector_polygon):

    workspace = {
        "detection_polygon": [
            {"x": int(x), "y": int(y)}
            for x, y in detection_polygon
        ],

        "collector_polygon": [
            {"x": int(x), "y": int(y)}
            for x, y in collector_polygon
        ]
    }

    with open(WORKSPACE_FILE, "w") as f:
        json.dump(workspace, f, indent=4)

    print("\nSaved workspace_config.json")


def mouse(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)} : ({x},{y})")


def draw(frame, detection_polygon):

    img = frame.copy()

    det = np.array(detection_polygon, np.int32)

    cv2.polylines(
        img,
        [det],
        True,
        (255,255,0),
        2
    )

    if len(points):

        col = np.array(points, np.int32)

        cv2.polylines(
            img,
            [col],
            False,
            (0,255,255),
            2
        )

        if len(points) >= 3:
            cv2.polylines(
                img,
                [col],
                True,
                (0,255,255),
                3
            )

    for i,p in enumerate(points):
        cv2.circle(img,p,5,(0,255,255),-1)

        cv2.putText(
            img,
            str(i+1),
            (p[0]+5,p[1]-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            2
        )

    cv2.putText(
        img,
        "Draw COLLECTOR polygon | S=Save C=Clear Q=Quit",
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    return img


def main():

    detection_polygon = load_detection_polygon()

    robot = NiryoRobot(ROBOT_IP)

    cv2.namedWindow("Collector Zone")

    cv2.setMouseCallback(
        "Collector Zone",
        mouse
    )

    try:

        while True:

            img = robot.get_img_compressed()

            frame = cv2.imdecode(
                np.frombuffer(img,np.uint8),
                cv2.IMREAD_COLOR
            )

            frame = draw(
                frame,
                detection_polygon
            )

            cv2.imshow(
                "Collector Zone",
                frame
            )

            key = cv2.waitKey(1)&0xFF

            if key==ord("c"):
                points.clear()

            elif key==ord("s"):

                if len(points)<3:
                    print("Need at least 3 points.")
                    continue

                save_workspace(
                    detection_polygon,
                    points
                )

            elif key==ord("q"):
                break

    finally:

        robot.close_connection()

        cv2.destroyAllWindows()


if __name__=="__main__":
    main()