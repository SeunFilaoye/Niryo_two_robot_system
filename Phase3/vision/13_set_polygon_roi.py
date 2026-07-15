import cv2, json, numpy as np
from pathlib import Path
from pyniryo import NiryoRobot

ROBOT_IP = "192.168.0.199"
BASE_DIR = Path(__file__).resolve().parent.parent
POLYGON_FILE = BASE_DIR / "polygon_roi.json"

points = []
mouse = (0, 0)

def save_polygon():
    data = {"polygon_roi": [{"x": int(x), "y": int(y)} for x, y in points]}
    with open(POLYGON_FILE, "w") as f:
        json.dump(data, f, indent=4)
    print("Saved polygon ROI:", POLYGON_FILE)

def mouse_callback(event, x, y, flags, param):
    global mouse
    mouse = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

def draw(frame):
    out = frame.copy()
    for i, p in enumerate(points):
        cv2.circle(out, p, 5, (255, 255, 0), -1)
        cv2.putText(out, str(i + 1), (p[0] + 5, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    if len(points) > 1:
        cv2.polylines(out, [np.array(points, np.int32)], False, (255,255,0), 2)
    if len(points) > 2:
        cv2.polylines(out, [np.array(points, np.int32)], True, (255,255,0), 3)
    cv2.putText(out, "Click polygon points | S=save | C=clear | Q=quit",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    return out

def main():
    robot = NiryoRobot(ROBOT_IP)
    cv2.namedWindow("Draw Polygon ROI")
    cv2.setMouseCallback("Draw Polygon ROI", mouse_callback)

    try:
        while True:
            img = robot.get_img_compressed()
            frame = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow("Draw Polygon ROI", draw(frame))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if len(points) >= 3:
                    save_polygon()
                else:
                    print("Need at least 3 points.")
            elif key == ord("c"):
                points.clear()
                print("Cleared polygon.")
            elif key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        robot.close_connection()

if __name__ == "__main__":
    main()