"""
18_run_fifo_system.py

Phase 3 permanent FIFO system.

Rules:
1. Only detections inside the saved polygon ROI are valid.
2. A new valid chip is appended permanently to FIFO memory.
3. Its dwell timer (3 s, config/system.json) starts once and never resets.
4. Only the first FIFO item can be picked.
5. When its timer finishes and it is visible, the robot picks it.
6. The item remains in memory until pick-and-place succeeds.
7. After successful drop-off and return home, it is removed.
8. New detections are appended after all existing FIFO items.
9. Conveyor remains ON during normal operation.

Press Q to quit.
"""

import math
import time

import cv2
import numpy as np
from pyniryo import (
    ConveyorDirection,
    NiryoRobot,
)

from vision.detector import Detector
from core.picker import Picker


# ============================================================
# SETTINGS
# ============================================================

from common.settings import get as get_settings  # --robot / --conveyor

S = get_settings(need_conveyor=True)

ROBOT_NAME = S.ROBOT_NAME
ROBOT_IP = S.ROBOT_IP

CONFIDENCE_THRESHOLD = 0.75
READY_DELAY_SEC = S.SYSTEM["robot_runtime"]["dwell_sec"]  # 3 s view dwell (config/system.json)

CONVEYOR_SPEED = 50
CONVEYOR_DIRECTION = ConveyorDirection.FORWARD

POST_PICK_DELAY_SEC = 0.75

CAMERA_WINDOW = f"{ROBOT_NAME} - Phase 3 FIFO Camera"
QUEUE_WINDOW = f"{ROBOT_NAME} - FIFO Memory"


# ============================================================
# FIFO MEMORY
# ============================================================

class SimpleFIFO:
    def __init__(self, ready_delay_sec=4.0):
        self.ready_delay_sec = ready_delay_sec
        self.items = []
        self.next_id = 1

    def active_items(self):
        return [
            item
            for item in self.items
            if item["status"] == "waiting"
        ]

    @staticmethod
    def distance(item, detection):
        return math.hypot(
            item["cx"] - detection["cx"],
            item["cy"] - detection["cy"],
        )

    def add_item(self, detection):
        now = time.monotonic()

        item = {
            "id": self.next_id,
            "class_name": detection["class_name"],
            "confidence": detection["confidence"],
            "bbox": detection["bbox"],
            "cx": detection["cx"],
            "cy": detection["cy"],
            "registered_at": now,
            "ready_at": now + self.ready_delay_sec,
            "visible": True,
            "status": "waiting",
        }

        self.items.append(item)

        print(
            f"FIFO ADD: #{item['id']} "
            f"{item['class_name']} | "
            f"ready in {self.ready_delay_sec:.1f}s"
        )

        self.next_id += 1

    @staticmethod
    def update_item(item, detection):
        item["class_name"] = detection["class_name"]
        item["confidence"] = detection["confidence"]
        item["bbox"] = detection["bbox"]
        item["cx"] = detection["cx"]
        item["cy"] = detection["cy"]
        item["visible"] = True

    def update(self, detections):
        active = self.active_items()

        for item in active:
            item["visible"] = False
            # The item remains permanently in FIFO memory.
            # Only its current visibility changes.

        class_names = {
            detection["class_name"]
            for detection in detections
        }

        class_names.update(
            item["class_name"]
            for item in active
        )

        for class_name in class_names:
            class_items = [
                item
                for item in active
                if item["class_name"] == class_name
            ]

            class_detections = [
                detection
                for detection in detections
                if detection["class_name"] == class_name
            ]

            candidate_pairs = []

            for item_index, item in enumerate(class_items):
                for detection_index, detection in enumerate(
                    class_detections
                ):
                    candidate_pairs.append((
                        self.distance(
                            item,
                            detection,
                        ),
                        item_index,
                        detection_index,
                    ))

            candidate_pairs.sort(
                key=lambda pair: pair[0]
            )

            matched_item_indexes = set()
            matched_detection_indexes = set()

            for _, item_index, detection_index in candidate_pairs:
                if item_index in matched_item_indexes:
                    continue

                if detection_index in matched_detection_indexes:
                    continue

                self.update_item(
                    class_items[item_index],
                    class_detections[detection_index],
                )

                matched_item_indexes.add(item_index)
                matched_detection_indexes.add(detection_index)

            for detection_index, detection in enumerate(
                class_detections
            ):
                if detection_index not in matched_detection_indexes:
                    self.add_item(detection)

    def front_item(self):
        active = self.active_items()
        return active[0] if active else None

    def next_ready_item(self):
        front = self.front_item()

        if front is None:
            return None

        timer_finished = (
            time.monotonic() >= front["ready_at"]
        )

        if timer_finished and front["visible"]:
            return front

        return None

    @staticmethod
    def remaining_time(item):
        return max(
            0.0,
            item["ready_at"] - time.monotonic(),
        )

    @staticmethod
    def mark_complete(item):
        item["status"] = "complete"
        item["completed_at"] = time.monotonic()

        print(
            "FIFO COMPLETE/REMOVED: "
            f"#{item['id']} {item['class_name']}"
        )

    @staticmethod
    def keep_after_failure(item):
        item["visible"] = False

        print(
            "FIFO PICK FAILED — ITEM RETAINED: "
            f"#{item['id']} {item['class_name']}"
        )


# ============================================================
# CAMERA
# ============================================================

def get_camera_frame(robot):
    compressed_image = robot.get_img_compressed()

    if compressed_image is None:
        return None

    image_array = np.frombuffer(
        compressed_image,
        dtype=np.uint8,
    )

    return cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )


# ============================================================
# CONVEYOR
# ============================================================

def get_existing_conveyor(robot):
    conveyor_ids = robot.get_connected_conveyors_id()

    if not conveyor_ids:
        raise RuntimeError(
            f"No conveyor is registered with {ROBOT_NAME}."
        )

    # Each robot drives two conveyors; use the one identified for this
    # station by 01b_identify_conveyors.py instead of the first reported.
    if S.CONVEYOR_HARDWARE is None:
        raise RuntimeError(
            f"{S.CONVEYOR} hardware ID unknown. Run 01b_identify_conveyors.py."
        )

    conveyor_id = next(
        cid for cid in conveyor_ids if cid.name == S.CONVEYOR_HARDWARE
    )

    print(f"Using connected conveyor: {conveyor_id}")

    return conveyor_id


def start_conveyor(robot, conveyor_id):
    robot.run_conveyor(
        conveyor_id,
        speed=CONVEYOR_SPEED,
        direction=CONVEYOR_DIRECTION,
    )

    print("Conveyor running continuously.")


def stop_conveyor(robot, conveyor_id):
    if conveyor_id is None:
        return

    try:
        robot.stop_conveyor(conveyor_id)
        print("Conveyor stopped.")

    except Exception as error:
        print(f"Conveyor stop warning: {error}")


# ============================================================
# USER INTERFACE
# ============================================================

def draw_memory_on_camera(
    frame,
    fifo,
    ready_item,
    robot_busy,
):
    front = fifo.front_item()

    for item in fifo.active_items():
        if not item["visible"]:
            continue

        x1, y1, x2, y2 = item["bbox"]

        if ready_item is item and not robot_busy:
            color = (0, 165, 255)
            label = "READY"

        elif front is item:
            color = (0, 255, 255)
            label = "FIFO FRONT"

        else:
            color = (255, 255, 255)
            label = "QUEUED"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3,
        )

        cv2.circle(
            frame,
            (item["cx"], item["cy"]),
            5,
            color,
            -1,
        )

        cv2.putText(
            frame,
            f"{label} #{item['id']} {item['class_name']}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
        )


def make_queue_window(
    fifo,
    ready_item,
    robot_busy,
):
    width = 600
    height = 680

    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    image[:] = (30, 30, 30)

    white = (245, 245, 245)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    orange = (0, 165, 255)
    green = (0, 255, 0)
    gray = (155, 155, 155)

    cv2.rectangle(
        image,
        (0, 0),
        (width, 118),
        (42, 42, 42),
        -1,
    )

    cv2.putText(
        image,
        f"{ROBOT_NAME} FIFO MEMORY",
        (22, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        white,
        2,
    )

    cv2.putText(
        image,
        f"Permanent queue | One-time {READY_DELAY_SEC:.0f}-second timer",
        (22, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        cyan,
        1,
    )

    robot_text = (
        "PICKING / DROPPING"
        if robot_busy
        else "VIEWING / SCANNING"
    )

    cv2.putText(
        image,
        f"ROBOT: {robot_text}",
        (22, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        orange if robot_busy else green,
        2,
    )

    front = fifo.front_item()

    cv2.rectangle(
        image,
        (18, 132),
        (width - 18, 208),
        (46, 46, 46),
        -1,
    )

    if ready_item is not None and not robot_busy:
        headline = (
            f"READY: #{ready_item['id']} "
            f"{ready_item['class_name']}"
        )
        detail = "Timer complete | Current position visible"
        headline_color = orange

    elif front is not None:
        remaining = fifo.remaining_time(front)

        headline = (
            f"FRONT: #{front['id']} "
            f"{front['class_name']}"
        )

        if remaining > 0:
            detail = f"Ready in {remaining:.2f} seconds"
        elif not front["visible"]:
            detail = "Timer complete | Waiting for visibility"
        else:
            detail = "Timer complete"

        headline_color = yellow

    else:
        headline = "QUEUE EMPTY"
        detail = "Waiting for a chip inside the polygon ROI"
        headline_color = white

    cv2.putText(
        image,
        headline,
        (32, 162),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        headline_color,
        2,
    )

    cv2.putText(
        image,
        detail,
        (32, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        gray,
        1,
    )

    cv2.putText(
        image,
        "ORDER",
        (22, 242),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        cyan,
        1,
    )

    y = 278

    for position, item in enumerate(
        fifo.active_items()[:10],
        start=1,
    ):
        color = white

        if front is item:
            color = yellow

        if ready_item is item and not robot_busy:
            color = orange

        remaining = fifo.remaining_time(item)

        if remaining > 0:
            timer_state = f"{remaining:.1f}s"
        else:
            timer_state = "READY"

        visibility = (
            "VISIBLE"
            if item["visible"]
            else "MEMORY"
        )

        cv2.rectangle(
            image,
            (18, y - 27),
            (width - 18, y + 24),
            (40, 40, 40),
            -1,
        )

        cv2.putText(
            image,
            (
                f"{position:02d}   #{item['id']}   "
                f"{item['class_name']}"
            ),
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.51,
            color,
            2,
        )

        cv2.putText(
            image,
            (
                f"timer {timer_state}   |   {visibility}"
            ),
            (300, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            cyan,
            1,
        )

        y += 58

    return image


def draw_status_bar(
    frame,
    robot_busy,
    detection_count,
    queue_count,
):
    _, width = frame.shape[:2]

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 78),
        (32, 32, 32),
        -1,
    )

    status = (
        "ROBOT PICKING | FIFO LOCKED"
        if robot_busy
        else "FIFO SCANNING | CONVEYOR ON"
    )

    cv2.putText(
        frame,
        f"{ROBOT_NAME}  |  {status}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        2,
    )

    cv2.putText(
        frame,
        (
            f"VALID DETECTIONS: {detection_count}   |   "
            f"QUEUE: {queue_count}   |   Q = QUIT"
        ),
        (16, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (255, 255, 0),
        2,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print(f"{ROBOT_NAME} PHASE 3 PERMANENT FIFO SYSTEM")
    print("Polygon ROI enabled.")
    print("Conveyor remains ON during normal operation.")
    print("Each chip is remembered permanently after first entry.")
    print("Each FIFO timer starts once and never resets.")
    print("=" * 70)

    confirmation = input(
        "Type YES to begin automatic robot and conveyor motion: "
    ).strip()

    if confirmation != "YES":
        print("Cancelled.")
        return

    print("Loading detector...")
    detector = Detector(
        confidence=CONFIDENCE_THRESHOLD,
    )

    fifo = SimpleFIFO(
        ready_delay_sec=READY_DELAY_SEC,
    )

    picker = Picker()

    robot = None
    conveyor_id = None
    robot_busy = False

    try:
        print(f"Connecting to {ROBOT_NAME}...")
        robot = NiryoRobot(ROBOT_IP)
        print(f"Connected to {ROBOT_NAME}.")

        robot.update_tool()

        picker.go_home(robot)

        conveyor_id = get_existing_conveyor(robot)
        start_conveyor(robot, conveyor_id)

        cv2.namedWindow(CAMERA_WINDOW)
        cv2.namedWindow(QUEUE_WINDOW)

        print("=" * 70)
        print("SYSTEM RUNNING")
        print("Timer starts when a chip first enters the polygon ROI.")
        print("FIFO item is removed only after successful drop-off.")
        print("Press Q to quit.")
        print("=" * 70)

        while True:
            frame = get_camera_frame(robot)

            if frame is None:
                print("Could not decode camera frame.")
                time.sleep(0.05)
                continue

            if not robot_busy:
                detections = detector.detect(frame)
                fifo.update(detections)
            else:
                detections = []

            ready_item = fifo.next_ready_item()

            detector.draw(
                frame,
                detections,
            )

            draw_memory_on_camera(
                frame,
                fifo,
                ready_item,
                robot_busy,
            )

            draw_status_bar(
                frame,
                robot_busy,
                len(detections),
                len(fifo.active_items()),
            )

            queue_window = make_queue_window(
                fifo,
                ready_item,
                robot_busy,
            )

            cv2.imshow(
                CAMERA_WINDOW,
                frame,
            )

            cv2.imshow(
                QUEUE_WINDOW,
                queue_window,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if ready_item is not None and not robot_busy:
                robot_busy = True
                locked_item = ready_item.copy()

                print("=" * 65)
                print(
                    "PICKING FIFO FRONT: "
                    f"#{locked_item['id']} "
                    f"{locked_item['class_name']}"
                )
                print(
                    "Locked pixel: "
                    f"({locked_item['cx']}, "
                    f"{locked_item['cy']})"
                )
                print("=" * 65)

                try:
                    success = picker.pick_and_place(
                        robot,
                        locked_item,
                    )

                except Exception as error:
                    print(f"Pick-and-place error: {error}")
                    success = False

                if success:
                    fifo.mark_complete(
                        ready_item,
                    )
                else:
                    fifo.keep_after_failure(
                        ready_item,
                    )

                time.sleep(POST_PICK_DELAY_SEC)
                robot_busy = False

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")

    except Exception as error:
        print("")
        print(f"{ROBOT_NAME} Phase 3 failed.")
        print(f"Error: {error}")

    finally:
        print("Shutting down Phase 3...")

        if robot is not None:
            stop_conveyor(
                robot,
                conveyor_id,
            )

            try:
                robot.release_with_tool()
            except Exception:
                pass

            try:
                robot.close_connection()
                print(f"Disconnected from {ROBOT_NAME}.")
            except Exception as error:
                print(f"Disconnect warning: {error}")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

