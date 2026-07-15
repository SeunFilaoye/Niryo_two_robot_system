"""
18_run_fifo_system.py

Simple Phase 3 FIFO system.

Rules:
1. A chip enters the single detection ROI.
2. Its class is added permanently to FIFO memory.
3. A four-second timer starts once and never resets.
4. Only the first FIFO item can be picked.
5. When its four-second timer finishes and it is visible, the robot picks it.
6. The item remains in memory until pick-and-place finishes successfully.
7. After successful drop-off and return home, the item is removed.
8. New detections are appended after all existing FIFO items.

No:
- trigger line
- collector ROI
- motion state
- idle state
- stability check
- age reset
"""

import math
import time
from typing import Optional

import cv2
import numpy as np
from pyniryo import ConveyorDirection, NiryoRobot

from vision.detector import Detector
from core.picker import Picker


# ============================================================
# SETTINGS
# ============================================================

ROBOT_IP = "192.168.0.199"

CONFIDENCE_THRESHOLD = 0.75
READY_DELAY_SEC = 4.0

CONVEYOR_SPEED = 50
CONVEYOR_DIRECTION = ConveyorDirection.FORWARD

POST_PICK_DELAY_SEC = 0.75

CAMERA_WINDOW = "Phase 3 FIFO Camera"
QUEUE_WINDOW = "Phase 3 FIFO Memory"


# ============================================================
# SIMPLE FIFO MEMORY
# ============================================================

class SimpleFIFO:
    """
    Permanent FIFO memory.

    The ready timer begins once when an item is created.
    It is never reset because of:
    - lost detection
    - position changes
    - bounding-box jitter
    - robot movement
    - chip movement

    An item is removed only after successful pick-and-place.
    """

    def __init__(self, ready_delay_sec: float = 4.0):
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
        now = time.time()

        item = {
            "id": self.next_id,
            "class_name": detection["class_name"],
            "confidence": detection["confidence"],
            "bbox": detection["bbox"],
            "cx": detection["cx"],
            "cy": detection["cy"],

            # Created once. Never reset.
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
        """
        Update only the current location.

        Never modify registered_at or ready_at.
        """
        item["class_name"] = detection["class_name"]
        item["confidence"] = detection["confidence"]
        item["bbox"] = detection["bbox"]
        item["cx"] = detection["cx"]
        item["cy"] = detection["cy"]
        item["visible"] = True

    def update(self, detections):
        """
        Match visible detections to existing FIFO entries first.

        Matching is performed separately for each class.

        If three blue squares are already in memory and three blue
        squares are detected, all three detections update the existing
        entries. No new queue items are created.

        If four blue squares are detected, the fourth detection becomes
        a new FIFO item at the end of the queue.
        """
        active = self.active_items()

        for item in active:
            item["visible"] = False

        classes = {
            detection["class_name"]
            for detection in detections
        }

        classes.update(
            item["class_name"]
            for item in active
        )

        for class_name in classes:
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

            # Build all possible item/detection pair distances.
            candidate_pairs = []

            for item_index, item in enumerate(class_items):
                for detection_index, detection in enumerate(class_detections):
                    candidate_pairs.append((
                        self.distance(item, detection),
                        item_index,
                        detection_index,
                    ))

            candidate_pairs.sort(
                key=lambda pair: pair[0]
            )

            matched_item_indexes = set()
            matched_detection_indexes = set()

            # Greedy nearest matching.
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

            # Only unmatched extra detections are new chips.
            for detection_index, detection in enumerate(class_detections):
                if detection_index not in matched_detection_indexes:
                    self.add_item(detection)

    def front_item(self):
        active = self.active_items()

        if not active:
            return None

        # The list already preserves insertion/FIFO order.
        return active[0]

    def next_ready_item(self):
        front = self.front_item()

        if front is None:
            return None

        timer_finished = time.time() >= front["ready_at"]

        # Timer never resets, but the robot needs a current visible
        # position before executing the pick.
        if timer_finished and front["visible"]:
            return front

        return None

    def remaining_time(self, item):
        return max(
            0.0,
            item["ready_at"] - time.time(),
        )

    def mark_complete(self, item):
        """
        Called only after vacuum release, drop-off and return home
        successfully finish inside picker.pick_and_place().
        """
        item["status"] = "complete"

        print(
            f"FIFO COMPLETE/REMOVED: "
            f"#{item['id']} {item['class_name']}"
        )

    def keep_after_failure(self, item):
        """
        Do not remove or reset the timer after a failed attempt.
        The same item remains first in FIFO.
        """
        item["visible"] = False

        print(
            f"FIFO PICK FAILED — ITEM RETAINED: "
            f"#{item['id']} {item['class_name']}"
        )


# ============================================================
# CAMERA
# ============================================================

def get_camera_frame(robot):
    compressed_image = robot.get_img_compressed()

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

def setup_conveyor(robot):
    conveyor_id = robot.set_conveyor()

    print(f"Conveyor connected: {conveyor_id}")

    return conveyor_id


def start_conveyor(robot, conveyor_id):
    robot.run_conveyor(
        conveyor_id,
        speed=CONVEYOR_SPEED,
        direction=CONVEYOR_DIRECTION,
    )

    print("Conveyor running.")


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
            label = "PRIORITY"

        else:
            color = (255, 255, 255)
            label = "FIFO"

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
    width = 620
    height = 700

    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    image[:] = (35, 35, 35)

    white = (245, 245, 245)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    orange = (0, 165, 255)
    green = (0, 255, 0)

    cv2.putText(
        image,
        "PHASE 3 FIFO MEMORY",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        white,
        2,
    )

    cv2.putText(
        image,
        "Permanent queue | Fixed 4-second timer",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
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
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        orange if robot_busy else green,
        2,
    )

    front = fifo.front_item()

    if ready_item is not None and not robot_busy:
        cv2.putText(
            image,
            (
                f"READY: #{ready_item['id']} "
                f"{ready_item['class_name']}"
            ),
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            orange,
            2,
        )

    elif front is not None:
        remaining = fifo.remaining_time(front)

        cv2.putText(
            image,
            (
                f"PRIORITY: #{front['id']} "
                f"{front['class_name']}"
            ),
            (20, 148),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            yellow,
            2,
        )

        if remaining > 0:
            timer_text = f"Ready in: {remaining:.2f}s"
        elif not front["visible"]:
            timer_text = "Timer complete — waiting for visibility"
        else:
            timer_text = "Timer complete"

        cv2.putText(
            image,
            timer_text,
            (20, 178),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            yellow,
            1,
        )

    else:
        cv2.putText(
            image,
            "QUEUE EMPTY",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            white,
            2,
        )

    y = 225

    for position, item in enumerate(
        fifo.active_items()[:11],
        start=1,
    ):
        color = white

        if front is item:
            color = yellow

        if ready_item is item and not robot_busy:
            color = orange

        visibility = (
            "VISIBLE"
            if item["visible"]
            else "MEMORY"
        )

        remaining = fifo.remaining_time(item)

        cv2.putText(
            image,
            (
                f"{position}. #{item['id']} "
                f"{item['class_name']}"
            ),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        cv2.putText(
            image,
            (
                f"   timer={remaining:.1f}s | "
                f"{visibility}"
            ),
            (20, y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
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
        (width, 72),
        (35, 35, 35),
        -1,
    )

    status = (
        "ROBOT PICKING — FIFO MEMORY LOCKED"
        if robot_busy
        else "FIFO SCANNING"
    )

    cv2.putText(
        frame,
        status,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        2,
    )

    cv2.putText(
        frame,
        (
            f"Detections: {detection_count} | "
            f"Queue: {queue_count} | Q=quit"
        ),
        (15, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 0),
        2,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading detector...")

    detector = Detector(
        confidence=CONFIDENCE_THRESHOLD,
    )

    fifo = SimpleFIFO(
        ready_delay_sec=READY_DELAY_SEC,
    )

    picker = Picker()

    print("Connecting robot...")

    robot = NiryoRobot(ROBOT_IP)

    print("Connected.")

    conveyor_id = None
    robot_busy = False

    try:
        robot.update_tool()

        print("Moving robot to viewing pose...")
        picker.go_home(robot)

        conveyor_id = setup_conveyor(robot)
        start_conveyor(robot, conveyor_id)

        print("=" * 65)
        print("SIMPLE PHASE 3 FIFO SYSTEM")
        print("Timer starts once when a chip enters the ROI.")
        print("Timer never resets.")
        print("No motion, idle or stability logic.")
        print("FIFO item is removed only after successful drop-off.")
        print("Conveyor remains ON.")
        print("=" * 65)

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
                # Do not alter FIFO while pick-and-place is running.
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

                # Lock the newest available location after the
                # independent four-second timer has completed.
                locked_item = ready_item.copy()

                print("=" * 60)
                print(
                    f"PICKING FIFO PRIORITY: "
                    f"#{locked_item['id']} "
                    f"{locked_item['class_name']}"
                )
                print(
                    f"Locked pixel: "
                    f"({locked_item['cx']}, "
                    f"{locked_item['cy']})"
                )
                print("=" * 60)

                try:
                    success = picker.pick_and_place(
                        robot,
                        locked_item,
                    )

                except Exception as error:
                    print(f"Pick-and-place error: {error}")
                    success = False

                if success:
                    # picker.pick_and_place() must return True only
                    # after release at drop-off and return home.
                    fifo.mark_complete(
                        ready_item,
                    )
                else:
                    # Never remove or reset the FIFO item.
                    fifo.keep_after_failure(
                        ready_item,
                    )

                time.sleep(
                    POST_PICK_DELAY_SEC,
                )

                robot_busy = False

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")

    finally:
        print("Shutting down Phase 3...")

        stop_conveyor(
            robot,
            conveyor_id,
        )

        cv2.destroyAllWindows()

        try:
            robot.close_connection()
        except Exception as error:
            print(f"Disconnect warning: {error}")

        print("Disconnected.")


if __name__ == "__main__":
    main()