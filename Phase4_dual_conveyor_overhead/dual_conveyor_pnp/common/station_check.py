"""Phase 3 FIFO pick loop at one conveyor station.

Used by 18_run_fifo_system.py-style checks and by the robot worker when
the supervisor sends CHECK_CONVEYOR.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime

import cv2

from common.niryo_utils import get_frame
from common.station import Station, draw_station_roi
from common.wrist_vision import SimpleFIFO, WristDetector, draw_detections, draw_fifo


def log_pick(station: Station, item: dict, x: float, y: float, status: str) -> None:
    path = station.paths.pick_log
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new:
            writer.writerow(["timestamp", "robot", "conveyor", "class", "confidence",
                             "pixel_x", "pixel_y", "robot_x", "robot_y", "status"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), station.robot, station.conveyor,
            item["class_name"], f"{item['confidence']:.4f}", item["cx"], item["cy"],
            f"{x:.4f}", f"{y:.4f}", status,
        ])


def draw_banner(frame, lines, color=(0, 255, 255)) -> None:
    height = 22 + 22 * len(lines)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], height), (32, 32, 32), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color if i == 0 else (245, 245, 245), 1 if i else 2)


def run_station_check(
    niryo,
    station: Station,
    detector: WristDetector,
    runtime: dict,
    window: str,
    empty_timeout_sec: float | None,
    should_stop=lambda: False,
    on_tick=lambda: None,
    log=print,
) -> tuple[int, str]:
    """Pick chips at this station until it stays empty.

    Returns (picked_count, result) where result is COMPLETE, EMPTY,
    QUIT, ABORTED or ERROR. `empty_timeout_sec=None` runs until Q.
    """
    fifo = SimpleFIFO(runtime["dwell_sec"])
    max_picks = runtime["max_picks_per_check"] if empty_timeout_sec is not None else 10**9
    picked = 0
    last_seen = time.monotonic()
    label = f"{station.robot} | {station.conveyor}"

    station.go_view(niryo)
    time.sleep(runtime["view_settle_sec"])

    while True:
        on_tick()
        if should_stop():
            return picked, "ABORTED"

        frame = get_frame(niryo)
        if frame is None:
            time.sleep(0.05)
            continue

        detections = detector.detect(frame, station)
        fifo.update(detections)
        if detections or fifo.active_items():
            last_seen = time.monotonic()

        # The ROI for this conveyor is drawn only while at its view pose.
        draw_station_roi(frame, station.polygon, f"{station.conveyor} ROI")
        draw_detections(frame, detections)
        draw_fifo(frame, fifo)
        idle_for = time.monotonic() - last_seen
        status = f"CHECKING {station.conveyor} | picked {picked} | queue {len(fifo.active_items())}"
        if empty_timeout_sec is not None:
            status += f" | empty {idle_for:.1f}/{empty_timeout_sec:.1f}s"
        draw_banner(frame, [label, status])
        cv2.imshow(window, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            return picked, "QUIT"

        if empty_timeout_sec is not None and idle_for >= empty_timeout_sec:
            return picked, ("COMPLETE" if picked else "EMPTY")
        if picked >= max_picks:
            return picked, "COMPLETE"

        item = fifo.ready()
        if item is None:
            continue

        x, y = station.pixel_to_robot(item["cx"], item["cy"])
        log(f"{label}: picking #{item['id']} {item['class_name']} px=({item['cx']},{item['cy']}) "
            f"-> X={x:.4f} Y={y:.4f}")
        if not station.is_safe(x, y, runtime):
            log("UNSAFE TARGET")
            log_pick(station, item, x, y, "unsafe_target")
            fifo.failed(item)
            continue

        ok = station.pick_and_place(niryo, x, y, log=log)
        log_pick(station, item, x, y, "pick_complete" if ok else "pick_failed")
        if ok:
            fifo.complete(item)
            picked += 1
        else:
            fifo.failed(item)
            try:
                niryo.clear_collision_detected()
                station.go_view(niryo)
            except Exception as error:
                log(f"recovery failed: {error}")
                return picked, "ERROR"
        time.sleep(runtime["post_pick_delay_sec"])
        last_seen = time.monotonic()
