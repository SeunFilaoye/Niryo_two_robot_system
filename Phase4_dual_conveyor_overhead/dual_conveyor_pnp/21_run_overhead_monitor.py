"""
21_run_overhead_monitor.py

Overhead AprilTag chip monitor for all eight conveyors C1-C8 (tags 0-7),
across every overhead camera. Mirrors robo_confirm/test_all_apriltag_rois.py
--detect-chips and adds the ALERT ROI.

Per conveyor tag (config/system.json "conveyors"):
  conveyor ROI  chips on that conveyor (display + zone counts)       all C1-C8
  entry ROI     PART_ENTERED_CONVEYOR info events                    all C1-C8
  ALERT ROI     CHIP_IN_ALERT_ZONE -> sent to the supervisor, which sends the
                conveyor's robot to its view pose to check and pick.
                Only conveyors owned by one robot (C1, C2 -> robot .150) have
                one; conveyors with "robot": null never raise alerts.

The first 30 detections of each tag are averaged and locked for the session;
the camera with the largest view of a tag owns it, and only the owner may
raise events. Restart after moving a camera or conveyor.

    python 21_run_overhead_monitor.py
    python 21_run_overhead_monitor.py --cameras 2 4
    python 21_run_overhead_monitor.py --no-detect-chips     # ROIs only

Q / Esc quits.
"""

import argparse
import csv
import json
import time
from datetime import datetime

import cv2

from common.apriltag_rois import (
    COLOR_ALERT, COLOR_CONVEYOR, COLOR_ENTRY, COLOR_PREVIEW, SessionTagReference,
    TagOwnership, ZoneGate, detect_tags, draw_polygon, draw_tag, inside_local,
    load_overhead_rois, local_to_pixels, make_detector, open_cameras,
    pixels_to_local, tag_local_corners,
)
from common.bus import TOPIC_ALERT, TOPIC_ENTRY, TOPIC_ZONES, Bus
from common.config import LOGS_DIR, alert_conveyors, load_system, model_path

COVERAGE_REPORT_AFTER_SEC = 10.0

EVENT_LOG = LOGS_DIR / "overhead_events.csv"


def log_event(event: dict) -> None:
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not EVENT_LOG.exists()
    with EVENT_LOG.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new:
            writer.writerow(["timestamp", "event", "conveyor", "tag_id", "class",
                             "confidence", "camera", "id"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event["event"],
            event.get("conveyor"), event.get("tag_id"), event.get("class_name"),
            event.get("confidence"), event.get("camera_index"), event.get("id"),
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", type=int, nargs="+")
    parser.add_argument("--detect-chips", dest="detect_chips", action="store_true", default=True)
    parser.add_argument("--no-detect-chips", dest="detect_chips", action="store_false")
    parser.add_argument("--model", help="override the overhead YOLO model")
    args = parser.parse_args()

    system = load_system()
    cfg = system["overhead"]
    conveyors = system["conveyors"]
    tag_to_conveyor = {c["tag_id"]: name for name, c in conveyors.items()}
    valid_tags = set(tag_to_conveyor)
    alerting = alert_conveyors(system)

    rois = load_overhead_rois()
    local_corners = tag_local_corners(rois.tag_size_in)
    for name in alerting:
        if not rois.alerts.get(name):
            print(f"WARNING: {name} has no ALERT ROI yet (run 20_draw_alert_rois.py). "
                  "No alerts will be raised for it.")

    model = None
    if args.detect_chips:
        from ultralytics import YOLO
        path = args.model or model_path("overhead", system)
        print(f"Overhead model: {path}")
        model = YOLO(str(path))

    indices = args.cameras or cfg.get("cameras") or range(cfg["max_camera_index"])
    captures = open_cameras(indices, cfg["frame_width"], cfg["frame_height"])
    if not captures:
        raise SystemExit(f"No working overhead camera among {list(indices)}")
    cameras = sorted(captures)
    print(f"Overhead cameras: {cameras} | tags {sorted(valid_tags)} | ALERT ROIs on {alerting}")

    bus = Bus("overhead", system)
    detector = make_detector()
    ownership = TagOwnership()
    refs = {(cam, tag): SessionTagReference(cfg["startup_frames"])
            for cam in cameras for tag in valid_tags}
    alert_gates = {name: ZoneGate(cfg["alert_stable_frames"], cfg["alert_rearm_empty_frames"])
                   for name in alerting}
    entry_gates = {name: ZoneGate(cfg["entry_stable_frames"], cfg["entry_rearm_empty_frames"])
                   for name in conveyors}
    counter = {"alert": 0, "entry": 0}
    last_zone_publish = 0.0
    started = time.monotonic()
    coverage_reported = False
    known_owners: dict[int, int] = {}
    windows = {cam: f"OVERHEAD CAMERA {cam}" for cam in cameras}
    for name in windows.values():
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 900, 675)

    def emit(kind: str, conveyor: str, camera: int, det: dict) -> None:
        counter[kind] += 1
        event = {
            "event": "CHIP_IN_ALERT_ZONE" if kind == "alert" else "PART_ENTERED_CONVEYOR",
            "id": f"{'A' if kind == 'alert' else 'P'}{counter[kind]:04d}",
            "conveyor": conveyor,
            "tag_id": conveyors[conveyor]["tag_id"],
            "robot": conveyors[conveyor]["robot"],
            "class_name": det["class_name"],
            "confidence": round(det["confidence"], 4),
            "camera_index": camera,
            "tag_position_in": {"x": round(det["tag_x"], 3), "y": round(det["tag_y"], 3)},
        }
        bus.send("supervisor", TOPIC_ALERT if kind == "alert" else TOPIC_ENTRY, event)
        log_event(event)
        print(json.dumps(event, sort_keys=True), flush=True)

    try:
        while True:
            frames = {}
            for cam, cap in captures.items():
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames[cam] = frame

            locked = {}
            for cam, frame in frames.items():
                tags = detect_tags(frame, detector, valid_tags)
                locked[cam] = {}
                for tag in valid_tags:
                    ref = refs[(cam, tag)]
                    if ref.observe(tags.get(tag)):
                        print(f"Camera {cam}, tag {tag}: AOI locked for this session.")
                    if ref.ready:
                        locked[cam][tag] = ref.locked
            ownership.update(locked)
            if ownership.owners != known_owners:
                for tag in sorted(set(ownership.owners) - set(known_owners)):
                    print(f"Tag {tag} ({tag_to_conveyor[tag]}) -> camera {ownership.owner(tag)}")
                known_owners = dict(ownership.owners)
            unseen = sorted(t for t in valid_tags if not any(refs[(c, t)].preview is not None for c in cameras))
            if not coverage_reported and time.monotonic() - started >= COVERAGE_REPORT_AFTER_SEC:
                coverage_reported = True
                owned = {t: ownership.owner(t) for t in sorted(valid_tags)}
                print("Tag coverage: " + ", ".join(f"{t}->{'cam ' + str(c) if c is not None else 'NONE'}"
                                                    for t, c in owned.items()))
                if unseen:
                    print(f"WARNING: tags {unseen} not seen by any camera "
                          f"({', '.join(tag_to_conveyor[t] for t in unseen)}). Chips there are not tracked.")

            alert_candidates = {name: [] for name in conveyors}
            entry_candidates = {name: [] for name in conveyors}
            zone_counts = {name: {"on_conveyor": 0, "in_alert": 0} for name in conveyors}
            displays = {cam: frame.copy() for cam, frame in frames.items()}

            for cam, frame in frames.items():
                display = displays[cam]
                for tag in valid_tags:
                    corners = refs[(cam, tag)].displayed()
                    if corners is None:
                        continue
                    name = tag_to_conveyor[tag]
                    is_owner = ownership.owner(tag) == cam and refs[(cam, tag)].ready
                    draw_tag(display, corners, tag, is_owner)
                    corners = rois.oriented(name, corners)
                    dim = 1.0 if is_owner else 0.45
                    scale = lambda c: tuple(int(v * dim) for v in c)
                    conveyor_color = COLOR_CONVEYOR if refs[(cam, tag)].ready else COLOR_PREVIEW
                    draw_polygon(display, local_to_pixels(rois.conveyor, corners, local_corners, rois.mode(name)),
                                 scale(conveyor_color), 0.04 if is_owner else 0, 1, name)
                    draw_polygon(display, local_to_pixels(rois.entry, corners, local_corners, rois.mode(name)),
                                 scale(COLOR_ENTRY), 0.04 if is_owner else 0, 1)
                    if name in alerting and rois.alerts.get(name):
                        draw_polygon(display, local_to_pixels(rois.alerts[name], corners, local_corners, rois.mode(name)),
                                     scale(COLOR_ALERT), 0.10 if is_owner else 0, 2, f"{name} ALERT")

                if model is None:
                    continue
                result = model(frame, conf=cfg["confidence"], verbose=False)[0]
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    det = {
                        "class_name": model.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2,
                    }
                    matches = []
                    for tag, corners in locked[cam].items():
                        corners = rois.oriented(tag_to_conveyor[tag], corners)
                        local = pixels_to_local([[det["cx"], det["cy"]]], corners, local_corners,
                                                rois.mode(tag_to_conveyor[tag]))
                        if local and inside_local(rois.conveyor, *local[0]):
                            matches.append((tag, local[0]))
                    if len(matches) != 1:
                        continue  # outside every conveyor or ambiguous
                    tag, (tx, ty) = matches[0]
                    name = tag_to_conveyor[tag]
                    det.update(tag_x=tx, tag_y=ty)
                    owner = ownership.owner(tag) == cam
                    in_alert = name in alerting and inside_local(rois.alerts.get(name), tx, ty)
                    color = COLOR_ALERT if in_alert else (0, 255, 0) if owner else (150, 150, 150)
                    cv2.rectangle(displays[cam], (x1, y1), (x2, y2), color, 2)
                    cv2.putText(displays[cam], det["class_name"], (x1, max(14, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
                    if not owner:
                        continue
                    zone_counts[name]["on_conveyor"] += 1
                    if in_alert:
                        zone_counts[name]["in_alert"] += 1
                        alert_candidates[name].append(det)
                    if inside_local(rois.entry, tx, ty):
                        entry_candidates[name].append(det)

            if model is not None:
                for name in conveyors:
                    tag = conveyors[name]["tag_id"]
                    owner = ownership.owner(tag)
                    observable = owner is not None and owner in frames and tag in locked.get(owner, {})
                    if name in alert_gates:
                        fired, rearmed = alert_gates[name].update(alert_candidates[name], observable)
                        if fired is not None:
                            emit("alert", name, owner, fired)
                        if rearmed:
                            print(json.dumps({"event": "ALERT_REARMED", "conveyor": name}), flush=True)
                    fired, _ = entry_gates[name].update(entry_candidates[name], observable)
                    if fired is not None:
                        emit("entry", name, owner, fired)
                    zone_counts[name]["observable"] = observable

                now = time.monotonic()
                if now - last_zone_publish >= cfg["zone_state_period_sec"]:
                    bus.send("supervisor", TOPIC_ZONES, {"zones": zone_counts})
                    last_zone_publish = now

            alert_text = "  ".join(
                f"{n}: {zone_counts[n]['in_alert']} alert/{zone_counts[n]['on_conveyor']}"
                for n in alerting)
            other_text = "  ".join(f"{n}:{zone_counts[n]['on_conveyor']}"
                                   for n in conveyors if n not in alerting)
            for cam, display in displays.items():
                owned = [t for t in sorted(valid_tags) if ownership.owner(t) == cam]
                header = f"cam {cam} | owns tags {' '.join(map(str, owned)) or '-'}"
                if unseen:
                    header += f" | not seen: {' '.join(map(str, unseen))}"
                cv2.rectangle(display, (0, 0), (display.shape[1], 20), (30, 30, 30), -1)
                cv2.putText(display, header, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1, cv2.LINE_AA)
                h = display.shape[0]
                cv2.rectangle(display, (0, h - 40), (display.shape[1], h), (30, 30, 30), -1)
                cv2.putText(display, alert_text, (8, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            COLOR_ALERT, 1, cv2.LINE_AA)
                cv2.putText(display, other_text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(windows[cam], display)

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        for cap in captures.values():
            cap.release()
        bus.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
