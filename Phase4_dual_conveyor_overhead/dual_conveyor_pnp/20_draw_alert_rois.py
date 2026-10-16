"""
20_draw_alert_rois.py

Draw the third overhead ROI - the ALERT zone - for each robot-owned conveyor.

All eight conveyor tags (0-7 -> C1-C8) are shown with their conveyor and
entry ROIs. Only conveyors owned by one robot (C1, C2 -> robot .150; see
"robot" in config/system.json) can have an ALERT ROI.

Overhead ROIs per conveyor (all anchored to that conveyor's AprilTag):
  green   conveyor ROI  (shared template from robo_confirm)
  magenta entry ROI     (shared template from robo_confirm)
  red     ALERT ROI     (drawn here, per conveyor) -> a stable chip inside it
                        tells the supervisor to send the robot to check
                        that conveyor

Points are clicked on the overhead image and saved in AprilTag-local inches,
so the ALERT ROI follows the tag like the other two.

Keys:
  1 - 8           select conveyor C1 - C8 (ALERT ROI only on C1 / C2)
  T               ROI mapping for the selected conveyor (saved at once):
                  homography -> similarity -> affine. Use similarity when the
                  ROI widens the farther it gets from the tag.
  O               rotate the selected tag's ROIs by 90 deg (saved at once) -
                  use it when a tag is mounted turned relative to tag 0
  N               next camera
  LEFT CLICK      add point (tag must be LOCKED)
  U / C           undo / clear points
  S               save ALERT ROI for the selected conveyor
  D               delete the saved ALERT ROI for the selected conveyor
  Q / Esc         quit

    python 20_draw_alert_rois.py
    python 20_draw_alert_rois.py --cameras 2 4
    python 20_draw_alert_rois.py --delete C2        # remove a saved ALERT ROI
"""

import argparse

import cv2

from common.apriltag_rois import (
    COLOR_ALERT, COLOR_CONVEYOR, COLOR_ENTRY, SessionTagReference, detect_tags,
    draw_polygon, draw_tag, load_overhead_rois, local_to_pixels, make_detector,
    MAPPING_MODES, open_cameras, pixels_to_local, save_alert_roi, save_mapping,
    save_quarter_turns, tag_local_corners,
)
from common.config import alert_conveyors, conveyor_key, load_system


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", type=int, nargs="+")
    parser.add_argument("--delete", nargs="+", metavar="CONVEYOR",
                        help="delete the saved ALERT ROI of these conveyors and exit")
    args = parser.parse_args()

    if args.delete:
        for name in args.delete:
            name = conveyor_key(name)
            save_alert_roi(name, None)
            print(f"Deleted ALERT ROI for {name}.")
        return

    system = load_system()
    overhead = system["overhead"]
    tag_by_conveyor = {name: c["tag_id"] for name, c in system["conveyors"].items()}
    conveyor_by_tag = {tag: name for name, tag in tag_by_conveyor.items()}
    valid_tags = set(conveyor_by_tag)
    alerting = alert_conveyors(system)

    indices = args.cameras or overhead.get("cameras") or range(overhead["max_camera_index"])
    captures = open_cameras(indices, overhead["frame_width"], overhead["frame_height"])
    if not captures:
        raise SystemExit(f"No working overhead camera among {list(indices)}")
    cameras = sorted(captures)
    print(f"Cameras: {cameras}")

    rois = load_overhead_rois()
    local_corners = tag_local_corners(rois.tag_size_in)
    refs = {(cam, tag): SessionTagReference(overhead["startup_frames"])
            for cam in cameras for tag in valid_tags}
    detector = make_detector()

    state = {"camera_i": 0, "conveyor": "C1", "points": []}
    window = "ALERT ROI editor"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["points"].append([x, y])

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 720)
    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            camera = cameras[state["camera_i"]]
            ok, frame = captures[camera].read()
            if not ok or frame is None:
                continue
            tags = detect_tags(frame, detector, valid_tags)
            for tag in valid_tags:
                ref = refs[(camera, tag)]
                if ref.observe(tags.get(tag)):
                    print(f"Camera {camera} tag {tag}: LOCKED")

            selected_tag = tag_by_conveyor[state["conveyor"]]
            display = frame.copy()
            for tag in valid_tags:
                corners = refs[(camera, tag)].displayed()
                if corners is None:
                    continue
                name = conveyor_by_tag[tag]
                draw_tag(display, corners, tag, tag == selected_tag)
                corners = rois.oriented(name, corners)
                draw_polygon(display, local_to_pixels(rois.conveyor, corners, local_corners, rois.mode(name)),
                             COLOR_CONVEYOR, 0.03, 1, name)
                draw_polygon(display, local_to_pixels(rois.entry, corners, local_corners, rois.mode(name)),
                             COLOR_ENTRY, 0.03, 1)
                alert = rois.alerts.get(name) if name in alerting else None
                if alert:
                    draw_polygon(display, local_to_pixels(alert, corners, local_corners, rois.mode(name)),
                                 COLOR_ALERT, 0.10, 2, f"{name} ALERT")

            if state["points"]:
                for x, y in state["points"]:
                    cv2.circle(display, (x, y), 4, (0, 255, 255), -1)
                draw_polygon(display, state["points"], (0, 255, 255), 0.15, 2)

            ref = refs[(camera, selected_tag)]
            lock = "LOCKED" if ref.ready else (
                f"averaging {len(ref.samples)}/{ref.sample_count}" if ref.preview is not None
                else "NOT VISIBLE in this camera")
            cv2.rectangle(display, (0, 0), (display.shape[1], 46), (30, 30, 30), -1)
            turns = int(rois.quarter_turns.get(state["conveyor"], 0)) % 4
            cv2.putText(display, f"camera {camera} | {state['conveyor']} (tag {selected_tag}, {rois.mode(state['conveyor'])}, {turns * 90} deg) {lock}",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(display, "1-8 conveyor | T mapping | O rotate | N camera | click | U undo | C clear | S save | D delete | Q quit",
                        (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (245, 245, 245), 1)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if ord("A") <= key <= ord("Z"):
                key += 32  # Caps Lock / Shift: treat D like d
            if key in (ord("q"), 27):
                break
            if key < 128 and f"C{chr(key)}" in tag_by_conveyor:
                state["conveyor"], state["points"] = f"C{chr(key)}", []
            elif key == ord("n"):
                state["camera_i"] = (state["camera_i"] + 1) % len(cameras)
                state["points"] = []
            elif key == ord("u") and state["points"]:
                state["points"].pop()
            elif key == ord("c"):
                state["points"] = []
            elif key == ord("t"):
                current = rois.mode(state["conveyor"])
                mode = MAPPING_MODES[(MAPPING_MODES.index(current) + 1) % len(MAPPING_MODES)]
                save_mapping(state["conveyor"], mode)
                rois = load_overhead_rois()
                print(f"{state['conveyor']} ROI mapping: {mode} (saved)")
            elif key == ord("o"):
                turns = (int(rois.quarter_turns.get(state["conveyor"], 0)) + 1) % 4
                save_quarter_turns(state["conveyor"], turns)
                rois = load_overhead_rois()
                print(f"{state['conveyor']} tag orientation: {turns * 90} deg (saved)")
            elif key == ord("d"):
                save_alert_roi(state["conveyor"], None)
                rois = load_overhead_rois()
                print(f"Deleted ALERT ROI for {state['conveyor']}")
            elif key == ord("s"):
                if state["conveyor"] not in alerting:
                    print(f"{state['conveyor']} has no robot, so it gets no ALERT ROI "
                          f"(only {', '.join(alerting)}). T / O still work for it.")
                elif not ref.ready:
                    print("Tag is not LOCKED in this camera yet; cannot save.")
                elif len(state["points"]) < 3:
                    print("Need at least 3 points.")
                else:
                    local = pixels_to_local(state["points"], rois.oriented(state["conveyor"], ref.locked),
                                            local_corners, rois.mode(state["conveyor"]))
                    save_alert_roi(state["conveyor"], [[round(x, 4), round(y, 4)] for x, y in local])
                    rois = load_overhead_rois()
                    state["points"] = []
                    print(f"Saved ALERT ROI for {state['conveyor']} ({len(local)} points, tag-local inches)")
    finally:
        for cap in captures.values():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
