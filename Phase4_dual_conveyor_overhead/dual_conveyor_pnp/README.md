# Dual-Conveyor Overhead-Supervised Dynamic Pick-and-Place

Robot **192.168.0.150** with two conveyors (C1, C2), an overhead AprilTag
camera system, and a supervisor that sends the robot to a conveyor only when
the overhead camera sees a chip in that conveyor's ALERT zone.
(Robot .227 and its conveyors C5/C6 were dropped from this system on 2026-10-15.)

| Robot | Conveyor | AprilTag | Hardware ID | View pose |
|---|---|---|---|---|
| `robot_150` | C1 | 0 | ID_1 | saved |
| `robot_150` | C2 | 1 | ID_2 | saved |
| none (watched only) | C3-C8 | 2-7 | - | - |

The overhead cameras watch **all eight conveyors** (AprilTags 0-7 -> C1-C8)
across both overhead cameras: every conveyor gets its conveyor ROI (green) and
entry AOI (magenta), and chips are assigned to the conveyor they sit on.
Only **C1 and C2** have an ALERT ROI (red), because they are the only
conveyors owned by one robot (.150). A conveyor with `"robot": null` in
`config/system.json` never raises an alert. To give another conveyor alerts
later, set its `"robot"` and add it to that robot's `"conveyors"` list.

## Same training as the original robots

Scripts **01, 02, 03, 04, 05, 06, 08 and 18**, plus `core/picker.py` and
`vision/detector.py`, are copies of `dynamic_pnp/Robot 1/`. Phases 1 and 2
(07 and 09) are skipped. Only these things differ from the originals:

| Original | Here | Why |
|---|---|---|
| `ROBOT_NAME`, `ROBOT_IP`, `HOME_VIEW_JOINTS` typed into each script | `--robot 150 --conveyor C1` on the command line (`common/settings.py`) | Each robot has two conveyors, each with its own view pose |
| `MIDPOINT_JOINTS`, `DROP_JOINTS` typed into 06, 07, 09 and the picker | Recorded once with **04b**, read from `poses.json` | Same reason |
| Files saved in `Robot N/` | Files saved in `robots/robot_150/C1/` (same file names) | One folder per conveyor |
| Robot 1's hard-coded `SAFE_X/Y` box | Arm reach only: target 0.10–0.42 m from the base (`min_reach_m`, `max_reach_m`) | The homography covers the whole conveyor plane, so any chip in the 08 ROI can be picked |
| `PICK_X/Y_OFFSET_M` (Robots 2/3/5) | `"pick_offset_m": [dx, dy]` in `pick_reference.json`, default 0 | Same correction |
| 18 used the first conveyor it found | 18 uses this conveyor's hardware ID from **01b** | Two conveyors per robot |

New scripts: **01b** identifies the conveyors, **04b** records the midpoint and
drop, and **20–24** run the overhead system and the supervisor.

## Before every session

Open a terminal in VS Code (**Terminal → New Terminal**) and run:

```bash
cd ~/multi_robocv/dynamic_pnp/dual_conveyor_pnp
source ~/multi_robocv/.venv/bin/activate
```

How to read the instructions:
- **Terminal**: you type in the VS Code terminal.
- **Window**: click the camera window first, then press keys or click in it.
- Keep the e-stop within reach. Scripts that take `--conveyor` move the robot to that conveyor's view pose as soon as they connect.

---

## Training: per robot (once)

### 01_connection_test.py: check the connection
```bash
python 01_connection_test.py --robot 150
```
- Connects, prints "Connected successfully", and disconnects. The robot does not move.
- If it fails, check the robot's power, the network, and the IP.

### 10_clear_collision.py: if a move fails with "clear_collision_detected() must be called"
```bash
python 10_clear_collision.py --robot 150
```
- Clears the robot's collision flag, the same as Robot 3's `10_clear_collision.py`. The robot does not move.
- Check that nothing is blocking the arm, then rerun the script that failed.

### 01b_identify_conveyors.py: which conveyor is which
```bash
python 01b_identify_conveyors.py --robot 150
```
1. **Terminal:** type `YES` and press Enter.
2. The first conveyor runs for 2.5 s. **Terminal:** type the one that moved (`C1` or `C2`) and press Enter.
3. The same happens for the second conveyor.
4. The mapping is saved to `robots/robot_150/robot.json`. Robot .150 is already done (C1 = ID_1, C2 = ID_2).

### 02_wrist_camera_test.py: wrist camera only
```bash
python 02_wrist_camera_test.py --robot 150
```
- The robot does not move. A window shows the wrist camera.
- **Window:** `Q` quits.

---

## Training: per conveyor
Repeat this whole section for **C1** and **C2**.

### 03_test_yolo_detection.py: YOLO at the view pose
```bash
python 03_test_yolo_detection.py --robot 150 --conveyor C1
```
- The robot moves to C1's view pose.
- It draws boxes, class, confidence and center for each chip, plus the original fixed ROI rectangle and trigger line, which are for display only.
- Put chips on C1 and confirm they are detected with confidence ≥ 0.75.
- **Window:** `Q` quits.

### 04_workspace_calibration.py: homography, hover and contact (TCP)
```bash
python 04_workspace_calibration.py --robot 150 --conveyor C1
```
1. If a calibration exists, **Terminal:** type `YES` to replace it.
2. The robot moves to C1's view pose.
3. **Window:** left-click the 4 workspace corners in this order: **top-left → top-right → bottom-right → bottom-left**. `C` clears; `Q` continues once all 4 are set.
4. For each of the 4 corners, in the same order: move the nozzle tip onto that physical corner by hand (free-motion button on the arm), then press **Enter in the terminal**. It records the TCP X/Y.
5. **HOVER:** move the nozzle safely above a chip, then press Enter.
6. **CONTACT:** lower the nozzle to vacuum pickup height on the chip, then press Enter.
7. It saves `homography_matrix.npy`, `image_points.npy`, `robot_points_xy.npy`, `pick_reference.json` and `calibration_points.json` in `robots/robot_150/C1/calibration/`.

### 04b_record_midpoint_drop.py: midpoint and drop (joints + TCP)
> **Already set for C1 and C2** from the ROS2 `robot_positions.yaml` (2026-10-15):
> - midpoint = `home_joints` (the safe point between C1, C2, C3 and C5)
> - C1 drop = `D13` (onto conveyor 3)
> - C2 drop = `D15` (onto conveyor 5)
>
> This drop mapping is only for this test; in the ROS2 system the routing
> algorithm picks the drop. After each pickup the arm lifts back to the HOVER
> height recorded in 04, so `hold_joints` is not used. Run 04b only to change
> these or to record `--park`.
```bash
python 04b_record_midpoint_drop.py --robot 150 --conveyor C1
```
1. The robot moves to C1's view pose.
2. Move the arm to a **safe midpoint** between the conveyor and the drop, then press **Enter in the terminal**.
3. Move the arm to the **drop-off** position, then press Enter.
4. Both are saved to `robots/robot_150/C1/poses.json`.
   - To redo one: add `--only midpoint` or `--only drop`.
   - **Once per robot**, record where it waits between checks: `python 04b_record_midpoint_drop.py --robot 150 --park`.

### 05_test_pixel_to_robot.py: check the mapping
```bash
python 05_test_pixel_to_robot.py --robot 150 --conveyor C1
```
- The robot moves to the view pose.
- **Window:** left-click anywhere and the estimated robot X/Y (m) shows in the window and the terminal. `Q` quits.
- Click the 4 corners you calibrated. They should return the X/Y recorded in 04.

### 06_manual_pick_test.py: physical pick, no YOLO
```bash
python 06_manual_pick_test.py --robot 150 --conveyor C1
```
1. Place one chip exactly where you recorded CONTACT in 04.
2. **Terminal:** type `YES`.
3. The robot's conveyors are stopped. The robot goes view → hover → contact → vacuum on → hover → midpoint → drop → vacuum off → midpoint → view.
4. If the drop or path is wrong, rerun 04b.

### 08_set_polygon_roi_and_priority.py: wrist-camera ROI for this view
```bash
python 08_set_polygon_roi_and_priority.py --robot 150 --conveyor C1
```
- The robot moves to C1's view pose.
- **Window:**
  - **left-click**: add a polygon vertex
  - `P`: close the polygon
  - **right-click**: set the priority point
  - `C`: clear the polygon
  - `R`: clear the priority point
  - `S`: save
  - `Q`: quit
- It saves `robots/robot_150/C1/phase2_config.json`. This ROI is the one that appears whenever the robot is at C1's view pose.

### 18_run_fifo_system.py: Phase 3 FIFO dynamic pick (this conveyor alone)
```bash
python 18_run_fifo_system.py --robot 150 --conveyor C1
```
1. **Terminal:** type `YES`.
2. The robot goes to C1's view pose and C1 starts and stays on.
3. Each chip that enters the ROI joins the FIFO with a one-time 3 s dwell timer. The front chip is picked once its timer is done and it is visible.
4. Two windows open: the camera, and the FIFO memory list.
5. **Window:** `Q` stops it. The conveyor stops on exit.

Once 18 works on every conveyor, training for that robot is done.

---

## Overhead + supervisor (after training)

### 21_run_overhead_monitor.py --no-detect-chips: check the tags
```bash
python 21_run_overhead_monitor.py --no-detect-chips
```
- It opens every overhead camera and looks for tags **0-7** (C1-C8).
- After 30 frames each tag **locks**, and its conveyor ROI (green) and entry ROI (magenta) are drawn.
- The camera with the largest view of a tag **owns** it (bright outline; other cameras draw it dim). Only the owner reports chips on that conveyor, so a chip seen by both cameras is counted once.
- The top bar shows which tags each camera owns and any tag **not seen** by any camera. After 10 s the terminal prints a `Tag coverage:` line, e.g. `0->cam 0, ..., 5->cam 2`.
- The bottom bar shows `C1/C2: alert/on-conveyor` counts (red) and chip counts for C3-C8.
- Keep the cameras and conveyors still. **Window:** `Q` or Esc quits.

### 20_draw_alert_rois.py: draw the third ROI (ALERT) per conveyor
```bash
python 20_draw_alert_rois.py
```
1. **Window:** press `1`-`8` to pick a conveyor (ALERT ROIs can only be saved on `1` / `2`; `T` and `O` work on all eight), and `N` to switch cameras until that tag shows **LOCKED**.
   - If a conveyor's green ROI **gets wider (or narrower) the farther it is from the tag**, press `T` to switch that conveyor's ROI mapping: `homography` → `similarity` → `affine`. Similarity has no perspective term, so the ROI keeps its real shape at any distance; small tag-corner errors no longer blow up. It saves immediately (`roi_mapping` in `config/overhead_rois.json`). C2 starts on `similarity`; C1 stays on `homography`, which already looked right.
   - If a conveyor's green ROI points the wrong way (for example across the screen), its tag is mounted turned relative to tag 0. Press `O` until the green ROI lies on the conveyor. Each press turns it 90°, and the setting is saved right away (`tag_quarter_turns` in `config/overhead_rois.json`). Fix the orientation **before** drawing that conveyor's ALERT ROI.
2. Left-click the corners of the ALERT zone (the area the robot can reach). `U` undoes, `C` clears.
3. `S` saves. The red ALERT ROI appears and is stored in tag-local inches in `config/overhead_rois.json`. `D` deletes it. `Q` quits.

### 21_run_overhead_monitor.py: chip detection and alerts
```bash
python 21_run_overhead_monitor.py
```
- Chips on any of C1-C8 are counted and `PART_ENTERED_CONVEYOR` fires for every conveyor's entry AOI.
- When a chip stays in C1's or C2's ALERT ROI for 4 frames, it prints `CHIP_IN_ALERT_ZONE` and sends it to the supervisor.
- It re-arms after the zone has been empty for 12 frames.

### Full system: three terminals, same folder and venv
```bash
python 22_run_supervisor.py
```
```bash
python 23_run_robot_worker.py --robot 150
```
```bash
python 21_run_overhead_monitor.py
```
- The worker asks for `YES`, starts its conveyors, parks, and waits.
- On an alert for C1, the supervisor sends .150 to check C1. The robot goes to C1's view pose, C1's ROI appears, and it runs the same FIFO pick as 18 until C1 has been empty for 3 s. Then it reports back and parks.

Watch all messages:
```bash
python 24_bus_monitor.py
```

Test without chips or cameras: run 22 and one worker, then:
```bash
python 24_bus_monitor.py --fake-alert C1
```

## Folder layout
```
dual_conveyor_pnp/
├── config/system.json            robots, conveyors↔tags, ports, timing
├── config/overhead_rois.json     tag-local ROIs (conveyor + entry shared, ALERT per conveyor)
├── core/picker.py, vision/detector.py   original Phase 3 modules
├── common/                       settings, stations, AprilTag ROIs, bus
├── robots/robot_150/
│   ├── robot.json                conveyor hardware IDs, park pose
│   └── C1/ poses.json, phase2_config.json, calibration/, logs/
```
Models:
- Wrist: `dynamic_pnp/models/best.pt` (copied from robo_confirm).
- Overhead: `chip_training/models/best_chip_v13.pt`.

## Supervisor rules
- Alerts and zone reports for conveyors without a robot (C3-C8) are ignored.
- Each alert queues its conveyor once for the robot that owns it. Duplicates are ignored while that conveyor is queued or being checked.
- Each robot runs one check at a time, oldest request first.
- If the ALERT zone is still occupied 6 s after a check of that conveyor, it is queued again.
- A robot with no heartbeat for 5 s is OFFLINE, and its in-flight check is re-queued.

## Bus topics (planned ROS2 topics, std_msgs/String JSON)
| Topic | From → To | Payload |
|---|---|---|
| `/overhead/alert` | overhead → supervisor | `{"event":"CHIP_IN_ALERT_ZONE","conveyor":"C1","tag_id":0,"robot":"robot_150",...}` |
| `/overhead/entry` | overhead → supervisor | `PART_ENTERED_CONVEYOR` (info) |
| `/overhead/zone_state` | overhead → supervisor | `{"zones":{"C1":{"on_conveyor":2,"in_alert":1,"observable":true}}}` |
| `/supervisor/robot_150/command` | supervisor → robot | `{"command":"CHECK_CONVEYOR","conveyor":"C1","request_id":"R0001"}` |
| `/robot_150/status` | robot → supervisor (1 Hz) | `{"state":"IDLE|CHECKING|RETURNING|OFFLINE","conveyor":"C1"}` |
| `/robot_150/check_result` | robot → supervisor | `{"request_id":"R0001","conveyor":"C1","result":"COMPLETE|EMPTY|ERROR","picked":2}` |

ROS2 integration: see [ROS2_INTEGRATION.md](ROS2_INTEGRATION.md) (step-by-step guide for an AI agent).
