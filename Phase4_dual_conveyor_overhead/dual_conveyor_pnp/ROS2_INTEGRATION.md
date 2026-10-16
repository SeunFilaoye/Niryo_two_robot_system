# ROS2 Integration Guide: Overhead-Supervised Dynamic Pick for Robot R1 (192.168.0.150)

**Audience:** an AI coding agent integrating this folder into the ROS2 testbed at
`~/multi_robocv/robo_confirm`. Read this whole file before editing anything.
Follow the phases in order, and finish each phase's checks before starting the next.

---

## Two rules for R1 / C1 / C2 (read first)

1. **No IR sensors.** This system does not use the conveyor IR sensors on C1 or C2 at all. The overhead camera's ALERT ROI is the only pickup trigger. IR edges on conveyors 1 and 2 must not stop a belt, must not publish `ir_trigger`, and must not produce `IR_READY`.
2. **C1 and C2 stay ON all the time.** Both belts are started once at bring-up and run continuously: during detection, while the robot moves to a view pose, during the pick, and while the robot is away at the other conveyor. Nothing in the pick flow sends `STOP` to them. They stop only on shutdown or e-stop. This is what "dynamic pick" means here: the robot picks a moving chip.

## 0. What exists and works today (do not change its behavior)

`~/multi_robocv/dynamic_pnp/dual_conveyor_pnp/` runs as plain Python with
pyniryo. It was validated end to end on hardware on 2026-10-15:

1. Overhead cameras watch AprilTag 0 (conveyor C1) and tag 1 (C2).
2. A chip that stays in a conveyor's **ALERT ROI** for 4 frames raises `CHIP_IN_ALERT_ZONE`.
3. The supervisor sends robot .150 to that conveyor's **view pose**.
4. The wrist camera runs YOLO inside that conveyor's polygon ROI.
5. The FIFO waits **3 s** of dwell, then the robot picks with the conveyor still running:
   - hover → contact → vacuum → **back up to hover**
   - midpoint (`home_joints`) → drop → release → midpoint → view
6. Once the station ROI has been empty for 3 s, the robot reports back and parks.

| Piece | File | Becomes in ROS2 |
|---|---|---|
| Overhead tag ROIs (all tags 0-7 / C1-C8) + chip alerts (C1, C2 only) | `21_run_overhead_monitor.py`, `common/apriltag_rois.py`, `config/overhead_rois.json` | **New node** `overhead_alert_node.py` |
| Supervisor (alert → which conveyor to check) | `22_run_supervisor.py` | superfactory's existing pickup gate, fed by the overhead `CHIP_READY` event instead of IR (Phase 4) |
| Robot worker (go to view, FIFO pick) | `23_run_robot_worker.py`, `common/station*.py` | Existing `robot_controller.py` + `niryo_factory_node.py`, extended for R1 (Phases 2–3) |
| Wrist YOLO + homography | `vision/detector.py`, `core/picker.py` | Existing `cv_robot_node.py`, made station-aware for R1 (Phase 2) |
| Message bus (UDP JSON) | `common/bus.py` | `std_msgs/String` JSON topics |
| Per-conveyor calibration | `robots/robot_150/C1/`, `robots/robot_150/C2/` | `niryo_vendor/cv_assets/robot1/P11/`, `.../P12/` |

Mapping between the two naming schemes:

| Here | ROS2 testbed |
|---|---|
| `robot_150` (192.168.0.150) | **R1**, driver namespace `/robot1`, command topics `/robot_1/...` |
| C1 (tag 0, pyniryo `ConveyorID.ID_1`) | `conveyor_1`, robot1 hw id 10, pickup key **P11** |
| C2 (tag 1, pyniryo `ConveyorID.ID_2`) | `conveyor_2`, robot1 hw id 9, pickup key **P12** |
| midpoint | `robot_1.home_joints` (already in `robot_positions.yaml`) |
| drop (test only: C1→D13, C2→D15) | decided by routing: `node_mapping.yaml` command templates |

Evidence for .150 = R1: `robo_confirm/check_direct_conveyors.py` and
`record_conveyor_dataset.py` both map `"R1": "192.168.0.150"`.
`conveyors_config.yaml` puts conveyor_1 and conveyor_2 on `robot_namespace: "robot1"`.
(`test_multi_conveyor_fifo.py` maps "C2" to .150; that file uses an older scheme, so ignore it.)

---

## 1. Facts about the target system (verified 2026-10-15)

All paths below are relative to `~/multi_robocv/robo_confirm/`.

**Packaging and launch**
- There is no colcon package or launch file. Every node is a plain script started with `python3 <script>.py [id]` after sourcing ROS2.
- README.md says Humble; this PC has a Jazzy source build at `~/ros2_jazzy`. Check `echo $ROS_DISTRO` on the ROS2 computer and use that one.

**Robot motion** (`niryo_vendor/base_robot.py`)
- Arm moves use the action `/robot{N}/niryo_robot_arm_commander/robot_action` (`RobotMove`, `ArmMoveCommand.JOINTS` / `POSE`).
- The vacuum uses `ToolCommand` services `/robot{N}/niryo_robot/tools/pull_air_vacuum_pump` and `push_air_vacuum_pump`.
- Named positions live in `niryo_vendor/robot_positions.yaml` and run through `move_to_position(key)`.
- robot_1 already has `home_joints`, `hold_joints`, `P11`–`P13`, `D12`, `D13` and `D15`. It has no CV view pose.

**Command flow**
1. `superfactory.py` → `/niryo_factory/command` (`{"src","cmd","token"}`, ACL-checked) → `niryo_factory_node.py` → `/robot_1/command`.
2. The robot replies on `/robot_1/status` (`IDLE` = done) and `/robot_1/ack`.
3. The factory then reports `COMMAND_COMPLETE:R1` on `/niryo_factory/status`.
- **Do not publish to `/niryo_factory/command` from a new node.** `fault_containment/acl_configs/niryo_acl.yaml` only accepts `/super_factory` with a token.

**Routing and drops**
- `algorithm_vendor/node_mapping.yaml` command templates decide the drop:
  - `C1_PICKUP→C5_PICKUP`: `"R1 | PICKUP - P11 | DROP - D15"`
  - `C2_PICKUP→C3_PICKUP`: `"R1 | PICKUP - P12 | DROP - D13"`
  - `C2_PICKUP→C5_PICKUP`: `"R1 | PICKUP - P12 | DROP - D15"`
- **In production, C1 only routes to D15.** The C1→D13 drop in this folder was a bench test; do not port it.

**Existing CV pick path** (robots 2, 3, 5, 6)
- `cv_robot_node.py <id>` subscribes to `/robot{N}/niryo_robot_vision/compressed_video_stream`. It runs YOLO at 1 Hz on `cv_assets/models/best.pt` (conf 0.50), using `cv_assets/robotN/phase2_config.json` (`polygon_roi`) and `cv_assets/robotN/calibration/homography_matrix.npy`.
- It publishes `/robot{N}/cv_detection` as a JSON String with keys `robot`, `robot_id`, `class`, `confidence`, `pixel_x`, `pixel_y`, `robot_x`, `robot_y`. It sends only the highest-confidence detection inside the ROI.
- `niryo_factory_node.py`, when it receives a pick/place command for a CV robot:
  1. Arms the command.
  2. Waits for a **new** detection, then a dwell (`cv_ready_delay = 5.75` s), and requires the detection to still be visible (age ≤ 0.75 s). The timeout is 30 s.
  3. Sends `{"cmd": "<route cmd>", "cv_pick": true, "cv_x", "cv_y", "cv_class", "cv_confidence"}` to `/robot_{N}/command`.
- `robot_controller.py` CV path:
  1. Adds `cv_x_offset` / `cv_y_offset` from `cv_assets/robotN/runtime_config.json`.
  2. Moves to the hover pose and the contact pose (Z and RPY from `cv_assets/robotN/calibration/pick_reference.json` → `hover_pose`, `contact_pose`), then grips.
  3. Goes to **`hold_joints`**, then the route's drop, releases, and returns to `cv_view_joints`.
- **The CV path is hard-gated to robots (2, 3, 5, 6) in five places:**
  - `robot_controller.py` (`self.cv_enabled = self.robot_id in (2, 3, 5, 6)`)
  - `niryo_factory_node.py` (`self.cv_robot_ids = (2, 3, 5, 6)`)
  - `cv_robot_node.py` (`CV_ROBOTS = (2, 3, 5, 6)`)
  - `conveyor_control.py` (`rid in {2, 3, 5, 6}`)
  - `conveyor_controllers.py` (`rid in {2, 3, 5, 6}`)
  - `super_factory/superfactory.py` has a separate set, `cv_robot_ids = {'R2', 'R3', 'R5', 'R6'}`, which **skips** the pickup gate. **Do not add R1 there.** R1 keeps the gate, but in Phase 4 the gate is opened by the overhead `CHIP_READY` event, not by IR.
- Each CV robot has **one** view pose and one calibration. R1 needs **two** (P11 and P12). That is the main structural change.

**Conveyors** (`niryo_vendor/conveyor_control.py <id>`)
- Run one process per conveyor.
- Commands arrive on `/niryo_factory/conveyor_{id}/control_cmd` (`START` / `STOP`). Status goes to `/niryo_factory/conveyor_{id}/status`.
- An IR edge **stops the belt** and publishes `/niryo_factory/conveyor_{id}/ir_trigger`, except for CV robots, which keep continuous flow.
- `conveyor_factory_node.py` turns the trigger into `/conveyor_factory/status` = `"IR_READY | R1 | P11 | C1"`.
- `superfactory.py` holds non-CV pickups until that `IR_READY` arrives (`status_msg.upper().startswith('IR_READY')` → `_handle_ir_ready()`). That handler only uses the pickup key (`parts[2]`, e.g. `P11`) to release the waiting task.
- **For C1/C2 this whole IR chain is switched off (Phase 4).** It stays as-is for every other conveyor.

**Overhead**
- Nothing in robo_confirm publishes overhead events on ROS2 today. `test_all_apriltag_rois.py` only prints JSON lines to stdout.

---

## 2. Integration design

```
overhead_alert_node.py ── /overhead/alert (JSON) ───────────────► logs / dashboard
        │
        └─ CHIP_IN_ALERT_ZONE C1 ─► /overhead/pickup_ready "CHIP_READY | R1 | P11 | C1"
                                     (overhead camera, NOT an IR sensor)
superfactory (existing pickup gate, opened by CHIP_READY) ─► "R1 | PICKUP - P11 | DROP - D15" ─► niryo_factory_node
niryo_factory_node (R1 = CV robot with stations):
   1. /robot_1/command  "R1 | SEQUENCE | cv_view_P11"          (go to C1 view pose)
   2. /robot_1/cv_station  "P11"                               (tell the CV node which station)
   3. wait for /robot1/cv_detection with station == "P11" + 3.0 s dwell
   4. /robot_1/command  {"cmd": "...P11...D15", "cv_pick": true, "cv_x", "cv_y", "station": "P11", ...}
robot_controller 1:  hover → contact → grip → hover → home_joints → D15 → release → home_joints → cv_view_P11
```

Key decisions:
- **The overhead ALERT is the only trigger for C1/C2; IR sensors are not used.** The overhead node publishes `CHIP_READY | R1 | P11 | C1` on its own topic, `/overhead/pickup_ready`. superfactory subscribes to it and passes it to the same handler `IR_READY` uses, so the gate logic is reused and the IR hardware is not. The routing algorithm still chooses where the chip goes.
- **The existing CV dwell/lock logic in `niryo_factory_node.py` stays.** For R1 only, it gets a "move to station view first" step and `cv_ready_delay = 3.0`.
- **C1 and C2 run continuously.** They are started once at bring-up and never stopped by the pick flow or by IR (see "Two rules" at the top).
- **R1 retreats to hover after the grip, not to `hold_joints`.** The user asked for this; the recorded hover height is the safe retreat. The other robots keep `hold_joints`.

---

## 3. Phase 1: Copy the R1 assets (no code changes)

Source: `dynamic_pnp/dual_conveyor_pnp/robots/robot_150/`.
Target: `robo_confirm/niryo_vendor/cv_assets/robot1/`.

```bash
SRC=~/multi_robocv/dynamic_pnp/dual_conveyor_pnp/robots/robot_150
DST=~/multi_robocv/robo_confirm/niryo_vendor/cv_assets/robot1
mkdir -p $DST/P11 $DST/P12
cp -r $SRC/C1/calibration $SRC/C1/phase2_config.json $DST/P11/
cp -r $SRC/C2/calibration $SRC/C2/phase2_config.json $DST/P12/
```

Create `$DST/P11/runtime_config.json` and `$DST/P12/runtime_config.json`. The offsets come from `pick_offset_m` in that station's `calibration/pick_reference.json` (default 0):
```json
{"robot_id": 1, "station": "P11", "cv_x_offset": 0.0, "cv_y_offset": 0.0,
 "validated_pick_place": true,
 "notes": "Validated with dual_conveyor_pnp 18_run_fifo_system.py on 2026-10-15."}
```

Add to `robot_1:` in `niryo_vendor/robot_positions.yaml`. Copy the joints from each station's `poses.json` → `poses.view.joints`; these are the values as of 2026-10-15:
```yaml
  cv_view_P11:
    joints: [-1.6626, 0.3267, -0.7628, -0.0428, -1.2088, -1.2117]
    description: Wrist-camera view of conveyor 1 (homography/ROI valid only here)
    type: CV_VIEW
  cv_view_P12:
    joints: [-0.5288, 0.1767, -0.7946, 0.0277, -0.9557, -1.2117]
    description: Wrist-camera view of conveyor 2 (homography/ROI valid only here)
    type: CV_VIEW
  cv_view_joints:
    joints: [-1.6626, 0.3267, -0.7628, -0.0428, -1.2088, -1.2117]
    description: Default CV park/view (= cv_view_P11; existing code expects this key)
    type: CV_VIEW
```

**Checks**
- `pick_reference.json` has `hover_pose` and `contact_pose`, each with `x y z roll pitch yaw`. This is the same format `robot_controller.py` already reads.
- `md5sum cv_assets/models/best.pt` matches `dynamic_pnp/models/best.pt` (`9e7ea7bc08d145c07f3838daf268554a`).
- A YAML load of `robot_positions.yaml` succeeds and `robot_1` keeps every existing key.

**Must hold true:** the homography and ROI are valid **only** at the exact view joints they were calibrated at. `calibration_points.json` → `home_view_joints` must equal the YAML `cv_view_P1x` joints within 1e-3 rad. If they differ, recalibrate with `dual_conveyor_pnp/04_workspace_calibration.py`; never "fix" the numbers by hand.

---

## 4. Phase 2: Make `cv_robot_node.py` station-aware for R1

Goal: `python3 cv_robot_node.py 1` works and robots 2/3/5/6 behave exactly as before.

1. `CV_ROBOTS = (1, 2, 3, 5, 6)`.
2. Add `STATION_ROBOTS = {1: ("P11", "P12")}`. For robots in it, load `cv_assets/robot1/<station>/phase2_config.json` and `.../calibration/homography_matrix.npy` for every station into a dict.
3. Subscribe to `/robot_1/cv_station` (`std_msgs/String`, data `"P11"`, `"P12"` or `"NONE"`). Keep the latest value. While it is `"NONE"` or unset, publish nothing. The camera then looks somewhere else and the homography would be wrong.
4. In `process_frame`, use the active station's ROI and H. Add `"station": "<P11|P12>"` to the published JSON and keep every existing key.
5. Use timer period **0.25 s** for R1 (the dwell is 3 s, so 1 Hz is too coarse) and `CONFIDENCE = 0.75` for R1, the value validated in dual_conveyor_pnp. Leave other robots at 1.0 s / 0.50.

**Check:** set the station by hand and watch the detections.
```bash
ros2 topic pub --once /robot_1/cv_station std_msgs/String "{data: P11}"
```
```bash
ros2 topic echo /robot1/cv_detection
```
With the arm at `cv_view_P11` and a chip inside the ROI, `robot_x` / `robot_y` must match what `dual_conveyor_pnp/05_test_pixel_to_robot.py --robot 150 --conveyor C1` prints for the same pixel.

---

## 5. Phase 3: R1 pick path in `robot_controller.py` and `niryo_factory_node.py`

### robot_controller.py
1. `self.cv_enabled = self.robot_id in (1, 2, 3, 5, 6)`.
2. For robots with stations (R1), load `pick_reference.json` and `runtime_config.json` **per station** from `cv_assets/robot1/<station>/`. Pick them using the `station` field in the CV command envelope. Reject a CV command for R1 without `station`.
3. Infer the station from the route's pickup key, then cross-check it against the envelope: `PICKUP - P11` → `P11`.
4. R1 CV sequence (others unchanged):
   1. `move_to_pose(hover)` → `move_to_pose(contact)` → `grip()` → `time.sleep(0.75)`
   2. `move_to_pose(hover)` ← **retreat to hover; do not use `hold_joints`**
   3. `move_to_position('home_joints')`
   4. `move_to_position(drop_pos)` (from the route) → `release()` → `time.sleep(0.5)` → `tool_neutral()`
   5. `move_to_position('home_joints')` → `move_to_position(f'cv_view_{station}')` → `IDLE`
5. The existing `SEQUENCE` command must accept a single position. `"R1 | SEQUENCE | cv_view_P11"` is used to move to a station view. After that move succeeds, publish `/robot_1/cv_station` = `"P11"`. Publish `"NONE"` when a non-view move starts.
6. Before sending each pose, keep the reach check: `0.10 m ≤ hypot(x, y) ≤ 0.42 m`. Otherwise set `STATE_ERROR` and do not move. This is the same rule as `dual_conveyor_pnp/core/picker.py`.

### niryo_factory_node.py
1. `self.cv_robot_ids = (1, 2, 3, 5, 6)`.
2. For R1 only, before the existing CV wait:
   1. Read the pickup key from the route (`P11` / `P12`).
   2. Publish `"R1 | SEQUENCE | cv_view_P11"` on `/robot_1/command` and wait for `/robot_1/status` == `IDLE`.
   3. Then start the existing wait loop. Accept only detections with `station` == `P11`.
3. For R1, `cv_ready_delay = 3.0` (the validated value). Keep `cv_visible_max_age = 0.75` and `cv_wait_timeout = 30.0`.
4. Add `"station": "P11"` to the envelope sent to the robot.

**Check:** first with the belts stopped (bench check only) and a chip placed in the C1 ROI, then with C1 running, inject a route as superfactory would:
```bash
ros2 topic pub --once /super_factory/trajectory std_msgs/String "{data: 'R1 | PICKUP - P11 | DROP - D15'}"
```
Expected order on `/robot_1/status`: moving to `cv_view_P11` → `IDLE` → (3 s) → pick states → `IDLE` at `cv_view_P11`. Then `/niryo_factory/status` shows `COMMAND_COMPLETE:R1`. Repeat with `P12` / `D13`.

---

## 6. Phase 4: Overhead alert node and conveyor flow

### New file: `robo_confirm/overhead/overhead_alert_node.py`
Port `dynamic_pnp/dual_conveyor_pnp/21_run_overhead_monitor.py` to rclpy:
- **Copy, don't rewrite:**
  - `common/apriltag_rois.py`, which has session tag lock, camera ownership, `ZoneGate`, `roi_mapping` (`similarity` for C2) and `tag_quarter_turns`.
  - `config/overhead_rois.json`.
  - The overhead model `~/multi_robocv/chip_training/models/best_chip_v13.pt`.
  - Thresholds from `config/system.json` → `overhead`.
- Keep the camera loop and YOLO the same. Run them in a worker thread or a timer, and keep `cv2.imshow` only behind a `--display` flag.
- Replace `bus.send(...)` with publishers:

| Topic | Type | Payload |
|---|---|---|
| `/overhead/alert` | `std_msgs/String` | the same JSON as today: `event` `CHIP_IN_ALERT_ZONE`, `id`, `conveyor`, `tag_id`, `robot`, `class_name`, `confidence`, `camera_index`, `tag_position_in` |
| `/overhead/entry` | `std_msgs/String` | `PART_ENTERED_CONVEYOR` (info only) |
| `/overhead/zone_state` | `std_msgs/String` | `{"zones": {"C1": {"on_conveyor", "in_alert", "observable"}, ..., "C8": {...}}}` at 2 Hz, all eight conveyors (`in_alert` stays 0 for C3-C8) |
| `/overhead/pickup_ready` | `std_msgs/String` | on each alert: `"CHIP_READY \| R1 \| P11 \| C1"` (C1) or `"CHIP_READY \| R1 \| P12 \| C2"` (C2) |

- The monitor watches all tags 0-7 (C1-C8) and reports positions / entries for every conveyor, but **only C1 and C2 raise alerts or `CHIP_READY`**: they are the only conveyors owned by one robot. In `config/system.json`, C3-C8 have `"robot": null`. Keep that rule in the node.
- Conveyor → pickup key map: `{"C1": ("R1", "P11"), "C2": ("R1", "P12")}`. Put it in the node's config, not in code.
- The `ZoneGate` re-arm (12 empty frames) already prevents duplicate `CHIP_READY` for one chip. Keep it.

### superfactory.py: accept the overhead trigger
- Add a subscription to `/overhead/pickup_ready` (`std_msgs/String`). In its callback, if the data starts with `CHIP_READY`, call the existing `_handle_ir_ready(msg.data)`. The parsing is identical (`parts[2]` = pickup key). You may rename the handler `_handle_pickup_ready` and keep `_handle_ir_ready` as an alias.
- Change nothing else in the gate. R1 stays out of `cv_robot_ids`.

### conveyor_control.py / conveyor_controllers.py: C1/C2 always on, no IR
- In `niryo_vendor/conveyors_config.yaml`, add to `conveyor_1` and `conveyor_2`:
  ```yaml
    ir_sensor_enabled: false   # overhead ALERT ROI is the trigger
    always_on: true            # start at bring-up, never stopped by the pick flow
  ```
- In `conveyor_control.py` (and the older `conveyor_controllers.py`), when `ir_sensor_enabled` is false:
  - do not poll the IR pin
  - never stop the belt on an IR edge
  - never publish `/niryo_factory/conveyor_{id}/ir_trigger`
  
  That means `conveyor_factory_node.py` never produces `IR_READY` for C1/C2.
- When `always_on` is true, send the belt the run command (`ControlConveyor` with `control_on: true`, configured `speed`) once at node startup. Keep it running and re-send it after any service error.
  - Still honour an explicit `STOP` on `/niryo_factory/conveyor_{id}/control_cmd`, for manual or safety stops.
  - Stop on node shutdown.
- Leave every other conveyor's IR behavior unchanged.

**Check, without robots:**
```bash
ros2 topic echo /conveyor_factory/status
```
```bash
ros2 topic echo /overhead/pickup_ready
```
- With `conveyor_control.py 1` and `2` running, both belts must be moving and stay moving.
- Block the IR sensors on C1/C2 by hand. Nothing must happen: no belt stop, no `ir_trigger`, no `IR_READY`.
- Put a chip in the C1 ALERT zone. Exactly one `CHIP_READY | R1 | P11 | C1` must appear on `/overhead/pickup_ready`, plus one `/overhead/alert`, and **no** `IR_READY` on `/conveyor_factory/status`.

---

## 7. Phase 5: End-to-end test

Start in this order, one terminal each, with ROS2 sourced:
1. `robot_controller.py 1`
2. `conveyor_control.py 1` (C1 starts and keeps running)
3. `conveyor_control.py 2` (C2 starts and keeps running)
4. `cv_robot_node.py 1`
5. `niryo_factory_node.py`
6. `algorithm_node.py` + `superfactory.py` (as in robo_confirm README)
7. `overhead/overhead_alert_node.py`

Pass criteria:
1. Both belts run for the whole test and are never stopped. A chip on C1 reaches the ALERT zone, `CHIP_READY | R1 | P11 | C1` is published, and superfactory releases the R1 route. No IR event appears anywhere.
2. R1 moves to `cv_view_P11`. Within about 3 s of the chip being visible it picks the moving chip with the belt still running, then retreats to hover, goes home, drops at the routed position, goes home, and returns to `cv_view_P11`.
3. `COMMAND_COMPLETE:R1` is published.
4. The same works for C2 → `cv_view_P12`.
5. Robots 2, 3, 5 and 6 are unchanged: run one of their existing CV routes.

---

## 8. Pitfalls

- **Two namespaces:** the driver uses `/robot1/...` and the testbed uses `/robot_1/...`. Use each where the existing code does.
- **Do not run dual_conveyor_pnp scripts while `robot_controller.py 1` is running.** pyniryo and the ROS2 driver would both command the arm.
- **View pose = calibration:** any change to `cv_view_P11` / `cv_view_P12` invalidates that station's homography and ROI. Recalibrate with `dual_conveyor_pnp` 04 → 08, then copy again (Phase 1).
- **Drops come from routing.** Never hard-code D13/D15 per conveyor in ROS2 code.
- **The overhead camera owns its USB device.** Close `test_all_apriltag_rois.py` and `21_run_overhead_monitor.py` before starting the node.
- **C2's overhead ROI needs `roi_mapping: similarity`.** With `homography` it widens far from tag 1 (corner noise amplified by perspective). Copy `overhead_rois.json` as-is.
- **superfactory's CV set stays `{'R2','R3','R5','R6'}`.** Adding R1 would skip the pickup gate, and R1 would then be sent picks without any overhead alert.
- **No IR on C1/C2, and the belts never stop for a pick.** If a belt stops during a pick or an `IR_READY` for P11/P12 appears, Phase 4 is wrong. Don't "fix" it by adding a stop/start around the pick.
- **ACL:** new nodes must not publish to `/niryo_factory/command`. The overhead node only publishes on `/overhead/*`, so it needs no ACL change.
- **Keep the old behavior reachable:** add R1 changes behind `if robot_id == 1` or the station dict, and never change the code path for 2/3/5/6.

## 9. Reference: validated parameters (2026-10-15)

| Parameter | Value | Source |
|---|---|---|
| Wrist YOLO confidence | 0.75 | `config/system.json` `robot_runtime.wrist_confidence` |
| FIFO dwell before pick | 3.0 s | `robot_runtime.dwell_sec` |
| Station empty timeout | 3.0 s | `robot_runtime.empty_timeout_sec` |
| Grip settle / release settle | 0.75 s / 0.5 s | `core/picker.py` |
| Reach limits | 0.10–0.42 m | `robot_runtime.min_reach_m` / `max_reach_m` |
| Conveyor speed | 50 | `robot_runtime.conveyor_speed` |
| Overhead confidence / alert frames / re-arm frames | 0.75 / 4 / 12 | `config/system.json` `overhead` |
| Tag size | 3.0 in, Tag36h11 | `config/overhead_rois.json` |
