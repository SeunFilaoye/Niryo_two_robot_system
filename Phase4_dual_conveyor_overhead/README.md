# Phase 4 - Dual-conveyor dynamic pick-and-place with overhead webcam supervision

One Niryo Ned2 (robot .150, `192.168.0.150`) serves two conveyors (C1, C2).
Overhead USB webcams find AprilTags (Tag36h11, 3 in) on the conveyors, project
conveyor / entry / ALERT regions from each tag, and run the YOLO chip model.
When a chip sits in a conveyor's ALERT region, the supervisor sends the robot
to that conveyor; the robot's wrist camera finds the chip and picks it.

```
Phase4_dual_conveyor_overhead/
├── models/
│   ├── best.pt              wrist-camera chip model (6 classes)
│   └── best_chip_v13.pt     overhead-camera chip model (12 classes)
└── dual_conveyor_pnp/
    ├── README.md            full system guide and run order
    ├── ROS2_INTEGRATION.md  step-by-step ROS2 integration plan
    ├── config/              system.json (robots, conveyors, tags, ports), overhead_rois.json
    ├── robots/robot_150/    poses, wrist ROIs, calibrations for C1 and C2
    ├── 01-18                robot setup / calibration / FIFO scripts
    ├── 20_draw_alert_rois.py      overhead ROI editor
    ├── 21_run_overhead_monitor.py overhead AprilTag + chip monitor
    └── 22-24                supervisor, robot worker, bus monitor
```

`config/system.json` points at `../models/`, so keep `models/` beside
`dual_conveyor_pnp/`. Run scripts from inside `dual_conveyor_pnp/`:

```bash
cd dual_conveyor_pnp
python 21_run_overhead_monitor.py      # overhead cameras
python 22_run_supervisor.py            # supervisor
python 23_run_robot_worker.py --robot 150
```
