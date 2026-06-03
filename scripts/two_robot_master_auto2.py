import cv2
import json
import time
import numpy as np
from pathlib import Path
from pyniryo import NiryoRobot, ConveyorDirection
from ultralytics import YOLO
 
# =========================================================
# SETTINGS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
 
R1_IP = "192.168.0.229"
R2_IP = "192.168.0.201"
 
R1_CALIB_DIR = PROJECT_ROOT / "calibration_r1"
R2_CALIB_DIR = PROJECT_ROOT / "calibration_r2"
 
CONFIDENCE = 0.35
 
# =========================================================
# ROBOT 1 POSES
# =========================================================
R1_OBSERVATION_JOINTS = (0.5243, -0.2777, -0.0568, -0.0643, -1.3944, -0.9509)
R1_HOME_SAFE_JOINTS = (0.5243, 0.3282, -1.34, -0.0643, -0.6888, -0.9494)
R1_SAFE_TRANSFER_JOINTS = (1.824, 0.1252, -0.4052, -0.075, -1.2088, -0.954)
R1_CONVEYOR2_HOVER_JOINTS = (2.8635, 0.0252, -0.6416, -0.0413, -1.0447, -0.9509)
R1_CONVEYOR2_DROP_JOINTS = (2.9, -0.4247, -0.6537, -0.072, -0.6888, -0.9402)
 
# =========================================================
# ROBOT 2 POSES
# =========================================================
R2_OBSERVATION_JOINTS    = (-0.0372,  0.5494, -0.6476, -0.0045, -1.7166,  0.0185)
R2_HOME_SAFE_JOINTS      = ( 0.0753,  0.4009, -1.3021,  0.0108, -0.3529,  0.0154)
 
# Safe midpoint between both bins — arm passes through here in both directions
R2_BIN_MIDPOINT_JOINTS   = ( 1.2792,  0.1146, -0.4264, -0.0198, -1.7181,  0.0154)
 
# Red bin — hover above then drop
R2_RED_BIN_HOVER_JOINTS  = ( 0.4771, -0.5337, -0.0235, -0.0474, -0.9711,  0.0277)
R2_RED_BIN_DROP_JOINTS   = ( 0.4360, -0.8064,  0.0143, -0.0152, -0.9711,  0.0277)
 
# Green bin — hover above then drop
R2_GREEN_BIN_HOVER_JOINTS = ( 2.2121,  0.1282, -0.6764, -0.0075, -1.0570,  0.0277)
R2_GREEN_BIN_DROP_JOINTS  = ( 2.2562, -0.4822, -0.6870,  0.1473, -0.4817,  0.0261)
 
# =========================================================
# ROI
# =========================================================
R1_ROI = (180, 100, 500, 380)
R2_ROI = (180, 100, 500, 380)
 
# =========================================================
# TIMING / PICK HEIGHTS
# =========================================================
R1_SETTLE_TIME_SEC = 0.45
R2_SETTLE_TIME_SEC = 0.45
R1_FINAL_MATCH_MAX_DIST_PX = 140
R2_FINAL_MATCH_MAX_DIST_PX = 140
 
R1_APPROACH_HOVER_OFFSET_M = 0.020
R1_DESCEND_FROM_HOVER_M = 0.05
R2_APPROACH_HOVER_OFFSET_M = 0.020
R2_DESCEND_FROM_HOVER_M = 0.065
 
PAUSE_AT_PICK_SEC = 0.5
GRIPPER_WAIT_SEC = 0.8
RELEASE_WAIT_SEC = 0.6
TRANSFER_SETTLE_ON_CONVEYOR2_SEC = 1.0
 
CONVEYOR_SPEED_R1 = 40
CONVEYOR_SPEED_R2 = 50
CONVEYOR_DIR = ConveyorDirection.FORWARD
 
R1_VALID_CLASSES = {"red_square", "red_circle", "green_square", "green_circle"}
R2_VALID_CLASSES = {"red_square", "red_circle", "green_square", "green_circle"}
 
CLASS_COLORS = {
    "red_square":   (60,  60,  220),
    "red_circle":   (40,  40,  180),
    "blue_square":  (200, 150,  20),
    "blue_circle":  (180, 100,  10),
    "green_square": ( 40, 200,  80),
    "green_circle": ( 20, 160,  50),
}
 
# =========================================================
# SHARED STATE  (written by logic, read by renderer)
# =========================================================
state = {
    # robot statuses
    "r1_status": "IDLE",        # IDLE | DETECTING | PICKING | TRANSFERRING
    "r2_status": "IDLE",        # IDLE | DETECTING | PICKING | SORTING
    "r1_chip":   None,          # dict with class_name, cx, cy, robot_x_m, robot_y_m, conf
    "r2_chip":   None,
    # conveyor statuses
    "conv1_running": False,
    "conv1_speed":   CONVEYOR_SPEED_R1,
    "conv1_dir":     "FWD",
    "conv2_running": False,
    "conv2_speed":   CONVEYOR_SPEED_R2,
    "conv2_dir":     "FWD",
    # counters
    "r1_picks":  0,
    "r2_sorts":  0,
    # cycle timing
    "cycle_start": None,
    "last_cycle_ms": None,
    "start_time": time.time(),
}
 
# =========================================================
# COLOUR PALETTE  (BGR)
# =========================================================
C = {
    "bg":          ( 14,  17,  22),
    "panel":       ( 22,  27,  35),
    "panel_light": ( 30,  37,  48),
    "accent":      ( 20, 200, 255),   # amber-ish yellow
    "accent2":     ( 60, 255, 160),   # mint green
    "danger":      ( 40,  50, 230),   # red in BGR
    "warn":        ( 20, 160, 250),   # orange
    "ok":          ( 60, 220,  80),   # green
    "dim":         ( 80,  90, 105),
    "white":       (230, 235, 240),
    "chip_red":    ( 60,  70, 220),
    "chip_green":  ( 50, 200,  70),
    "border":      ( 40,  50,  65),
}
 
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO   = cv2.FONT_HERSHEY_PLAIN   # slightly different feel for data
FONT_BOLD   = cv2.FONT_HERSHEY_DUPLEX
 
# =========================================================
# DRAWING PRIMITIVES
# =========================================================
def filled_rect(img, x, y, w, h, color, alpha=1.0, radius=4):
    """Draw a filled rounded-ish rectangle (OpenCV has no native corner radius,
    so we fake it with overlapping rects + circles at corners)."""
    if alpha < 1.0:
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
 
 
def outlined_rect(img, x, y, w, h, color, thickness=1):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
 
 
def corner_marks(img, x, y, w, h, color, size=12, thickness=2):
    """Draw corner bracket marks (industrial HUD style)."""
    pts = [
        ((x, y+size), (x, y), (x+size, y)),
        ((x+w-size, y), (x+w, y), (x+w, y+size)),
        ((x+w, y+h-size), (x+w, y+h), (x+w-size, y+h)),
        ((x+size, y+h), (x, y+h), (x, y+h-size)),
    ]
    for p1, corner, p2 in pts:
        cv2.line(img, p1, corner, color, thickness)
        cv2.line(img, corner, p2, color, thickness)
 
 
def label(img, text, x, y, scale=0.45, color=None, font=FONT, thickness=1):
    color = color or C["dim"]
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
 
 
def value(img, text, x, y, scale=0.55, color=None, thickness=1):
    color = color or C["white"]
    cv2.putText(img, text, (x, y), FONT_BOLD, scale, color, thickness, cv2.LINE_AA)
 
 
def tag(img, text, x, y, bg_color, fg_color=None, pad_x=6, pad_y=3, scale=0.38):
    """Pill-style status tag."""
    fg_color = fg_color or C["bg"]
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    w = tw + pad_x * 2
    h = th + pad_y * 2
    filled_rect(img, x, y - h + pad_y, w, h, bg_color)
    cv2.putText(img, text, (x + pad_x, y + pad_y - 2), FONT, scale, fg_color, 1, cv2.LINE_AA)
    return w
 
 
def hline(img, x, y, w, color=None, thickness=1):
    color = color or C["border"]
    cv2.line(img, (x, y), (x + w, y), color, thickness)
 
 
def vline(img, x, y, h, color=None, thickness=1):
    color = color or C["border"]
    cv2.line(img, (x, y), (x, y + h), color, thickness)
 
 
def dot_indicator(img, x, y, on, r=5):
    color = C["ok"] if on else C["dim"]
    cv2.circle(img, (x, y), r, color, -1)
    if on:
        cv2.circle(img, (x, y), r + 2, (*color[:2], max(0, color[2] - 60)), 1)
 
 
def progress_bar(img, x, y, w, h, value_pct, fg=None, bg=None):
    fg = fg or C["accent"]
    bg = bg or C["panel_light"]
    filled_rect(img, x, y, w, h, bg)
    filled_rect(img, x, y, int(w * value_pct), h, fg)
    outlined_rect(img, x, y, w, h, C["border"])
 
 
def chip_badge(img, x, y, class_name):
    """Draw a compact chip identity badge."""
    if class_name is None:
        label(img, "NO CHIP", x, y + 12, 0.38, C["dim"])
        return
 
    is_red   = "red"   in class_name
    is_green = "green" in class_name
    is_sq    = "square" in class_name
 
    color = C["chip_red"] if is_red else C["chip_green"]
    shape_char = "■" if is_sq else "●"
 
    # colour swatch
    filled_rect(img, x, y, 14, 14, color)
    outlined_rect(img, x, y, 14, 14, C["border"])
 
    short = class_name.upper().replace("_", " ")
    cv2.putText(img, short, (x + 20, y + 11), FONT_BOLD, 0.42, color, 1, cv2.LINE_AA)
 
 
# =========================================================
# ROBOT STATUS COLOUR
# =========================================================
STATUS_COLOR = {
    "IDLE":         C["dim"],
    "DETECTING":    C["warn"],
    "PICKING":      C["accent"],
    "TRANSFERRING": C["accent2"],
    "SORTING":      C["accent2"],
    "COMPLETE":     C["ok"],
    "ERROR":        C["danger"],
}
 
def status_color(s):
    return STATUS_COLOR.get(s, C["dim"])
 
 
# =========================================================
# SIDE PANEL RENDERER  (returns a tall narrow image)
# =========================================================
PANEL_W  = 280
PANEL_H  = 600   # will be stretched to match camera height
 
def render_side_panel(height):
    panel = np.full((height, PANEL_W, 3), C["bg"], dtype=np.uint8)
 
    # --- vertical accent strip on left edge
    cv2.rectangle(panel, (0, 0), (2, height), C["accent"], -1)
 
    # ---- HEADER ----
    filled_rect(panel, 3, 0, PANEL_W - 3, 52, C["panel"])
    cv2.putText(panel, "NIRYO",   (14, 22), FONT_BOLD, 0.65, C["accent"],  1, cv2.LINE_AA)
    cv2.putText(panel, "CONTROL", (14, 40), FONT_BOLD, 0.45, C["white"],   1, cv2.LINE_AA)
 
    # uptime
    elapsed = int(time.time() - state["start_time"])
    h_e = elapsed // 3600
    m_e = (elapsed % 3600) // 60
    s_e = elapsed % 60
    uptime_str = f"UP  {h_e:02d}:{m_e:02d}:{s_e:02d}"
    label(panel, uptime_str, PANEL_W - 110, 38, 0.36, C["dim"])
 
    hline(panel, 3, 53, PANEL_W - 6, C["border"])
 
    y = 68
 
    # ==== ROBOT SECTIONS ====
    for idx, (rname, rkey_status, rkey_chip) in enumerate([
        ("ROBOT 1", "r1_status", "r1_chip"),
        ("ROBOT 2", "r2_status", "r2_chip"),
    ]):
        status   = state[rkey_status]
        chip     = state[rkey_chip]
        picks    = state["r1_picks"] if idx == 0 else state["r2_sorts"]
        stat_col = status_color(status)
 
        # robot card bg
        filled_rect(panel, 8, y, PANEL_W - 16, 148, C["panel"])
        outlined_rect(panel, 8, y, PANEL_W - 16, 148, C["border"])
        corner_marks(panel, 8, y, PANEL_W - 16, 148, stat_col, size=9, thickness=1)
 
        # robot name + dot
        dot_indicator(panel, 22, y + 13, status != "IDLE")
        cv2.putText(panel, rname, (33, y + 17), FONT_BOLD, 0.50, C["white"], 1, cv2.LINE_AA)
 
        # status tag  (top-right of card)
        tag_w = tag(panel, status, PANEL_W - 16 - 8 - len(status)*7 - 4, y + 6,
                    stat_col, C["bg"], scale=0.35)
 
        hline(panel, 12, y + 24, PANEL_W - 24, C["border"])
 
        # chip identity
        label(panel, "CHIP DETECTED", 16, y + 39, 0.34, C["dim"])
        chip_badge(panel, 16, y + 43, chip["class_name"] if chip else None)
 
        # coordinates
        label(panel, "POSITION (robot frame)", 16, y + 74, 0.34, C["dim"])
        if chip and "robot_x_m" in chip:
            rx = chip["robot_x_m"]
            ry = chip["robot_y_m"]
            cv2.putText(panel, f"X  {rx:+.4f} m", (16, y + 89), FONT_MONO, 0.78, C["accent"],  1, cv2.LINE_AA)
            cv2.putText(panel, f"Y  {ry:+.4f} m", (16, y + 104), FONT_MONO, 0.78, C["accent2"], 1, cv2.LINE_AA)
        else:
            label(panel, "X  ---", 16, y + 89,  0.44, C["dim"])
            label(panel, "Y  ---", 16, y + 104, 0.44, C["dim"])
 
        # pixel coords
        if chip:
            px_str = f"px ({chip['cx']}, {chip['cy']})"
            label(panel, px_str, 16, y + 118, 0.36, C["dim"])
        
        # pick counter
        counter_label = "PICKS" if idx == 0 else "SORTS"
        label(panel, counter_label, PANEL_W - 70, y + 89, 0.34, C["dim"])
        cv2.putText(panel, str(picks), (PANEL_W - 70, y + 110), FONT_BOLD, 0.70, C["white"], 1, cv2.LINE_AA)
 
        hline(panel, 12, y + 126, PANEL_W - 24, C["border"])
 
        # last cycle time
        if state["last_cycle_ms"] and idx == 1:
            label(panel, f"LAST CYCLE  {state['last_cycle_ms']}ms", 16, y + 140, 0.35, C["dim"])
        else:
            label(panel, "AWAITING CYCLE", 16, y + 140, 0.35, C["dim"])
 
        y += 160
 
    # ==== CONVEYOR SECTION ====
    hline(panel, 3, y - 4, PANEL_W - 6, C["border"])
 
    cv2.putText(panel, "CONVEYORS", (12, y + 14), FONT_BOLD, 0.45, C["white"], 1, cv2.LINE_AA)
    y += 22
 
    for cidx, (cname, run_key, spd_key, dir_key) in enumerate([
        ("CONV 1  →  ROBOT 1", "conv1_running", "conv1_speed", "conv1_dir"),
        ("CONV 2  →  ROBOT 2", "conv2_running", "conv2_speed", "conv2_dir"),
    ]):
        running = state[run_key]
        speed   = state[spd_key]
        dirstr  = state[dir_key]
 
        filled_rect(panel, 8, y, PANEL_W - 16, 62, C["panel_light"] if running else C["panel"])
        outlined_rect(panel, 8, y, PANEL_W - 16, 62, C["border"])
 
        # animated "belt teeth" if running
        if running:
            belt_y = y + 30
            tick = int(time.time() * 8) % 16
            for bx in range(16, PANEL_W - 24, 16):
                ox = (bx + tick) % (PANEL_W - 32)
                cv2.rectangle(panel, (8 + ox, belt_y), (8 + ox + 6, belt_y + 4), C["border"], -1)
 
        dot_indicator(panel, 20, y + 12, running)
        label(panel, cname, 32, y + 16, 0.36, C["white"] if running else C["dim"])
 
        # speed bar
        label(panel, "SPEED", 16, y + 34, 0.33, C["dim"])
        progress_bar(panel, 16, y + 38, 100, 7, speed / 100.0,
                     fg=C["accent"] if running else C["dim"])
        label(panel, f"{speed}%", 122, y + 46, 0.36, C["accent"] if running else C["dim"])
 
        # direction
        dir_col = C["accent2"] if running else C["dim"]
        label(panel, f"DIR  {dirstr}", PANEL_W - 85, y + 46, 0.36, dir_col)
 
        # status tag
        tag(panel, "RUNNING" if running else "STOPPED", PANEL_W - 90, y + 16,
            C["ok"] if running else C["dim"], C["bg"], scale=0.33)
 
        y += 70
 
    # ==== FOOTER ====
    hline(panel, 3, height - 28, PANEL_W - 6, C["border"])
    label(panel, "Q  QUIT", 14, height - 12, 0.36, C["dim"])
    label(panel, "v1.0  |  AUTO MODE", PANEL_W - 115, height - 12, 0.33, C["dim"])
 
    return panel
 
 
# =========================================================
# CAMERA FRAME OVERLAY
# =========================================================
def draw_camera_overlay(frame, detections, live_target, roi, robot_label, status):
    out = frame.copy()
    stat_col = status_color(status)
 
    # dark vignette border
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), C["bg"], 20)
    cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)
 
    # ROI bracket
    rx1, ry1, rx2, ry2 = roi
    corner_marks(out, rx1, ry1, rx2 - rx1, ry2 - ry1, stat_col, size=14, thickness=2)
    label(out, "WORKSPACE ROI", rx1 + 4, ry1 - 6, 0.38, stat_col)
 
    # detections
    for det in detections:
        class_name, x1, y1, x2, y2, cx, cy = det
        color = CLASS_COLORS.get(class_name, (200, 200, 200))
 
        # bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
 
        # crosshair at centroid
        csize = 8
        cv2.line(out, (cx - csize, cy), (cx + csize, cy), color, 1)
        cv2.line(out, (cx, cy - csize), (cx, cy + csize), color, 1)
        cv2.circle(out, (cx, cy), 3, color, -1)
 
        # class label bubble
        lbl = class_name.upper().replace("_", " ")
        (tw, th), _ = cv2.getTextSize(lbl, FONT, 0.38, 1)
        filled_rect(out, x1, y1 - th - 8, tw + 8, th + 6, color, 0.85)
        cv2.putText(out, lbl, (x1 + 4, y1 - 4), FONT, 0.38, C["bg"], 1, cv2.LINE_AA)
 
    # active target highlight
    if live_target is not None:
        tx, ty = live_target["cx"], live_target["cy"]
        cv2.circle(out, (tx, ty), 18, stat_col, 2)
        cv2.circle(out, (tx, ty), 5,  stat_col, -1)
 
        # coord readout near the target
        coord_txt = f"X{live_target['robot_x_m']:+.3f}  Y{live_target['robot_y_m']:+.3f}"
        cx_off = tx + 24
        cy_off = ty - 8
        (ctw, cth), _ = cv2.getTextSize(coord_txt, FONT, 0.38, 1)
        filled_rect(out, cx_off - 3, cy_off - cth - 3, ctw + 8, cth + 6, C["bg"], 0.75)
        cv2.putText(out, coord_txt, (cx_off + 1, cy_off), FONT, 0.38, stat_col, 1, cv2.LINE_AA)
 
    # top-left robot label bar
    bar_h = 28
    filled_rect(out, 0, 0, w, bar_h, C["bg"], 0.75)
    cv2.rectangle(out, (0, 0), (4, bar_h), stat_col, -1)
    cv2.putText(out, robot_label, (10, 19), FONT_BOLD, 0.52, C["white"], 1, cv2.LINE_AA)
    tag(out, status, w - len(status) * 8 - 20, 5, stat_col, C["bg"], scale=0.38)
 
    # bottom coord strip
    filled_rect(out, 0, h - 22, w, 22, C["bg"], 0.75)
    if live_target:
        pxstr = f"PIX  ({live_target['cx']}, {live_target['cy']})    ROBOT  X{live_target['robot_x_m']:+.4f}m  Y{live_target['robot_y_m']:+.4f}m"
    else:
        pxstr = "NO TARGET IN ROI"
    cv2.putText(out, pxstr, (8, h - 7), FONT, 0.36, C["dim"], 1, cv2.LINE_AA)
 
    return out
 
 
# =========================================================
# COMPOSE FULL WINDOW
# =========================================================
CAM_DISPLAY_W = 520
CAM_DISPLAY_H = 390
 
def compose_display(r1_frame, r1_detections, r1_live,
                    r2_frame, r2_detections, r2_live):
    def prep_cam(frame, detections, live, robot_label, status):
        if frame is None:
            blank = np.full((CAM_DISPLAY_H, CAM_DISPLAY_W, 3), C["bg"], dtype=np.uint8)
            cv2.putText(blank, "NO SIGNAL", (CAM_DISPLAY_W // 2 - 50, CAM_DISPLAY_H // 2),
                        FONT_BOLD, 0.65, C["danger"], 1, cv2.LINE_AA)
            return blank
        overlay = draw_camera_overlay(frame, detections, live, R1_ROI if "1" in robot_label else R2_ROI,
                                       robot_label, status)
        return cv2.resize(overlay, (CAM_DISPLAY_W, CAM_DISPLAY_H))
 
    cam1 = prep_cam(r1_frame, r1_detections, r1_live, "ROBOT 1", state["r1_status"])
    cam2 = prep_cam(r2_frame, r2_detections, r2_live, "ROBOT 2", state["r2_status"])
 
    total_h = CAM_DISPLAY_H * 2 + 4          # 2-pixel divider
    side    = render_side_panel(total_h)
 
    # stack camera views vertically, thin divider between them
    divider = np.full((4, CAM_DISPLAY_W, 3), C["border"], dtype=np.uint8)
    cams    = np.vstack([cam1, divider, cam2])
 
    # horizontal concat: cameras | side panel
    sep = np.full((total_h, 2, 3), C["accent"], dtype=np.uint8)
    window = np.hstack([cams, sep, side])
 
    return window
 
 
# =========================================================
# HELPERS  (unchanged logic from original)
# =========================================================
def decode_robot_image(img_compressed):
    arr = np.frombuffer(img_compressed, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
 
 
def load_calibration(calib_dir):
    with open(calib_dir / "workspace_calibration.json", "r") as f:
        cam = json.load(f)
    with open(calib_dir / "robot_workspace_points.json", "r") as f:
        pts = json.load(f)["points"]
    with open(calib_dir / "workspace_hover_reference.json", "r") as f:
        hover = json.load(f)["hover_reference"]
 
    image_points = np.array([
        cam["image_points"]["top_left"],
        cam["image_points"]["top_right"],
        cam["image_points"]["bottom_right"],
        cam["image_points"]["bottom_left"],
    ], dtype=np.float32)
 
    world_points = np.array([
        [0.0, 0.0],
        [float(cam["workspace_width_mm"]), 0.0],
        [float(cam["workspace_width_mm"]), float(cam["workspace_height_mm"])],
        [0.0, float(cam["workspace_height_mm"])],
    ], dtype=np.float32)
 
    H_img_to_ws  = cv2.getPerspectiveTransform(image_points, world_points)
 
    robot_xy = np.array([
        [pts["top_left"]["x_m"],     pts["top_left"]["y_m"]],
        [pts["top_right"]["x_m"],    pts["top_right"]["y_m"]],
        [pts["bottom_right"]["x_m"], pts["bottom_right"]["y_m"]],
        [pts["bottom_left"]["x_m"],  pts["bottom_left"]["y_m"]],
    ], dtype=np.float32)
 
    H_ws_to_robot = cv2.getPerspectiveTransform(world_points, robot_xy)
 
    return {
        "workspace_width_mm":  float(cam["workspace_width_mm"]),
        "workspace_height_mm": float(cam["workspace_height_mm"]),
        "image_points":        image_points,
        "H_img_to_ws":         H_img_to_ws,
        "H_ws_to_robot":       H_ws_to_robot,
        "hover_reference":     hover,
    }
 
 
def pixel_to_workspace(cx, cy, H_img_to_ws):
    src = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H_img_to_ws)
    return float(dst[0][0][0]), float(dst[0][0][1])
 
 
def workspace_to_robot(x_mm, y_mm, H_ws_to_robot):
    src = np.array([[[float(x_mm), float(y_mm)]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H_ws_to_robot)
    return float(dst[0][0][0]), float(dst[0][0][1])
 
 
def extract_roi_detections(results, model, roi, valid_classes):
    x1r, y1r, x2r, y2r = roi
    out = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id     = int(box.cls[0].item())
        class_name = model.names[cls_id]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if class_name in valid_classes and x1r <= cx <= x2r and y1r <= cy <= y2r:
            out.append((class_name, x1, y1, x2, y2, cx, cy))
    return out
 
 
def choose_target_in_roi(detections, calib, hover_ref, roi, approach_offset_m, descend_from_hover_m):
    if not detections:
        return None
 
    roi_cx = (roi[0] + roi[2]) // 2
    roi_cy = (roi[1] + roi[3]) // 2
    candidates = []
 
    for det in detections:
        class_name, x1, y1, x2, y2, cx, cy = det
        x_mm, y_mm       = pixel_to_workspace(cx, cy, calib["H_img_to_ws"])
        robot_x_m, robot_y_m = workspace_to_robot(x_mm, y_mm, calib["H_ws_to_robot"])
 
        base_hover_z = float(hover_ref["z_m"])
        approach_z   = base_hover_z + approach_offset_m
        pick_z       = base_hover_z - descend_from_hover_m
 
        target = {
            "class_name": class_name,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": cx, "cy": cy,
            "x_mm": x_mm, "y_mm": y_mm,
            "robot_x_m":  robot_x_m,
            "robot_y_m":  robot_y_m,
            "approach_z_m": approach_z,
            "robot_z_m":    base_hover_z,
            "pick_z_m":     pick_z,
            "roll_rad":  float(hover_ref["roll_rad"]),
            "pitch_rad": float(hover_ref["pitch_rad"]),
            "yaw_rad":   float(hover_ref["yaw_rad"]),
        }
        dist = ((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2) ** 0.5
        candidates.append((dist, target))
 
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]
 
 
def find_final_target_after_stop(robot, model, prev_target, calib, hover_ref, roi, valid_classes,
                                  settle_time_sec, final_match_max_dist_px,
                                  approach_offset_m, descend_from_hover_m):
    time.sleep(settle_time_sec)
    img_compressed = robot.get_img_compressed()
    if img_compressed is None:
        return None, None
 
    frame = decode_robot_image(img_compressed)
    if frame is None:
        return None, None
 
    results    = model(frame, conf=CONFIDENCE, verbose=False)[0]
    detections = extract_roi_detections(results, model, roi, valid_classes)
 
    candidates = []
    for det in detections:
        class_name, x1, y1, x2, y2, cx, cy = det
        if class_name != prev_target["class_name"]:
            continue
        dist = ((cx - prev_target["cx"]) ** 2 + (cy - prev_target["cy"]) ** 2) ** 0.5
        if dist <= final_match_max_dist_px:
            target = choose_target_in_roi(
                [(class_name, x1, y1, x2, y2, cx, cy)],
                calib, hover_ref, roi, approach_offset_m, descend_from_hover_m
            )
            if target is not None:
                candidates.append((dist, target))
 
    if not candidates:
        return frame, None
 
    candidates.sort(key=lambda item: item[0])
    return frame, candidates[0][1]
 
 
def move_above_target(robot, target):
    robot.move_pose(
        target["robot_x_m"], target["robot_y_m"], target["approach_z_m"],
        target["roll_rad"], target["pitch_rad"], target["yaw_rad"]
    )
 
def move_to_base_hover(robot, target):
    robot.move_pose(
        target["robot_x_m"], target["robot_y_m"], target["robot_z_m"],
        target["roll_rad"], target["pitch_rad"], target["yaw_rad"]
    )
 
def move_to_pick_height(robot, target):
    robot.move_pose(
        target["robot_x_m"], target["robot_y_m"], target["pick_z_m"],
        target["roll_rad"], target["pitch_rad"], target["yaw_rad"]
    )
 
def get_r2_bin_poses(class_name):
    """Return (hover_joints, drop_joints) for the correct bin based on chip colour."""
    if "green" in class_name:
        return R2_GREEN_BIN_HOVER_JOINTS, R2_GREEN_BIN_DROP_JOINTS
    return R2_RED_BIN_HOVER_JOINTS, R2_RED_BIN_DROP_JOINTS
 
 
# =========================================================
# LIVE DISPLAY REFRESH
# Shared references so action functions can push frames
# without needing to pass robots/model through every call.
# Populated in main() before any robot action runs.
# =========================================================
display_refs = {
    "r1": None,       # NiryoRobot instance
    "r2": None,
    "model": None,
    "r1_calib": None,
    "r1_hover": None,
    "r2_calib": None,
    "r2_hover": None,
    "last_r1_frame": None,
    "last_r1_det":   [],
    "last_r2_frame": None,
    "last_r2_det":   [],
}
 
def refresh_display():
    """Grab the latest frame from each robot, run YOLO, and repaint the window.
    Called between every blocking pose/gripper command so the feed stays live."""
    m = display_refs["model"]
 
    for rkey, roi, valid, calib_key, hover_key in [
        ("r1", R1_ROI, R1_VALID_CLASSES, "r1_calib", "r1_hover"),
        ("r2", R2_ROI, R2_VALID_CLASSES, "r2_calib", "r2_hover"),
    ]:
        robot = display_refs[rkey]
        if robot is None:
            continue
        try:
            raw = robot.get_img_compressed()
            if raw is None:
                continue
            frame = decode_robot_image(raw)
            if frame is None:
                continue
            results = m(frame, conf=CONFIDENCE, verbose=False)[0]
            det     = extract_roi_detections(results, m, roi, valid)
            display_refs[f"last_{rkey}_frame"] = frame
            display_refs[f"last_{rkey}_det"]   = det
        except Exception:
            pass  # never let a display refresh crash a robot move
 
    win = compose_display(
        display_refs["last_r1_frame"], display_refs["last_r1_det"], None,
        display_refs["last_r2_frame"], display_refs["last_r2_det"], None,
    )
    cv2.imshow("NIRYO CONTROL", win)
    cv2.waitKey(1)
 
 
# =========================================================
# ROBOT ACTIONS  (with state updates)
# =========================================================
def run_r1_pick_and_transfer(r1, target):
    state["r1_status"] = "PICKING"
    state["r1_chip"]   = target
 
    r1.update_tool()
    r1.release_with_tool()
    refresh_display()
 
    # descend to pick
    move_above_target(r1, target);   refresh_display()
    move_to_base_hover(r1, target);  refresh_display()
    move_to_pick_height(r1, target)
    time.sleep(PAUSE_AT_PICK_SEC);   refresh_display()
 
    r1.grasp_with_tool()
    time.sleep(GRIPPER_WAIT_SEC);    refresh_display()
 
    # lift and transfer
    state["r1_status"] = "TRANSFERRING"
    move_above_target(r1, target);               refresh_display()
    r1.move_joints(*R1_SAFE_TRANSFER_JOINTS);    refresh_display()
    r1.move_joints(*R1_CONVEYOR2_HOVER_JOINTS);  refresh_display()
    r1.move_joints(*R1_CONVEYOR2_DROP_JOINTS);   refresh_display()
 
    r1.release_with_tool()
    time.sleep(RELEASE_WAIT_SEC);                refresh_display()
 
    state["r1_picks"] += 1
 
    # return home
    r1.move_joints(*R1_CONVEYOR2_HOVER_JOINTS);  refresh_display()
    r1.move_joints(*R1_SAFE_TRANSFER_JOINTS);    refresh_display()
    r1.move_joints(*R1_OBSERVATION_JOINTS);      refresh_display()
 
    state["r1_status"] = "IDLE"
    state["r1_chip"]   = None
 
 
def run_r2_pick_and_sort(r2, target):
    bin_label = "GREEN bin" if "green" in target["class_name"] else "RED bin"
    print(f"[R2] Starting pick+sort -> {bin_label}")
 
    bin_hover, bin_drop = get_r2_bin_poses(target["class_name"])
 
    state["r2_status"] = "PICKING"
    state["r2_chip"]   = target
 
    r2.update_tool()
    r2.release_with_tool()
    refresh_display()
 
    # descend and pick
    move_above_target(r2, target);   refresh_display()
    move_to_base_hover(r2, target);  refresh_display()
    move_to_pick_height(r2, target)
    time.sleep(PAUSE_AT_PICK_SEC);   refresh_display()
 
    r2.grasp_with_tool()
    time.sleep(GRIPPER_WAIT_SEC);    refresh_display()
 
    # lift clear
    move_above_target(r2, target);   refresh_display()
 
    # observation -> midpoint -> bin hover -> bin drop
    state["r2_status"] = "SORTING"
    r2.move_joints(*R2_OBSERVATION_JOINTS);   refresh_display()
    r2.move_joints(*R2_BIN_MIDPOINT_JOINTS);  refresh_display()
    r2.move_joints(*bin_hover);               refresh_display()
    r2.move_joints(*bin_drop);                refresh_display()
 
    # release
    r2.release_with_tool()
    time.sleep(RELEASE_WAIT_SEC);             refresh_display()
    state["r2_sorts"] += 1
 
    # return: bin hover -> midpoint -> observation
    r2.move_joints(*bin_hover);               refresh_display()
    r2.move_joints(*R2_BIN_MIDPOINT_JOINTS);  refresh_display()
    r2.move_joints(*R2_OBSERVATION_JOINTS);   refresh_display()
 
    state["r2_status"] = "IDLE"
    state["r2_chip"]   = None
    print("[R2] Sort complete.")
 
 
 
 
# =========================================================
# MAIN
# =========================================================
def main():
    r1 = None
    r2 = None
    model = YOLO(str(MODEL_PATH))
 
    r1_calib = load_calibration(R1_CALIB_DIR)
    r2_calib = load_calibration(R2_CALIB_DIR)
    r1_hover = r1_calib["hover_reference"]
    r2_hover = r2_calib["hover_reference"]
 
    r1_conv_id = None
    r2_conv_id = None
 
    # cache last good frames for display continuity
    last_r1_frame = None
    last_r2_frame = None
    last_r1_det   = []
    last_r2_det   = []
    last_r1_live  = None
    last_r2_live  = None
 
    cv2.namedWindow("NIRYO CONTROL", cv2.WINDOW_NORMAL)
 
    try:
        print("Connecting to robot 1...")
        r1 = NiryoRobot(R1_IP)
        r1.clear_collision_detected()
        r1.update_tool()
 
        print("Connecting to robot 2...")
        r2 = NiryoRobot(R2_IP)
        r2.clear_collision_detected()
        r2.update_tool()
 
        r1_conveyors = r1.get_connected_conveyors_id()
        r2_conveyors = r2.get_connected_conveyors_id()
 
        if not r1_conveyors:
            print("No conveyor on robot 1.")
            return
        if not r2_conveyors:
            print("No conveyor on robot 2.")
            return
 
        r1_conv_id = r1_conveyors[0]
        r2_conv_id = r2_conveyors[0]
 
        # wire up live display so action functions can refresh the window
        display_refs["r1"]      = r1
        display_refs["r2"]      = r2
        display_refs["model"]   = model
        display_refs["r1_calib"] = r1_calib
        display_refs["r1_hover"] = r1_hover
        display_refs["r2_calib"] = r2_calib
        display_refs["r2_hover"] = r2_hover
 
        r1.move_joints(*R1_OBSERVATION_JOINTS)
        r2.move_joints(*R2_OBSERVATION_JOINTS)
 
        r1.run_conveyor(r1_conv_id, speed=CONVEYOR_SPEED_R1, direction=CONVEYOR_DIR)
        state["conv1_running"] = True
 
        while True:
            # ---------- ROBOT 1 frame ----------
            r1_img_raw = r1.get_img_compressed()
            if r1_img_raw is not None:
                r1_frame = decode_robot_image(r1_img_raw)
                if r1_frame is not None:
                    last_r1_frame = r1_frame
                    display_refs["last_r1_frame"] = r1_frame
                    r1_results   = model(r1_frame, conf=CONFIDENCE, verbose=False)[0]
                    last_r1_det  = extract_roi_detections(r1_results, model, R1_ROI, R1_VALID_CLASSES)
                    display_refs["last_r1_det"] = last_r1_det
                    last_r1_live = choose_target_in_roi(
                        last_r1_det, r1_calib, r1_hover, R1_ROI,
                        R1_APPROACH_HOVER_OFFSET_M, R1_DESCEND_FROM_HOVER_M
                    )
 
            # Status: only update to DETECTING/IDLE when robot is free
            if state["r1_status"] not in ("PICKING", "TRANSFERRING"):
                if last_r1_live is not None:
                    state["r1_status"] = "DETECTING"
                    state["r1_chip"]   = last_r1_live
                else:
                    state["r1_status"] = "IDLE"
                    state["r1_chip"]   = None
 
            # ---------- idle R2 frame ----------
            r2_img_raw = r2.get_img_compressed()
            if r2_img_raw is not None:
                r2_frame = decode_robot_image(r2_img_raw)
                if r2_frame is not None:
                    last_r2_frame = r2_frame
                    display_refs["last_r2_frame"] = r2_frame
                    r2_res_idle   = model(r2_frame, conf=CONFIDENCE, verbose=False)[0]
                    last_r2_det   = extract_roi_detections(r2_res_idle, model, R2_ROI, R2_VALID_CLASSES)
                    display_refs["last_r2_det"] = last_r2_det
                    last_r2_live  = choose_target_in_roi(
                        last_r2_det, r2_calib, r2_hover, R2_ROI,
                        R2_APPROACH_HOVER_OFFSET_M, R2_DESCEND_FROM_HOVER_M
                    )
 
            if state["r2_status"] not in ("PICKING", "SORTING"):
                if last_r2_live is not None:
                    state["r2_status"] = "DETECTING"
                    state["r2_chip"]   = last_r2_live
                else:
                    state["r2_status"] = "IDLE"
                    state["r2_chip"]   = None
 
            # ---------- display ----------
            window = compose_display(
                last_r1_frame, last_r1_det, last_r1_live,
                last_r2_frame, last_r2_det, last_r2_live,
            )
            cv2.imshow("NIRYO CONTROL", window)
 
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
 
            # ---------- trigger pick if R1 sees chip ----------
            if last_r1_live is not None:
                print(f"[MASTER] R1 detected {last_r1_live['class_name']} -> stopping conveyor 1")
                r1.stop_conveyor(r1_conv_id)
                state["conv1_running"] = False
                state["cycle_start"]   = time.time()
 
                # --- 2-second settle: keep display live while waiting ---
                print("[MASTER] R1 settling for 2s to get final chip position...")
                state["r1_status"] = "DETECTING"
                settle_deadline = time.time() + 2.0
                while time.time() < settle_deadline:
                    r1_img_raw2 = r1.get_img_compressed()
                    if r1_img_raw2 is not None:
                        r1f2 = decode_robot_image(r1_img_raw2)
                        if r1f2 is not None:
                            last_r1_frame = r1f2
                            r1_res2 = model(r1f2, conf=CONFIDENCE, verbose=False)[0]
                            last_r1_det  = extract_roi_detections(r1_res2, model, R1_ROI, R1_VALID_CLASSES)
                            last_r1_live = choose_target_in_roi(
                                last_r1_det, r1_calib, r1_hover, R1_ROI,
                                R1_APPROACH_HOVER_OFFSET_M, R1_DESCEND_FROM_HOVER_M
                            )
                            if last_r1_live:
                                state["r1_chip"] = last_r1_live
                    win_settle = compose_display(
                        last_r1_frame, last_r1_det, last_r1_live,
                        last_r2_frame, last_r2_det, last_r2_live,
                    )
                    cv2.imshow("NIRYO CONTROL", win_settle)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        return
 
                # use the freshest detection after settle
                r1_final = last_r1_live if last_r1_live is not None else state["r1_chip"]
                print(f"[MASTER] R1 final target after settle: {r1_final['class_name'] if r1_final else 'NONE'}")
 
                if r1_final is None:
                    # chip disappeared — restart conveyor and retry
                    r1.run_conveyor(r1_conv_id, speed=CONVEYOR_SPEED_R1, direction=CONVEYOR_DIR)
                    state["conv1_running"] = True
                    state["r1_status"] = "IDLE"
                    continue
 
                run_r1_pick_and_transfer(r1, r1_final)
 
                r2.run_conveyor(r2_conv_id, speed=CONVEYOR_SPEED_R2, direction=CONVEYOR_DIR)
                state["conv2_running"] = True
                time.sleep(TRANSFER_SETTLE_ON_CONVEYOR2_SEC)
 
                # ---------- R2 wait loop ----------
                r2_done = False
                while not r2_done:
                    r2_img_raw2 = r2.get_img_compressed()
                    if r2_img_raw2 is not None:
                        r2f = decode_robot_image(r2_img_raw2)
                        if r2f is not None:
                            last_r2_frame = r2f
                            display_refs["last_r2_frame"] = r2f
                            r2_res = model(r2f, conf=CONFIDENCE, verbose=False)[0]
                            last_r2_det  = extract_roi_detections(r2_res, model, R2_ROI, R2_VALID_CLASSES)
                            display_refs["last_r2_det"] = last_r2_det
                            last_r2_live = choose_target_in_roi(
                                last_r2_det, r2_calib, r2_hover, R2_ROI,
                                R2_APPROACH_HOVER_OFFSET_M, R2_DESCEND_FROM_HOVER_M
                            )
 
                    if state["r2_status"] not in ("PICKING", "SORTING"):
                        if last_r2_live is not None:
                            state["r2_status"] = "DETECTING"
                            state["r2_chip"]   = last_r2_live
                        else:
                            state["r2_status"] = "IDLE"
 
                    window2 = compose_display(
                        last_r1_frame, last_r1_det, None,
                        last_r2_frame, last_r2_det, last_r2_live,
                    )
                    cv2.imshow("NIRYO CONTROL", window2)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        return
 
                    if last_r2_live is not None:
                        r2.stop_conveyor(r2_conv_id)
                        state["conv2_running"] = False
 
                        # --- 2-second settle for R2 as well ---
                        print("[MASTER] R2 settling for 2s to get final chip position...")
                        state["r2_status"] = "DETECTING"
                        settle_deadline2 = time.time() + 2.0
                        while time.time() < settle_deadline2:
                            r2_img_s = r2.get_img_compressed()
                            if r2_img_s is not None:
                                r2f_s = decode_robot_image(r2_img_s)
                                if r2f_s is not None:
                                    last_r2_frame = r2f_s
                                    r2_res_s = model(r2f_s, conf=CONFIDENCE, verbose=False)[0]
                                    last_r2_det  = extract_roi_detections(r2_res_s, model, R2_ROI, R2_VALID_CLASSES)
                                    last_r2_live = choose_target_in_roi(
                                        last_r2_det, r2_calib, r2_hover, R2_ROI,
                                        R2_APPROACH_HOVER_OFFSET_M, R2_DESCEND_FROM_HOVER_M
                                    )
                                    if last_r2_live:
                                        state["r2_chip"] = last_r2_live
                            win_s2 = compose_display(
                                last_r1_frame, last_r1_det, None,
                                last_r2_frame, last_r2_det, last_r2_live,
                            )
                            cv2.imshow("NIRYO CONTROL", win_s2)
                            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                                return
 
                        r2_final = last_r2_live if last_r2_live is not None else state["r2_chip"]
                        print(f"[MASTER] R2 final target after settle: {r2_final['class_name'] if r2_final else 'NONE'}")
 
                        if r2_final is None:
                            # chip moved off ROI — restart conv2 and keep waiting
                            r2.run_conveyor(r2_conv_id, speed=CONVEYOR_SPEED_R2, direction=CONVEYOR_DIR)
                            state["conv2_running"] = True
                            state["r2_status"] = "IDLE"
                            continue
 
                        run_r2_pick_and_sort(r2, r2_final)
 
                        if state["cycle_start"]:
                            state["last_cycle_ms"] = int((time.time() - state["cycle_start"]) * 1000)
 
                        r2_done = True
 
                # restart conveyor 1
                r1.run_conveyor(r1_conv_id, speed=CONVEYOR_SPEED_R1, direction=CONVEYOR_DIR)
                state["conv1_running"] = True
 
    except Exception as e:
        print(f"ERROR: {e}")
        state["r1_status"] = "ERROR"
        state["r2_status"] = "ERROR"
 
    finally:
        cv2.destroyAllWindows()
        for robot, conv_id in [(r1, r1_conv_id), (r2, r2_conv_id)]:
            if robot is not None:
                try:
                    if conv_id is not None:
                        robot.stop_conveyor(conv_id)
                except Exception:
                    pass
                try:
                    robot.close_connection()
                except Exception:
                    pass
        print("Connections closed.")
 
 
if __name__ == "__main__":
    main()
 