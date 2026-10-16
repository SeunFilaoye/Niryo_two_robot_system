"""
23_run_robot_worker.py

Robot side of the supervised system. One process per robot.

  IDLE      parked, waiting for the supervisor
  CHECKING  moved to the requested conveyor's VIEW pose; that conveyor's
            station ROI appears; Phase 3 FIFO picks every chip in it
  -> reports CHECK_RESULT, returns to park, IDLE again

A check ends once the station ROI has been empty for empty_timeout_sec,
after max_picks_per_check picks, or on a STOP command.

    python 23_run_robot_worker.py --robot 150

Q in the camera window stops this worker.
"""

import threading
import time

import cv2

from common.bus import Bus, command_topic, result_topic, status_topic
from common.cli import confirm, parse_robot, robot_parser
from common.config import load_robot_file, load_system, model_path
from common.niryo_utils import (
    connect, conveyor_hardware_map, disconnect, get_frame, move_joints,
    run_conveyor, stop_all_conveyors,
)
from common.station import Station, StationNotReady
from common.station_check import draw_banner, run_station_check
from common.wrist_vision import WristDetector


class Worker:
    def __init__(self, robot: str):
        self.robot = robot
        self.system = load_system()
        self.runtime = self.system["robot_runtime"]
        self.bus = Bus(robot, self.system)
        self.state = "STARTING"
        self.conveyor = None
        self.stop_requested = False
        self.quit = False
        self.queue: list[dict] = []
        self.lock = threading.Lock()
        self.window = f"{robot} worker"

        self.stations: dict[str, Station] = {}
        for conveyor in self.system["robots"][robot]["conveyors"]:
            try:
                self.stations[conveyor] = Station(robot, conveyor)
                print(f"{conveyor}: station ready")
            except StationNotReady as error:
                print(f"{conveyor}: NOT READY - {error}")
        if not self.stations:
            raise SystemExit("No trained station for this robot. Finish the training steps (04, 04b, 08) first.")

        park = load_robot_file(robot).get("park")
        self.park_joints = park["joints"] if park else next(iter(self.stations.values())).view_joints
        self.detector = WristDetector(model_path("wrist", self.system),
                                      self.runtime["wrist_confidence"])

    # ------------------------------------------------------------ bus

    def set_state(self, state: str, conveyor=None) -> None:
        with self.lock:
            self.state, self.conveyor = state, conveyor
        self.send_status()

    def send_status(self) -> None:
        with self.lock:
            data = {"state": self.state, "conveyor": self.conveyor,
                    "ready_conveyors": sorted(self.stations)}
        self.bus.send("supervisor", status_topic(self.robot), data)

    def heartbeat_loop(self) -> None:
        while not self.quit:
            self.send_status()
            time.sleep(self.runtime["heartbeat_period_sec"])

    def poll_commands(self) -> None:
        for message in self.bus.receive():
            if message.get("topic") != command_topic(self.robot):
                continue
            data = message.get("data", {})
            if data.get("command") == "STOP":
                print("STOP received.")
                self.stop_requested = True
                self.queue.clear()
            elif data.get("command") == "CHECK_CONVEYOR":
                self.queue.append(data)

    # ------------------------------------------------------------ work

    def park(self, niryo) -> None:
        move_joints(niryo, self.park_joints, "PARK")

    def handle_check(self, niryo, job: dict) -> None:
        conveyor = job.get("conveyor")
        request_id = job.get("request_id")
        print(f"\nCHECK_CONVEYOR {conveyor} ({request_id}, reason={job.get('reason')})")
        picked, result = 0, "ERROR"
        station = self.stations.get(conveyor)
        if station is None:
            print(f"{conveyor} is not trained on {self.robot}; rejecting.")
            result = "NOT_TRAINED"
        else:
            self.set_state("CHECKING", conveyor)
            self.stop_requested = False
            try:
                picked, result = run_station_check(
                    niryo, station, self.detector, self.runtime, self.window,
                    empty_timeout_sec=self.runtime["empty_timeout_sec"],
                    should_stop=lambda: self.stop_requested,
                    on_tick=self.poll_commands,
                    log=lambda text: print(text, flush=True),
                )
            except Exception as error:
                print(f"Check failed: {error}")
                result = "ERROR"
                try:
                    niryo.release_with_tool()
                    niryo.clear_collision_detected()
                except Exception:
                    pass
            if result == "QUIT":
                self.quit = True

        self.bus.send("supervisor", result_topic(self.robot), {
            "request_id": request_id, "conveyor": conveyor,
            "result": result, "picked": picked,
        })
        print(f"CHECK_RESULT {conveyor}: {result}, picked {picked}")
        if not self.quit:
            self.set_state("RETURNING")
            self.park(niryo)

    def run(self) -> None:
        niryo = None
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        try:
            niryo = connect(self.robot)
            niryo.update_tool()

            if self.runtime["run_conveyors"]:
                hardware = conveyor_hardware_map(self.robot)
                for conveyor in self.system["robots"][self.robot]["conveyors"]:
                    if conveyor in hardware:
                        run_conveyor(niryo, hardware[conveyor], self.runtime["conveyor_speed"])
                        print(f"{conveyor} running.")
                    else:
                        print(f"{conveyor} hardware ID unknown (run 01b); not started.")

            self.park(niryo)
            self.set_state("IDLE")
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

            while not self.quit:
                self.poll_commands()
                if self.queue:
                    self.handle_check(niryo, self.queue.pop(0))
                    if not self.quit:
                        self.set_state("IDLE")
                    continue
                frame = get_frame(niryo)
                if frame is not None:
                    draw_banner(frame, [f"{self.robot} | IDLE",
                                        "parked - waiting for the supervisor"])
                    cv2.imshow(self.window, frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    self.quit = True
        except KeyboardInterrupt:
            print("\nWorker interrupted.")
        except Exception as error:
            print(f"Worker failed: {error}")
        finally:
            self.quit = True
            self.set_state("OFFLINE")
            cv2.destroyAllWindows()
            if niryo is not None:
                try:
                    niryo.release_with_tool()
                except Exception:
                    pass
                stop_all_conveyors(niryo)
            disconnect(niryo, self.robot)
            self.bus.close()


def main():
    args = parse_robot(robot_parser(__doc__))
    worker = Worker(args.robot)
    if not confirm(f"{args.robot} will run conveyors and pick automatically."):
        return
    worker.run()


if __name__ == "__main__":
    main()
