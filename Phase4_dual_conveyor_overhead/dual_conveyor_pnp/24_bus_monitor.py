"""
24_bus_monitor.py

Print every message on the bus (alerts, dispatches, robot states, results).
Same role as `ros2 topic echo` later.

    python 24_bus_monitor.py
    python 24_bus_monitor.py --hide /robot_150/status /overhead/zone_state

Send a fake alert (no camera needed) to test the supervisor + a robot:

    python 24_bus_monitor.py --fake-alert C1
"""

import argparse
import json
import time

from common.bus import TOPIC_ALERT, Bus
from common.config import conveyor_key, load_system


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hide", nargs="*", default=[])
    parser.add_argument("--fake-alert", metavar="CONVEYOR")
    args = parser.parse_args()
    system = load_system()

    if args.fake_alert:
        conveyor = conveyor_key(args.fake_alert)
        if not system["conveyors"][conveyor].get("robot"):
            raise SystemExit(f"{conveyor} has no robot; only robot-owned conveyors get alerts.")
        sender = Bus("overhead", system)
        sender.send("supervisor", TOPIC_ALERT, {
            "event": "CHIP_IN_ALERT_ZONE", "id": "FAKE", "conveyor": conveyor,
            "tag_id": system["conveyors"][conveyor]["tag_id"],
            "robot": system["conveyors"][conveyor]["robot"],
            "class_name": "manual_test", "confidence": 1.0, "camera_index": -1,
        })
        sender.close()
        print(f"Sent fake CHIP_IN_ALERT_ZONE for {conveyor}.")
        return

    bus = Bus("monitor", system)
    print("Listening. Ctrl+C to stop.")
    try:
        while True:
            for message in bus.receive():
                if message.get("topic") in args.hide:
                    continue
                print(f"{message.get('time', '')[11:19]} {message.get('sender'):>10} "
                      f"{message.get('topic')}: {json.dumps(message.get('data'))}", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()


if __name__ == "__main__":
    main()
