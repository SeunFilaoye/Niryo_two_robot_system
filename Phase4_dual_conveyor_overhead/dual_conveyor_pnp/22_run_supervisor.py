"""
22_run_supervisor.py

Supervisor between the overhead monitor and the two robots.

  overhead  --CHIP_IN_ALERT_ZONE(C1)-->  supervisor
  supervisor --CHECK_CONVEYOR(C1)-->     robot_150   (when that robot is IDLE)
  robot_150 --CHECK_RESULT-->            supervisor

Rules
- Each alert queues its conveyor for the robot that owns it (once; duplicates
  while queued or being checked are ignored).
- One check per robot at a time, oldest request first.
- If the overhead still sees a chip in a conveyor's ALERT ROI some time after
  that conveyor was last checked, it is queued again (catches missed chips).
- A robot with no heartbeat is OFFLINE; its in-flight check is re-queued.

    python 22_run_supervisor.py
"""

import argparse
import csv
import time
from datetime import datetime

from common.bus import (
    TOPIC_ALERT, TOPIC_ENTRY, TOPIC_ZONES, Bus, command_topic, result_topic, status_topic,
)
from common.config import LOGS_DIR, load_system

EVENT_LOG = LOGS_DIR / "supervisor_events.csv"


def log(event: str, **fields) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detail = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[{stamp}] {event} {detail}", flush=True)
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not EVENT_LOG.exists()
    with EVENT_LOG.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new:
            writer.writerow(["timestamp", "event", "details"])
        writer.writerow([stamp, event, detail])


class RobotSlot:
    def __init__(self, name: str, conveyors: list[str]):
        self.name = name
        self.conveyors = conveyors
        self.state = "OFFLINE"
        self.heartbeat = 0.0
        self.pending: list[dict] = []
        self.busy: dict | None = None

    def queued(self, conveyor: str) -> bool:
        return (any(p["conveyor"] == conveyor for p in self.pending)
                or (self.busy is not None and self.busy["conveyor"] == conveyor))


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    system = load_system()
    cfg = system["supervisor"]
    conveyors = system["conveyors"]
    slots = {name: RobotSlot(name, info["conveyors"]) for name, info in system["robots"].items()}
    status_topics = {status_topic(name): name for name in slots}
    result_topics = {result_topic(name): name for name in slots}

    bus = Bus("supervisor", system)
    zones: dict = {}
    zones_at = 0.0
    last_check_end = {name: 0.0 for name in conveyors}
    request_number = 0
    last_summary = 0.0

    def enqueue(conveyor: str, reason: str, alert_id=None):
        slot = slots[conveyors[conveyor]["robot"]]
        if slot.queued(conveyor):
            return
        slot.pending.append({"conveyor": conveyor, "reason": reason, "alert_id": alert_id,
                             "queued_at": time.monotonic()})
        log("QUEUED", robot=slot.name, conveyor=conveyor, reason=reason, alert=alert_id)

    log("SUPERVISOR_START", robots=list(slots), conveyors=list(conveyors))
    try:
        while True:
            now = time.monotonic()
            for message in bus.receive():
                topic, data = message.get("topic"), message.get("data", {})
                if topic == TOPIC_ALERT:
                    conveyor = data.get("conveyor")
                    if conveyor in conveyors:
                        log("ALERT", conveyor=conveyor, cls=data.get("class_name"),
                            id=data.get("id"))
                        enqueue(conveyor, "overhead_alert", data.get("id"))
                elif topic == TOPIC_ENTRY:
                    log("ENTRY", conveyor=data.get("conveyor"), cls=data.get("class_name"),
                        id=data.get("id"))
                elif topic == TOPIC_ZONES:
                    zones = data.get("zones", {})
                    zones_at = now
                elif topic in status_topics:
                    slot = slots[status_topics[topic]]
                    if slot.state != data.get("state"):
                        log("ROBOT_STATE", robot=slot.name, state=data.get("state"),
                            conveyor=data.get("conveyor"))
                    slot.state = data.get("state", "UNKNOWN")
                    slot.heartbeat = now
                elif topic in result_topics:
                    slot = slots[result_topics[topic]]
                    if slot.busy and data.get("request_id") == slot.busy["request_id"]:
                        log("CHECK_RESULT", robot=slot.name, conveyor=data.get("conveyor"),
                            result=data.get("result"), picked=data.get("picked"))
                        last_check_end[slot.busy["conveyor"]] = now
                        slot.busy = None

            for slot in slots.values():
                if slot.state != "OFFLINE" and now - slot.heartbeat > cfg["robot_offline_after_sec"]:
                    log("ROBOT_OFFLINE", robot=slot.name)
                    slot.state = "OFFLINE"
                    if slot.busy:
                        slot.pending.insert(0, {**slot.busy, "reason": "requeued_offline"})
                        slot.busy = None

                if slot.busy and now - slot.busy["sent_at"] > cfg["check_timeout_sec"]:
                    log("CHECK_TIMEOUT", robot=slot.name, conveyor=slot.busy["conveyor"])
                    last_check_end[slot.busy["conveyor"]] = now
                    slot.busy = None

            # Ignore zone reports once the overhead monitor goes quiet.
            fresh_zones = zones if now - zones_at < 3.0 else {}
            for conveyor, zone in fresh_zones.items():
                if (conveyor in conveyors and zone.get("observable") and zone.get("in_alert", 0) > 0
                        and now - last_check_end[conveyor] >= cfg["recheck_occupied_after_sec"]):
                    enqueue(conveyor, "alert_zone_still_occupied")

            for slot in slots.values():
                if slot.state == "IDLE" and slot.busy is None and slot.pending:
                    job = slot.pending.pop(0)
                    request_number += 1
                    request_id = f"R{request_number:04d}"
                    bus.send(slot.name, command_topic(slot.name), {
                        "command": "CHECK_CONVEYOR",
                        "request_id": request_id,
                        "conveyor": job["conveyor"],
                        "reason": job["reason"],
                        "alert_id": job.get("alert_id"),
                    })
                    slot.busy = {**job, "request_id": request_id, "sent_at": now}
                    slot.state = "DISPATCHED"
                    log("DISPATCH", robot=slot.name, conveyor=job["conveyor"],
                        request=request_id, reason=job["reason"])

            if now - last_summary > 10.0:
                last_summary = now
                summary = " | ".join(
                    f"{s.name}:{s.state} busy={s.busy['conveyor'] if s.busy else '-'} "
                    f"pending={[p['conveyor'] for p in s.pending]}"
                    for s in slots.values())
                print(f"STATUS {summary}", flush=True)

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nSupervisor stopping; telling robots to stop.")
        for name in slots:
            bus.send(name, command_topic(name), {"command": "STOP"})
    finally:
        bus.close()


if __name__ == "__main__":
    main()
