"""Tiny JSON message bus over localhost UDP.

Stands in for ROS2 topics while the system runs as plain Python processes.
Each message has the same shape a ROS2 std_msgs/String JSON payload will
have later, so the port to ROS2 replaces this file and nothing else:

    {"topic": "/overhead/alert", "sender": "overhead", "time": "...",
     "data": {...}}

Every participant owns one UDP port from config/system.json -> bus.ports.
Messages are also copied to the "monitor" port so 24_bus_monitor.py can
show all traffic. Sending to a port nobody listens on is harmless.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timezone

from common.config import load_system


class Bus:
    def __init__(self, name: str, system: dict | None = None):
        system = system or load_system()
        self.name = name
        self.host = system["bus"]["host"]
        self.ports = system["bus"]["ports"]
        if name not in self.ports:
            raise ValueError(f"No bus port configured for '{name}'")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.ports[name]))
        self.sock.setblocking(False)

    def send(self, target: str, topic: str, data: dict) -> None:
        message = {
            "topic": topic,
            "sender": self.name,
            "time": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        payload = json.dumps(message).encode("utf-8")
        for name in {target, "monitor"}:
            port = self.ports.get(name)
            if port is None or name == self.name:
                continue
            try:
                self.sock.sendto(payload, (self.host, port))
            except OSError:
                pass

    def receive(self) -> list[dict]:
        messages = []
        while True:
            try:
                payload, _ = self.sock.recvfrom(65535)
            except (BlockingIOError, InterruptedError):
                return messages
            except OSError:
                return messages
            try:
                messages.append(json.loads(payload.decode("utf-8")))
            except ValueError:
                continue

    def close(self) -> None:
        self.sock.close()


# Topic names (identical to the planned ROS2 topics)
TOPIC_ALERT = "/overhead/alert"
TOPIC_ENTRY = "/overhead/entry"
TOPIC_ZONES = "/overhead/zone_state"


def command_topic(robot: str) -> str:
    return f"/supervisor/{robot}/command"


def status_topic(robot: str) -> str:
    return f"/{robot}/status"


def result_topic(robot: str) -> str:
    return f"/{robot}/check_result"
