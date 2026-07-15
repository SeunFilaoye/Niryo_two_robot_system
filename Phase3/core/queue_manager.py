import math
import time


class QueueManager:
    """
    FIFO queue rules:

    1. A chip is registered when it appears inside the single ROI.
    2. Queue order is based on first-seen time.
    3. Only the oldest chip can be selected.
    4. The oldest chip must be tracked for two seconds before pickup.
    5. There is no trigger line, collector region, or motion state.
    """

    def __init__(
        self,
        match_distance_px=170,
        duplicate_distance_px=220,
        stale_after_sec=60.0,
        pick_delay_sec=2.0,
    ):
        self.queue = []
        self.next_id = 1

        self.match_distance_px = match_distance_px
        self.duplicate_distance_px = duplicate_distance_px
        self.stale_after_sec = stale_after_sec
        self.pick_delay_sec = pick_delay_sec

    @staticmethod
    def distance(first, second):
        return math.hypot(
            first["cx"] - second["cx"],
            first["cy"] - second["cy"],
        )

    def active_queue(self):
        return [
            item
            for item in self.queue
            if item["status"] == "active"
        ]

    def find_match(self, detection, excluded_ids):
        best_item = None
        best_distance = float("inf")

        for item in self.active_queue():
            if item["id"] in excluded_ids:
                continue

            if item["class_name"] != detection["class_name"]:
                continue

            distance = self.distance(
                detection,
                item,
            )

            if (
                distance <= self.match_distance_px
                and distance < best_distance
            ):
                best_item = item
                best_distance = distance

        return best_item

    def should_create_new(self, detection):
        same_class_items = [
            item
            for item in self.active_queue()
            if item["class_name"] == detection["class_name"]
        ]

        if not same_class_items:
            return True

        closest_distance = min(
            self.distance(detection, item)
            for item in same_class_items
        )

        return closest_distance > self.duplicate_distance_px

    def add_chip(self, detection):
        now = time.time()

        item = {
            "id": self.next_id,
            "class_name": detection["class_name"],
            "confidence": detection["confidence"],
            "bbox": detection["bbox"],
            "cx": detection["cx"],
            "cy": detection["cy"],

            "first_seen": now,
            "last_seen": now,
            "ready_at": now + self.pick_delay_sec,

            "status": "active",
        }

        self.queue.append(item)

        print(
            f"FIFO ADD: #{item['id']} "
            f"{item['class_name']}"
        )

        self.next_id += 1

    @staticmethod
    def update_item(item, detection):
        item["class_name"] = detection["class_name"]
        item["confidence"] = detection["confidence"]
        item["bbox"] = detection["bbox"]
        item["cx"] = detection["cx"]
        item["cy"] = detection["cy"]
        item["last_seen"] = time.time()

    def update(self, detections):
        matched_ids = set()

        for detection in detections:
            match = self.find_match(
                detection,
                matched_ids,
            )

            if match is not None:
                self.update_item(
                    match,
                    detection,
                )

                matched_ids.add(
                    match["id"],
                )

            elif self.should_create_new(detection):
                self.add_chip(detection)

        self.remove_stale()

    def remove_stale(self):
        now = time.time()

        for item in self.active_queue():
            if now - item["last_seen"] > self.stale_after_sec:
                item["status"] = "stale"

                print(
                    f"FIFO STALE: #{item['id']} "
                    f"{item['class_name']}"
                )

    def next_ready_chip(self):
        active = self.active_queue()

        if not active:
            return None, None

        oldest = active[0]
        now = time.time()

        wait_remaining = max(
            0.0,
            oldest["ready_at"] - now,
        )

        oldest["wait_remaining"] = wait_remaining
        oldest["is_ready"] = wait_remaining <= 0.0

        # The chip must still be visible when the timer finishes.
        recently_visible = (
            now - oldest["last_seen"]
            <= 1.0
        )

        if oldest["is_ready"] and recently_visible:
            return oldest, oldest

        return None, oldest

    def mark_picked(self, item):
        item["status"] = "picked"

        print(
            f"FIFO PICKED: #{item['id']} "
            f"{item['class_name']}"
        )

    def mark_failed(self, item):
        # Keep the same FIFO entry and wait two seconds again.
        now = time.time()

        item["ready_at"] = now + self.pick_delay_sec
        item["last_seen"] = now
        item["is_ready"] = False

        print(
            f"FIFO PICK FAILED — RETRYING: "
            f"#{item['id']} {item['class_name']}"
        )
