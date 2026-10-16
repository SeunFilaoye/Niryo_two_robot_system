"""
10_clear_collision.py

Same as Robot 3/10_clear_collision.py, with --robot.
Run it when a move fails with:
  "clear_collision_detected() must be called before attempting a new move command"

    python 10_clear_collision.py --robot 150
"""

from pyniryo import NiryoRobot

from common.settings import get as get_settings  # --robot

S = get_settings(need_conveyor=False)

robot = NiryoRobot(S.ROBOT_IP)
robot.clear_collision_detected()
print(f"Collision flag cleared on {S.ROBOT_NAME} ({S.ROBOT_IP}).")
robot.close_connection()
