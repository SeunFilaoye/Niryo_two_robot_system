from pyniryo import NiryoRobot

robot = NiryoRobot("192.168.0.199")
robot.clear_collision_detected()
robot.close_connection()