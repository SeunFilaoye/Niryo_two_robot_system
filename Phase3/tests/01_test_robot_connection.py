from pyniryo import NiryoRobot

ROBOT_IP = "192.168.0.199"  # change to new robot IP

robot = NiryoRobot(ROBOT_IP)

print("Connected to robot.")

# Do not move yet unless pose is safe.
# robot.move_joints(HOME_JOINTS)

robot.close_connection()
print("Disconnected.")