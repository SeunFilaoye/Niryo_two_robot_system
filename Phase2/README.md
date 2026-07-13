\# Single Robot Vision Pick-and-Place Baseline v1



This version successfully:

\- connects to Niryo robot at 192.168.0.199

\- streams wrist camera

\- detects normal chips using YOLO

\- uses bright blue ROI

\- stops conveyor when chip enters ROI

\- waits 2.5 seconds for camera settling

\- re-detects chip

\- maps pixel center to robot X/Y using homography

\- picks chip with vacuum pump

\- moves through midpoint

\- drops chip

\- returns home

\- restarts conveyor



Working command:



```powershell

.\\\\.venv\\\\Scripts\\\\Activate.ps1

python .\\\\07\\\_run\\\_pick\\\_place.py

Important files:



07\\\_run\\\_pick\\\_place.py = final working pipeline

04\\\_workspace\\\_calibration.py = calibration

calibration/ = robot-specific calibration files

models/best.pt = chip detection model

logs/pick\\\_place\\\_log.csv = run log



\\## 4. For another robot later



When moving to a new robot, keep the code but redo only:



```text

ROBOT\\\_IP

HOME\\\_JOINTS

MIDPOINT\\\_JOINTS

DROP\\\_JOINTS

calibration/

pick\\\_reference.json


