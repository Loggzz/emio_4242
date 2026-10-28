"""
HSV probe for the Emio camera. Click anywhere on the image to print that
pixel's HSV (OpenCV scale: H 0-180, S/V 0-255). Use it to set MARKER_LO/HI.
Press q to quit.
"""
import cv2
import numpy as np
from emioapi import EmioCamera

cam = EmioCamera(show=False, track_markers=False, configuration="extended")
if not cam.open():
    raise SystemExit("Camera failed")

hsv = None
def on_click(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN and hsv is not None:
        patch = hsv[max(0, y-2):y+3, max(0, x-2):x+3].reshape(-1, 3)
        print(f"({x},{y})  H={patch[:,0].mean():.0f}  S={patch[:,1].mean():.0f}  V={patch[:,2].mean():.0f}")

cv2.namedWindow("probe")
cv2.setMouseCallback("probe", on_click)
try:
    while cam.is_running:
        cam.update()
        f = cam.frame
        if f is None:
            continue
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        cv2.imshow("probe", f)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cam.close(); cv2.destroyAllWindows()