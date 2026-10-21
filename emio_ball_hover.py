"""
Emio: hover the gripper over an orange ball.

- 3 pieces of BLUE tape on the table corners define the table plane.
- ORANGE ball is detected and its position is mapped onto that plane.
- Target (effector goal) is moved above the ball; SOFA inverse solver drives the motors.

Run:
    runSofa -l SofaPython3,SofaImGui -g imgui emio_ball_hover.py

Tune the HSV ranges and HOVER_HEIGHT for your setup (see CONFIG).
"""

import numpy as np
import cv2
import Sofa
import Sofa.Core

from emio import Emio
from emio.utils import addHeader, addSolvers
from emioapi import EmioCamera

# ---------------- CONFIG ----------------
# OpenCV HSV: H in [0,180], S,V in [0,255]
BLUE_LO, BLUE_HI     = (95,  80,  60), (130, 255, 255)
ORANGE_LO, ORANGE_HI = (5,   120, 100), (20,  255, 255)

MIN_TAPE_AREA   = 150    # px^2
MIN_BALL_AREA   = 200    # px^2
BALL_RADIUS_MM  = 20.0   # ball radius
HOVER_HEIGHT_MM = 40.0   # gap between gripper and top of ball
SMOOTHING       = 0.15   # 0..1, target lerp per frame (lower = smoother)
SHOW_DEBUG      = True   # draw detections in a window
# ----------------------------------------


# ---------- vision helpers ----------
def find_blobs(hsv, lo, hi, min_area):
    """Return list of (cx, cy, area) for colour blobs, largest first."""
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        m = cv2.moments(c)
        blobs.append((m["m10"] / m["m00"], m["m01"] / m["m00"], a))
    blobs.sort(key=lambda b: -b[2])
    return blobs


def pick_corners(blobs):
    """From all blue blobs, return the 3 that form the largest triangle (the corners)."""
    from itertools import combinations
    if len(blobs) < 3:
        return []
    best, best_area = None, -1
    for a, b, c in combinations(blobs, 3):
        area = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2
        if area > best_area:
            best, best_area = (a, b, c), area
    return list(best)


def order_tape(points):
    """Order 3 tape points consistently: origin = closest to top-left of image."""
    pts = np.array(points, dtype=float)
    i0 = np.argmin(pts[:, 0] + pts[:, 1])
    rest = [i for i in range(3) if i != i0]
    # second point = one more to the right (x-axis), third = other (y-axis)
    i1, i2 = sorted(rest, key=lambda i: -pts[i, 0])
    return pts[[i0, i1, i2]]


class TablePlane:
    """
    Maps image pixels -> 3D simulation coords on the table plane,
    using 3 tape corners (pixel) and their 3D positions from the camera.
    """
    def __init__(self, px, p3d):
        px = np.asarray(px, float)      # (3,2) pixels
        p3d = np.asarray(p3d, float)    # (3,3) sim coords (mm)
        # affine: [u v 1] -> xyz
        A = np.hstack([px, np.ones((3, 1))])
        self.M = np.linalg.solve(A, p3d)          # (3,3)
        n = np.cross(p3d[1] - p3d[0], p3d[2] - p3d[0])
        n /= np.linalg.norm(n)
        if n[1] < 0:            # make normal point "up" (+y in Emio scene)
            n = -n
        self.normal = n

    def pixel_to_sim(self, u, v):
        return np.array([u, v, 1.0]) @ self.M


# ---------- SOFA controller ----------
class BallHoverController(Sofa.Core.Controller):
    def __init__(self, target_mo, camera):
        super().__init__(name="BallHoverController")
        self.target = target_mo
        self.cam = camera
        self.plane = None
        self.goal = None

    def onAnimateBeginEvent(self, _):
        if not self.cam.is_running:
            return
        self.cam.update()
        frame = self.cam.frame
        if frame is None:
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1) table plane from blue tape (recomputed each frame; cache if stable)
        tape = pick_corners(find_blobs(hsv, BLUE_LO, BLUE_HI, MIN_TAPE_AREA)[:8])
        if len(tape) == 3:
            px = order_tape([(b[0], b[1]) for b in tape])
            p3d = []
            for u, v in px:
                p = self.cam.image_to_simulation(int(u), int(v))
                if p is None or not np.all(np.isfinite(p)):
                    p3d = None
                    break
                p3d.append(p)
            if p3d is not None:
                self.plane = TablePlane(px, p3d)

        # 2) orange ball
        ball = find_blobs(hsv, ORANGE_LO, ORANGE_HI, MIN_BALL_AREA)
        if self.plane is not None and ball:
            bu, bv, _ = ball[0]
            contact = self.plane.pixel_to_sim(bu, bv)     # ball footprint on table
            goal = contact + self.plane.normal * (2 * BALL_RADIUS_MM + HOVER_HEIGHT_MM)
            self.goal = goal if self.goal is None else \
                        (1 - SMOOTHING) * self.goal + SMOOTHING * goal

            with self.target.position.writeable() as pos:
                pos[0][:3] = self.goal

        if SHOW_DEBUG:
            dbg = frame.copy()
            for u, v, _ in tape:
                cv2.circle(dbg, (int(u), int(v)), 8, (255, 0, 0), 2)
            if ball:
                cv2.circle(dbg, (int(ball[0][0]), int(ball[0][1])), 12, (0, 140, 255), 2)
            cv2.imshow("emio ball hover", dbg)
            cv2.waitKey(1)

    def onSimulationStopEvent(self, _):
        self.cam.close()
        cv2.destroyAllWindows()


# ---------- scene ----------
def createScene(root):
    settings, modelling, simulation = addHeader(root)
    addSolvers(simulation)

    emio = Emio(name="Emio",
                legsName=["blueleg"],
                centerPartName="whitepart",   # gripper center part
                centerPartType="gripper",
                extended=True)
    if not emio.isValid():
        return
    simulation.addChild(emio)
    emio.attachCenterPartToLegs()

    # Effector goal (Rigid3: x y z qx qy qz qw)
    target = modelling.addChild("Target")
    target_mo = target.addObject("MechanicalObject", template="Rigid3",
                                 position=[[0, -150, 0, 0, 0, 0, 1]],
                                 showObject=True, showObjectScale=10)

    emio.addInverseComponentAndGUI(target_mo.position.linkpath, withGUI=True)
    emio.addConnectionComponents()   # drive the real robot

    cam = EmioCamera(show=False, track_markers=False,
                     compute_point_cloud=False, configuration="extended")
    if not cam.open():
        print("Camera failed to open")
        return

    root.addObject(BallHoverController(target_mo, cam))
    return root

