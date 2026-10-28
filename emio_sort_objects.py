"""
Emio: sort 6 play-dough objects from the tray's HOME square to their numbered spots.

Objects (from the photos) -> spot numbers (edit OBJECT_TO_SPOT if different):
    cheese    -> 2        eggplant -> 1
    burger    -> 6        donut    -> 4
    bread     -> 3        tomato   -> 5

Pipeline per object
  1. Camera looks at the HOME square (centre of the tray, found from the 4 purple corners).
  2. The object there is classified by colour/shape (nearest neighbour against
     a reference set you record once with --calibrate).
  3. The pick-and-place motor sequence for that spot is executed.
  4. Repeat until HOME is empty.

Poses come from movej.py (same folder): hover/approach/pick/close_hover to grab
from HOME, then poseN_hover/approach/release/open_hover for spot N.

Run (from the emio_4242 folder):
    python emio_sort_objects.py --tune          # first: set marker colour with sliders
    python emio_sort_objects.py --calibrate     # once: all objects in their final spots, SPACE
    python emio_sort_objects.py --calibrate-one # alt: one object at a time in HOME
    python emio_sort_objects.py --dry           # detect + classify only, robot stays still
    python emio_sort_objects.py                 # full run
"""

import sys, json, os, time, math
import numpy as np
import cv2
from emioapi import EmioAPI

# ======================= POSES (from movej.py) =======================
# movej.py must sit next to this file. Its POSE_SEQUENCE is the single source
# of truth; we index it by pose name.
from movej import POSE_SEQUENCE
POSES = {name: list(angles) for name, angles in POSE_SEQUENCE}

HOME_POSE  = POSES["home"]
PICK_SEQ   = ["hover", "approach", "pick", "close_hover"]
def place_seq(spot):
    seq = [f"pose{spot}_hover", f"pose{spot}_approach", f"pose{spot}_release"]
    if f"pose{spot}_open_hover" in POSES:            # pose1 has no open_hover
        seq.append(f"pose{spot}_open_hover")
    return seq + ["hover"]                           # back over HOME, gripper open

MAX_VELOCITY = [25, 25, 25, 25]
MOVE_WAIT    = 5.0        # same settle time movej.py uses
SETTLE_POLL  = True       # poll motors.moving and return early if settled
# =====================================================================

OBJECT_TO_SPOT = {"cheese": 2, "eggplant": 1, "burger": 6,
                  "donut": 4, "bread": 3, "tomato": 5}
OBJECTS = list(OBJECT_TO_SPOT)

# Corner MARKERS are PURPLE. OpenCV HSV: H 0-180, S 0-255, V 0-255.
# Run  --tune  to set these with sliders; the result is saved to marker_hsv.json
# and loaded automatically on later runs (overrides the values below).
MARKER_LO, MARKER_HI = (125, 60, 50), (160, 255, 255)
MIN_MARKER_AREA = 25
MIN_TRAY_FRAC   = 0.03         # tray must cover >= this fraction of the frame
SHOW_MASK = True               # extra window showing the marker mask
HSV_FILE  = "marker_hsv.json"
if os.path.exists(HSV_FILE):
    _h = json.load(open(HSV_FILE))
    MARKER_LO, MARKER_HI = tuple(_h["lo"]), tuple(_h["hi"])
    MIN_MARKER_AREA = _h.get("min_area", MIN_MARKER_AREA)
MIN_OBJ_AREA = 300
SIG_FILE = "object_signatures.json"


# ---------- tray geometry ----------
# Tray as a 3x3 grid of blocks (rows top->bottom, cols left->right).
# A section may span several blocks. "H" = home square.
SECTIONS = [[2, 2, 1],
            [6, "H", 1],
            [3, 5, 4]]
# Orientation comes from the camera image: the layout's top row (2, 1) is the
# edge FURTHEST from the camera. FAR_SIDE says where that edge appears in the
# image: "top", "bottom", "left" or "right". Check the --dry overlay.
FAR_SIDE = "top"


def marker_blobs(hsv):
    mask = cv2.inRange(hsv, np.array(MARKER_LO), np.array(MARKER_HI))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if SHOW_MASK:
        cv2.imshow("marker mask", mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a >= MIN_MARKER_AREA:
            m = cv2.moments(c); blobs.append((a, m["m10"] / m["m00"], m["m01"] / m["m00"]))
    blobs.sort(reverse=True)
    return np.array([[u, v] for _, u, v in blobs], np.float32)


def _largest_quad(pts):
    """Pick the 4 blobs spanning the largest area (ignores stray marker-coloured blobs)."""
    from itertools import combinations
    if len(pts) == 4:
        return pts
    best, best_a = None, -1
    for q in combinations(range(len(pts)), 4):
        p = pts[list(q)]
        a = cv2.contourArea(cv2.convexHull(p.astype(np.float32)))
        if a > best_a:
            best, best_a = p, a
    return best


def corner_status():
    n = getattr(tray_corners, "last_n", 0)
    if n >= 3:  return f"markers: {n}"
    if n < 0:   return f"markers: {-n} but collinear/too small"
    return f"markers: {n} (need 3)"


def tray_corners(hsv):
    """
    Returns the 4 tray corners in image coords, ordered [TL, TR, BR, BL]
    of the SECTIONS layout. Uses 4 purple markers; falls back to 3 by
    reconstructing the 4th. Orientation: FAR_SIDE of the image = layout top.
    """
    pts = marker_blobs(hsv)
    tray_corners.last_n = len(pts)
    if len(pts) < 3:
        return None
    if len(pts) >= 4:
        quad = _largest_quad(pts[:8])
    else:                                              # 3 markers: reconstruct
        d = [np.linalg.norm(pts[(i+1) % 3] - pts[(i+2) % 3]) for i in range(3)]
        mid = int(np.argmax(d))
        quad = np.vstack([pts, (pts[(mid+1) % 3] + pts[(mid+2) % 3] - pts[mid])[None]])

    # Reject degenerate quads (collinear noise blobs -> a "line").
    area = cv2.contourArea(cv2.convexHull(quad.astype(np.float32)))
    if area < MIN_TRAY_FRAC * hsv.shape[0] * hsv.shape[1]:
        tray_corners.last_n = -len(pts)                # negative = degenerate
        return None

    # Rotate image coords so that FAR_SIDE becomes "top", then sort.
    x, y = quad[:, 0].copy(), quad[:, 1].copy()
    if FAR_SIDE == "bottom":  x, y = -x, -y
    elif FAR_SIDE == "left": x, y = -y,  x
    elif FAR_SIDE == "right": x, y = y, -x
    r = np.stack([x, y], 1)
    tl = quad[np.argmin(r[:, 0] + r[:, 1])]
    tr = quad[np.argmax(r[:, 0] - r[:, 1])]
    br = quad[np.argmax(r[:, 0] + r[:, 1])]
    bl = quad[np.argmin(r[:, 0] - r[:, 1])]
    return np.array([tl, tr, br, bl], np.float32)


def _poly(H, rect):
    """rect = (x0, y0, x1, y1) in tray unit coords -> image polygon."""
    x0, y0, x1, y1 = rect
    sq = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], np.float32)
    return cv2.perspectiveTransform(sq, H)[0].astype(int)


def tray_homography(corners):
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    return cv2.getPerspectiveTransform(unit, corners)


def section_rects():
    """{label: (x0, y0, x1, y1)} in tray unit coords; spanning blocks merged."""
    rows, cols = len(SECTIONS), len(SECTIONS[0])
    cells = {}
    for r in range(rows):
        for c in range(cols):
            cells.setdefault(SECTIONS[r][c], []).append((r, c))
    rects = {}
    for lab, cl in cells.items():
        r0 = min(r for r, _ in cl); r1 = max(r for r, _ in cl) + 1
        c0 = min(c for _, c in cl); c1 = max(c for _, c in cl) + 1
        rects[lab] = (c0/cols, r0/rows, c1/cols, r1/rows)
    return rects


def section_polys(corners):
    """{label: image polygon}."""
    H = tray_homography(corners)
    return {lab: _poly(H, rect) for lab, rect in section_rects().items()}


def region_mask(shape, corners, label):
    """Mask + polygon of one section ("H" or a spot number)."""
    H = tray_homography(corners)
    poly = _poly(H, section_rects()[label])
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask, poly


def home_roi_mask(shape, corners):
    return region_mask(shape, corners, "H")


# ---------- object detection + signature ----------
def object_in_home(hsv, roi_mask):
    """Largest coloured (non-white, non-blue-tape) blob inside HOME."""
    h, s, v = cv2.split(hsv)
    coloured = (s > 70) & (v > 60)
    blue_tape = (h > 95) & (h < 125)
    mask = (coloured & ~blue_tape).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, roi_mask)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= MIN_OBJ_AREA]
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    m = np.zeros(hsv.shape[:2], np.uint8); cv2.drawContours(m, [c], -1, 255, -1)
    return c, m


def signature(hsv, cnt, mask):
    """Feature vector: hue (cos,sin), sat, val, green fraction, rectangularity."""
    px = hsv[mask > 0]
    hue = px[:, 0].astype(float) * 2 * np.pi / 180
    green = ((px[:, 0] > 40) & (px[:, 0] < 85)).mean()
    non_green = px[(px[:, 0] <= 40) | (px[:, 0] >= 85)]
    if len(non_green):
        hue = non_green[:, 0].astype(float) * 2 * np.pi / 180
        sat, val = non_green[:, 1].mean() / 255, non_green[:, 2].mean() / 255
    else:
        sat, val = px[:, 1].mean() / 255, px[:, 2].mean() / 255
    rect = cv2.contourArea(cnt) / max(1, np.prod(cv2.minAreaRect(cnt)[1]))
    return np.array([np.cos(hue).mean(), np.sin(hue).mean(), sat, val, green, rect])


def classify(sig, refs):
    names = list(refs); M = np.array([refs[n] for n in names])
    d = np.linalg.norm(M - sig, axis=1)
    i = int(np.argmin(d))
    return names[i], float(d[i])


# ---------- camera helpers ----------
def grab(cam):
    cam.update()
    return cam.frame


def look_at_home(cam, refs=None, show=True, n_frames=5):
    """Returns (name, dist, poly) or (None, None, poly). Averages several frames."""
    sigs, poly, frame, corners = [], None, None, None
    for _ in range(n_frames):
        frame = grab(cam)
        if frame is None:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        corners = tray_corners(hsv)
        if corners is None:
            continue
        roi, poly = home_roi_mask(frame.shape, corners)
        cnt, m = object_in_home(hsv, roi)
        if cnt is not None:
            sigs.append(signature(hsv, cnt, m))
        time.sleep(0.05)
    name = dist = None
    if sigs and refs:
        name, dist = classify(np.mean(sigs, axis=0), refs)
    if show and frame is not None:
        if corners is not None:
            for num, p in section_polys(corners).items():
                cv2.polylines(frame, [p], True, (255, 0, 0), 1)
                cv2.putText(frame, str(num), tuple(p.mean(axis=0).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            for lab, c in zip(("TL", "TR", "BR", "BL"), corners):
                cv2.circle(frame, tuple(c.astype(int)), 8, (0, 0, 255), 2)
                cv2.putText(frame, lab, tuple(c.astype(int) + [10, -10]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if poly is not None:
            cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
        cv2.putText(frame, corner_status(), (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"{name or 'empty'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("emio sort", frame); cv2.waitKey(1)
    return (name, dist, np.mean(sigs, axis=0) if sigs else None)


# ---------- robot ----------
def wait_settled(motors, timeout=MOVE_WAIT):
    """Wait until motors stop moving (or timeout). Falls back to a fixed sleep."""
    if not SETTLE_POLL:
        time.sleep(timeout); return
    time.sleep(0.15)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            if not any(bool(m) for m in motors.moving):
                time.sleep(0.25); return
        except AttributeError:
            time.sleep(timeout); return
        time.sleep(0.05)


def go(motors, name, dry):
    pose = POSES[name]
    print(f"  -> {name:18s} {np.round(pose, 3).tolist()}")
    if not dry:
        motors.angles = [float(a) for a in pose]
        wait_settled(motors)
    else:
        time.sleep(0.2)


def pick_and_place(motors, spot, dry):
    for name in PICK_SEQ + place_seq(spot):
        go(motors, name, dry)


# ---------- modes ----------
def tune_markers(cam):
    """Sliders for the marker HSV range. Press s to save, q to quit."""
    win = "tune (s=save, q=quit)"
    cv2.namedWindow(win)
    names = ["H lo", "H hi", "S lo", "S hi", "V lo", "V hi", "min area"]
    init  = [MARKER_LO[0], MARKER_HI[0], MARKER_LO[1], MARKER_HI[1],
             MARKER_LO[2], MARKER_HI[2], MIN_MARKER_AREA]
    maxs  = [180, 180, 255, 255, 255, 255, 500]
    for n, v, m in zip(names, init, maxs):
        cv2.createTrackbar(n, win, int(v), m, lambda *_: None)
    print("Adjust until ONLY the 4 markers are white in the mask. s = save.")
    while True:
        frame = grab(cam)
        if frame is None:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = [cv2.getTrackbarPos(n, win) for n in names]
        lo, hi = (v[0], v[2], v[4]), (v[1], v[3], v[5])
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_ok = 0
        for c in cnts:
            if cv2.contourArea(c) >= v[6]:
                n_ok += 1
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"blobs: {n_ok}  lo={lo} hi={hi}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, frame); cv2.imshow("mask", mask)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            return
        if k == ord("s"):
            json.dump({"lo": list(lo), "hi": list(hi), "min_area": v[6]},
                      open(HSV_FILE, "w"))
            print("Saved", HSV_FILE, lo, hi, "min_area", v[6]); return


def calibrate_one_by_one(cam):
    refs = {}
    print("Place each object in HOME, then press the key shown. q = quit.")
    for name in OBJECTS:
        print(f"\nPlace {name.upper()} in HOME and press SPACE")
        while True:
            _, _, sig = look_at_home(cam)
            if cv2.waitKey(50) & 0xFF == 32 and sig is not None:
                refs[name] = sig.tolist(); print("  saved", np.round(sig, 2)); break
    json.dump(refs, open(SIG_FILE, "w"), indent=1)
    print("Saved", SIG_FILE)


def calibrate_layout(cam, n_frames=8):
    """All objects in their final spots at once; one signature per section."""
    spot_to_obj = {v: k for k, v in OBJECT_TO_SPOT.items()}
    print("Put every object in its assigned spot, HOME empty. SPACE to save, q to quit.")
    while True:
        sigs = {name: [] for name in OBJECTS}
        frame = corners = None
        for _ in range(n_frames):
            frame = grab(cam)
            if frame is None:
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            corners = tray_corners(hsv)
            if corners is None:
                continue
            for spot, name in spot_to_obj.items():
                roi, _ = region_mask(frame.shape, corners, spot)
                cnt, m = object_in_home(hsv, roi)
                if cnt is not None:
                    sigs[name].append(signature(hsv, cnt, m))
            time.sleep(0.05)

        found = {n: np.mean(s, axis=0) for n, s in sigs.items() if s}
        missing = [n for n in OBJECTS if n not in found]

        if frame is not None:
            if corners is not None:
                for spot, name in spot_to_obj.items():
                    _, p = region_mask(frame.shape, corners, spot)
                    ok = name in found
                    cv2.polylines(frame, [p], True, (0, 255, 0) if ok else (0, 0, 255), 2)
                    cv2.putText(frame, f"{spot}:{name}", tuple(p[0] + [4, 18]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if corners is None:
                msg = "no tray - " + corner_status()
            else:
                msg = "SPACE = save" if not missing else "missing: " + ", ".join(missing)
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("emio sort", frame)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            return
        if key == 32 and not missing:
            json.dump({n: s.tolist() for n, s in found.items()}, open(SIG_FILE, "w"), indent=1)
            for n, s in found.items():
                print(f"  {n:10s} {np.round(s, 2).tolist()}")
            print("Saved", SIG_FILE); return


def main():
    dry = "--dry" in sys.argv
    do_cal = "--calibrate" in sys.argv
    do_cal_one = "--calibrate-one" in sys.argv
    do_tune = "--tune" in sys.argv

    emio = EmioAPI(multiprocess_camera=False)
    if not emio.connectToEmioDevice():
        print("No Emio found"); return
    emio.camera.track_markers = False
    if not emio.camera.open():
        print("Camera failed"); emio.disconnect(); return

    try:
        if do_tune:
            tune_markers(emio.camera); return
        if do_cal_one:
            calibrate_one_by_one(emio.camera); return
        if do_cal:
            calibrate_layout(emio.camera); return
        if not os.path.exists(SIG_FILE):
            print("Run with --calibrate first"); return
        refs = json.load(open(SIG_FILE))

        if not dry:
            emio.motors.max_velocity = MAX_VELOCITY
        go(emio.motors, "home", dry)
        placed = set()
        for _ in range(len(OBJECTS)):
            name, dist, _ = look_at_home(emio.camera, refs)
            if name is None:
                print("HOME empty — done"); break
            spot = OBJECT_TO_SPOT[name]
            print(f"Detected {name} (d={dist:.2f}) -> spot {spot}")
            if spot in placed:
                print(f"  spot {spot} already filled — check classification"); break
            if dry:
                input("  dry run: move it by hand, then press Enter... ")
                placed.add(spot); continue
            pick_and_place(emio.motors, spot, dry)
            placed.add(spot)
    except KeyboardInterrupt:
        pass
    finally:
        go(emio.motors, "home", dry)
        emio.camera.close(); emio.disconnect(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()