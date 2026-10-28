"""
Emio: sort 6 play-dough objects from the tray's HOME square to their numbered spots.

Objects (from the photos) -> spot numbers (edit OBJECT_TO_SPOT if different):
    durian    -> 2        sushi    -> 1
    burger    -> 6        donut    -> 4
    maki      -> 3        tomato   -> 5

Pipeline per object
  1. Camera looks at the HOME square (centre of the tray, found from the 4 purple corners).
  2. The object there is classified by colour/shape (nearest neighbour against
     a reference set you record once with --calibrate).
  3. The pick-and-place motor sequence for that spot is executed.
  4. Repeat until HOME is empty.

All motion comes from movej.py (same folder): its POSE_SEQUENCE is split into
one segment per spot (each 'hover' ... up to the next 'hover'), and each move is
executed with movej.move_j(). This script only picks which segment to run.

Run (from the emio_4242 folder):
    python emio_sort_objects.py --view          # live view of what the camera sees (no motion)
    python emio_sort_objects.py --tune          # set marker colour with sliders
    python emio_sort_objects.py --calibrate     # once: all objects in their final spots, SPACE
    python emio_sort_objects.py --calibrate-one # alt: one object at a time in HOME
    python emio_sort_objects.py --dry           # detect + classify only (robot only goes to LOOK_POSE)
    python emio_sort_objects.py                 # full run
"""

import sys, json, os, time, math
import numpy as np
import cv2
from emioapi import EmioAPI

# ======================= MOTION: all from movej.py =======================
# movej.py (same folder) owns every motion setting: the poses, the order of
# poses within a pick-and-place, move_j(), the settle wait, and velocity.
# This script only decides WHICH spot's segment to run.
import movej
from movej import POSE_SEQUENCE, move_j

# Optional: add  MAX_VELOCITY = [25, 25, 25, 25]  to movej.py to control it
# there; otherwise we use the value movej.main() uses.
MAX_VELOCITY = getattr(movej, "MAX_VELOCITY", [25, 25, 25, 25])


def _split_segments(seq):
    """
    Split POSE_SEQUENCE into one segment per spot, exactly as written in
    movej.py. A segment starts at each 'hover' and runs until the next 'hover'.
    The spot number is read from the 'poseN_*' names inside it.
    Returns {spot: [(name, angles), ...]}, plus the 'home' pose.
    """
    home = next(a for n, a in seq if n == "home")
    segs, cur = {}, []
    def flush():
        if not cur:
            return
        spots = {int(n[4]) for n, _ in cur if n.startswith("pose") and n[4].isdigit()}
        if len(spots) == 1:
            segs[spots.pop()] = list(cur)
    for name, ang in seq:
        if name == "home":
            continue
        if name == "hover":
            flush(); cur = []
        cur.append((name, ang))
    flush()
    return home, segs


HOME_POSE, SEGMENTS = _split_segments(POSE_SEQUENCE)
print(f"movej.py loaded from: {movej.__file__}  (modified "
      f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(movej.__file__)))})")
print(f"  {len(POSE_SEQUENCE)} poses, spots {sorted(SEGMENTS)}, hover = {dict(POSE_SEQUENCE).get('hover', '?')}")
POSE_BY_NAME = {n: a for n, a in POSE_SEQUENCE}
# Pose (name from movej.py) the robot goes to for the FIRST look (and for
# tune/calibrate/dry) so the gripper is out of the camera's way.
LOOK_POSE = "hover"
# Pose to hold when the script ends (every mode). None = stay where it is.
END_POSE = "hover"
# After each pick-and-place: None = stay where the segment ended
# (poseN_open_hover, gripper off to the side) and look from there;
# or a pose name from movej.py to move to before looking.
AFTER_PLACE_LOOK = None
# After a segment, go back to 'hover' so the camera has a clear view of HOME.
HOVER_POSE = next(a for n, a in POSE_SEQUENCE if n == "hover")
# =========================================================================

OBJECT_TO_SPOT = {"durian": 2, "sushi": 1, "burger": 6,
                  "donut": 4, "maki": 3, "tomato": 5}
OBJECTS = list(OBJECT_TO_SPOT)

# Corner MARKERS are PURPLE. OpenCV HSV: H 0-180, S 0-255, V 0-255.
# Run  --tune  to set these with sliders; the result is saved to marker_hsv.json
# and loaded automatically on later runs (overrides the values below).
MARKER_LO, MARKER_HI = (125, 60, 50), (160, 255, 255)
MIN_MARKER_AREA = 25
MIN_TRAY_FRAC   = 0.005        # tray must cover >= this fraction of the frame
SHOW_MASK = True               # extra window showing the marker mask
HSV_FILE  = "marker_hsv.json"
if os.path.exists(HSV_FILE):
    _h = json.load(open(HSV_FILE))
    MARKER_LO, MARKER_HI = tuple(_h["lo"]), tuple(_h["hi"])
    MIN_MARKER_AREA = _h.get("min_area", MIN_MARKER_AREA)
    print(f"marker HSV from {HSV_FILE}: lo={MARKER_LO} hi={MARKER_HI} min_area={MIN_MARKER_AREA}")
else:
    print(f"WARNING: no {HSV_FILE} - using default lo={MARKER_LO} hi={MARKER_HI}. "
          "Run --tune and press s.")
MIN_OBJ_AREA = 300
DARK_V = 70                    # pixels darker than this count as object (black)
SIG_FILE = "object_signatures.json"


# ---------- tray geometry ----------
# Tray as a 3x3 grid of blocks (rows top->bottom, cols left->right).
# A section may span several blocks. "H" = home square.
SECTIONS = [[2, 2, 1],
            [6, "H", 1],
            [3, 5, 4]]
# Orientation:
#  - With 3 markers: the corner WITHOUT a marker is MISSING_CORNER (BR = next
#    to section 4). This is the normal setup.
#  - With all 4 markers visible: fall back to FAR_SIDE = where the layout's top
#    edge (sections 2/1) appears in the image: "top"/"right"/"bottom"/"left"
#    or an angle in degrees (0 = up, 90 = right).
MISSING_CORNER = "BR"
FAR_SIDE = "top"
# Final correction: number of 90-degree CLOCKWISE turns to apply to the grid
# overlay (0-3). Use this if the numbers are drawn rotated on the tray.
GRID_ROTATE_CW = 1


RELAX_LEVELS = (1.0, 0.7, 0.5, 0.3)   # S/V lower bounds scaled by these, in turn


def _marker_mask(hsv, relax):
    lo = (MARKER_LO[0], int(MARKER_LO[1] * relax), int(MARKER_LO[2] * relax))
    mask = cv2.inRange(hsv, np.array(lo), np.array(MARKER_HI))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _blobs_of(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a >= MIN_MARKER_AREA:
            m = cv2.moments(c); blobs.append((a, m["m10"] / m["m00"], m["m01"] / m["m00"]))
    blobs.sort(reverse=True)
    return blobs


def marker_blobs(hsv):
    """
    Marker detection that tolerates exposure drift: start from the tuned HSV
    range; if fewer than 3 (or more than 8) blobs, relax the S/V lower bounds
    step by step and take the first level that gives a sensible count.
    """
    best, best_mask, best_lvl = [], None, RELAX_LEVELS[0]
    for lvl in RELAX_LEVELS:
        mask = _marker_mask(hsv, lvl)
        blobs = _blobs_of(mask)
        if 3 <= len(blobs) <= 8:
            best, best_mask, best_lvl = blobs, mask, lvl
            break
        if best_mask is None or abs(len(blobs) - 3.5) < abs(len(best) - 3.5):
            best, best_mask, best_lvl = blobs, mask, lvl
    marker_blobs.level = best_lvl
    if SHOW_MASK:
        cv2.imshow("marker mask", best_mask)
    marker_blobs.last = [(int(u), int(v), int(a)) for a, u, v in best]
    return np.array([[u, v] for _, u, v in best], np.float32)


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


_last_diag = [0.0]
def diag_print(corners):
    """Once a second: what the marker detector sees (terminal)."""
    if time.time() - _last_diag[0] < 1.0:
        return
    _last_diag[0] = time.time()
    print(f"[diag] {corner_status()} | blobs (x,y,area): {getattr(marker_blobs, 'last', [])[:8]}"
          + ("" if corners is None else f" | corners ok: {corners.astype(int).tolist()}"))


def corner_status():
    n = getattr(tray_corners, "last_n", 0)
    lvl = getattr(marker_blobs, "level", 1.0)
    tag = "" if lvl == 1.0 else f" relax={lvl}"
    if n == 99: return "markers: (cached)"
    if n >= 3:  return f"markers: {n}{tag}"
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
    missing = None
    if len(pts) >= 4:
        quad = _largest_quad(pts[:8])
    else:                                              # 3 markers: reconstruct
        d = [np.linalg.norm(pts[(i+1) % 3] - pts[(i+2) % 3]) for i in range(3)]
        mid = int(np.argmax(d))
        missing = pts[(mid+1) % 3] + pts[(mid+2) % 3] - pts[mid]
        quad = np.vstack([pts, missing[None]])

    # Reject degenerate quads (collinear noise blobs -> a "line").
    area = cv2.contourArea(cv2.convexHull(quad.astype(np.float32)))
    if area < MIN_TRAY_FRAC * hsv.shape[0] * hsv.shape[1]:
        tray_corners.last_n = -len(pts)                # negative = degenerate
        return None

    ordered = order_corners(quad, missing)
    return np.roll(ordered, -GRID_ROTATE_CW, axis=0)   # rotate grid CW


def order_corners(quad, missing=None):
    """
    Rotation-proof ordering. Sort the 4 corners clockwise around their centre,
    then rotate the list so the FAR edge (sections 2/1) comes first:
    result = [TL, TR, BR, BL] of the SECTIONS layout.
    FAR_SIDE: "top"/"right"/"bottom"/"left", or an angle in degrees
    (0 = up in the image, 90 = right, 180 = down, 270 = left).
    """
    quad = np.asarray(quad, np.float32)
    ctr = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - ctr[1], quad[:, 0] - ctr[0])
    quad = quad[np.argsort(ang)]                       # clockwise (y is down)

    # Preferred: the corner with NO marker is MISSING_CORNER of the layout.
    if missing is not None:
        k = int(np.argmin(np.linalg.norm(quad - missing, axis=1)))
        want = ["TL", "TR", "BR", "BL"].index(MISSING_CORNER)
        return np.roll(quad, want - k, axis=0)         # [TL, TR, BR, BL]

    # Fallback (all 4 markers visible): far edge from FAR_SIDE.
    deg = {"top": 0, "right": 90, "bottom": 180, "left": 270}.get(FAR_SIDE, FAR_SIDE)
    t = np.deg2rad(float(deg))
    far_dir = np.array([np.sin(t), -np.cos(t)])        # image vector toward far side
    mids = (quad + np.roll(quad, -1, axis=0)) / 2      # edge i: quad[i] -> quad[i+1]
    k = int(np.argmax((mids - ctr) @ far_dir))         # the far edge
    return np.roll(quad, -k, axis=0)                   # [TL, TR, BR, BL]


CORNER_HOLD_S = 3.0            # reuse last good corners for this long if lost
_corner_cache = {"c": None, "t": 0.0}
def tray_corners_cached(hsv):
    c = tray_corners(hsv)
    if c is not None:
        _corner_cache["c"], _corner_cache["t"] = c, time.time()
        return c
    if _corner_cache["c"] is not None and time.time() - _corner_cache["t"] < CORNER_HOLD_S:
        tray_corners.last_n = 99                        # flag: cached
        return _corner_cache["c"]
    return None


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
    dark = v < DARK_V                       # black objects (nori / maki)
    blue_tape = (h > 95) & (h < 125) & (s > 70)
    mask = ((coloured | dark) & ~blue_tape).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, roi_mask)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_in_home.best_area = max((cv2.contourArea(c) for c in cnts), default=0)
    cnts = [c for c in cnts if cv2.contourArea(c) >= MIN_OBJ_AREA]
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    m = np.zeros(hsv.shape[:2], np.uint8); cv2.drawContours(m, [c], -1, 255, -1)
    return c, m


HUE_BINS = 18
# Feature weights: [hue histogram, mean sat, dark fraction, solidity, rectangularity, aspect]
FEAT_W = np.array([4.0] * HUE_BINS + [1.0, 2.0, 1.5, 0.7, 0.5])


def signature(hsv, cnt, mask):
    """
    Lighting-tolerant descriptor:
      - hue histogram (12 bins, only reasonably saturated pixels)
      - mean saturation, fraction of dark pixels (black maki -> high)
      - solidity  = area / convex-hull area   (donut ring -> low)
      - rectangularity = area / min-area-rect (bread -> high, tomato -> ~0.78)
      - aspect ratio of min-area-rect (eggplant -> elongated)
    """
    px = hsv[mask > 0]
    dark_frac = (px[:, 2] < DARK_V).mean()
    sat_sel = (px[:, 1] > 60) & (px[:, 2] >= DARK_V)
    if sat_sel.sum() > 0.1 * len(px):
        hist, _ = np.histogram(px[sat_sel, 0], bins=HUE_BINS, range=(0, 180))
        hist = hist.astype(float)
        hist = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))   # soft bins
        hist = hist / max(1e-6, hist.sum())
    else:
        hist = np.full(HUE_BINS, 1.0 / HUE_BINS)   # colourless object: flat
    sat = px[:, 1].mean() / 255
    area = cv2.contourArea(cnt)
    hull = cv2.contourArea(cv2.convexHull(cnt))
    solidity = area / max(1, hull)
    (w, h) = cv2.minAreaRect(cnt)[1]
    rect = area / max(1, w * h)
    aspect = min(w, h) / max(1, max(w, h))
    return np.concatenate([hist, [sat, dark_frac, solidity, rect, aspect]])


def classify(sig, refs):
    """Weighted nearest neighbour. Returns (name, dist, runner_up, dist2)."""
    names = list(refs); M = np.array([refs[n] for n in names])
    d = np.linalg.norm((M - sig) * FEAT_W, axis=1)
    order = np.argsort(d)
    i, j = order[0], (order[1] if len(order) > 1 else order[0])
    classify.runner_up = (names[j], float(d[j]))
    return names[i], float(d[i])


# ---------- camera helpers ----------
def grab(cam):
    cam.update()
    return cam.frame


def look_at_home(cam, refs=None, show=True, n_frames=8):
    """Returns (name, dist, poly) or (None, None, poly). Averages several frames."""
    sigs, poly, frame, corners = [], None, None, None
    tray_seen, best_area = 0, 0
    for _ in range(n_frames):
        frame = grab(cam)
        if frame is None:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        corners = tray_corners_cached(hsv)
        diag_print(corners)
        if corners is None:
            continue
        tray_seen += 1
        roi, poly = home_roi_mask(frame.shape, corners)
        cnt, m = object_in_home(hsv, roi)
        best_area = max(best_area, getattr(object_in_home, "best_area", 0))
        if cnt is not None:
            sigs.append(signature(hsv, cnt, m))
        time.sleep(0.05)
    name = dist = None
    if sigs and refs:
        name, dist = classify(np.mean(sigs, axis=0), refs)
    if name is None:
        reason = ("NO TRAY (" + corner_status() + ")" if tray_seen == 0 else
                  f"HOME empty (largest blob {best_area:.0f}px, need {MIN_OBJ_AREA})")
    else:
        reason = name
    look_at_home.reason = reason
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
        cv2.putText(frame, reason, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("emio sort", frame); cv2.waitKey(1)
    return (name, dist, np.mean(sigs, axis=0) if sigs else None)


# ---------- robot ----------
def go(emio, name, angles, dry):
    """One move, executed by movej.move_j (its own wait/settle logic)."""
    print(f"  -> {name:18s} {list(angles)}")
    if dry:
        time.sleep(0.2)
    else:
        move_j(angles, emio)


def pick_and_place(emio, spot, dry):
    """Run spot N's segment exactly as it appears in movej.POSE_SEQUENCE."""
    if spot not in SEGMENTS:
        raise KeyError(f"movej.py has no pose{spot}_* segment")
    for name, angles in SEGMENTS[spot]:
        go(emio, name, angles, dry)


# ---------- modes ----------
def refs_if_any():
    return json.load(open(SIG_FILE)) if os.path.exists(SIG_FILE) else None


LOOK_TIMEOUT_S = 10.0


def look_with_retry(cam, refs):
    """Keep looking until an object is classified, HOME is confirmed empty
    (tray seen), or LOOK_TIMEOUT_S passes. Shows the live window meanwhile."""
    t0 = time.time()
    while True:
        name, dist, sig = look_at_home(cam, refs)
        if name is not None or "HOME empty" in look_at_home.reason:
            return name, dist, sig
        if time.time() - t0 > LOOK_TIMEOUT_S:
            return name, dist, sig
        cv2.waitKey(30)


def view(cam, refs=None):
    """Live detection view, no robot motion. q to quit."""
    print("Live view. q to quit.")
    while True:
        look_at_home(cam, refs, n_frames=2)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            return


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
            corners = tray_corners_cached(hsv)
            diag_print(corners)
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
    do_view = "--view" in sys.argv

    emio = EmioAPI(multiprocess_camera=False)
    if not emio.connectToEmioDevice():
        print("No Emio found"); return
    emio.camera.track_markers = False
    if not emio.camera.open():
        print("Camera failed"); emio.disconnect(); return
    for _ in range(20):                      # let auto-exposure settle
        grab(emio.camera); time.sleep(0.05)

    try:
        emio.motors.max_velocity = MAX_VELOCITY
        print(f"Moving to LOOK_POSE '{LOOK_POSE}' so the camera has a clear view.")
        go(emio, LOOK_POSE, POSE_BY_NAME[LOOK_POSE], dry=False)  # always, even in --dry
        if do_view:
            view(emio.camera, refs_if_any()); return
        if do_tune:
            tune_markers(emio.camera); return
        if do_cal_one:
            calibrate_one_by_one(emio.camera); return
        if do_cal:
            calibrate_layout(emio.camera); return
        if not os.path.exists(SIG_FILE):
            print("Run with --calibrate first"); return
        refs = json.load(open(SIG_FILE))

        placed = set()
        for i in range(len(OBJECTS)):
            if i > 0 and AFTER_PLACE_LOOK is not None:
                go(emio, AFTER_PLACE_LOOK, POSE_BY_NAME[AFTER_PLACE_LOOK], dry=False)
            name, dist, _ = look_with_retry(emio.camera, refs)
            if name is None:
                print("Stopping:", look_at_home.reason)
                if "NO TRAY" in look_at_home.reason:
                    print("  -> the camera can't see the markers from this pose. "
                          "Set LOOK_POSE to a pose in movej.py where the tray is visible.")
                break
            spot = OBJECT_TO_SPOT[name]
            ru = getattr(classify, "runner_up", ("?", 0))
            print(f"Detected {name} (d={dist:.2f}; next: {ru[0]} d={ru[1]:.2f}) -> spot {spot}")
            if spot in placed:
                print(f"  spot {spot} already filled — check classification"); break
            if dry:
                input("  dry run: move it by hand, then press Enter... ")
                placed.add(spot); continue
            pick_and_place(emio, spot, dry)
            placed.add(spot)
    except KeyboardInterrupt:
        pass
    finally:
        if END_POSE is not None:
            go(emio, END_POSE, POSE_BY_NAME[END_POSE], dry=False)
        emio.camera.close(); emio.disconnect(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()