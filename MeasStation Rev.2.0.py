"""
PT MeasStation Rev.2.0
=======================
User-facing UI for 4-corner glass measurement.
Operator enters LOT + OPT ID, then presses [PROCESSING] 4 times.
Each press measures one corner in order:
  1. LEFT TOP  2. RIGHT TOP  3. RIGHT BOT  4. LEFT BOT

Results are saved automatically after the 4th measurement:
  OUTPUT/{lot}_{datetime}.csv
  OUTPUT/{lot}_{datetime}.jpg
"""

import threading
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import csv
from datetime import datetime
import os


try:
    from GetFrame import Camera
    HAS_GETFRAME = True
except ImportError:
    HAS_GETFRAME = False

# -------------------------------------------------------
#  DEFAULT CONFIG
# -------------------------------------------------------
DEFAULT_CONFIG = {
    "thresh_value":   180,
    "morph_size":     5,
    "min_area":       500000,
    "max_area":       900000,
    "target_w":       900,   # A  (width of rectangle to find)
    "target_h":       900,    # B  (height of rectangle to find)
    "size_tolerance": 90,    # ±px tolerance for A×B matching
    "transition_gap": 1,     # min pixel gap between transitions
    "px_per_um":      1.64,   # pixels per 1 µm  (0 = disabled)
}

# -------------------------------------------------------
#  GEOMETRY / SAMPLING HELPERS
# -------------------------------------------------------
def sample_line_pixels(gray, x0, y0, x1, y1):
    length = int(np.hypot(x1 - x0, y1 - y0))
    if length == 0:
        return np.array([]), []
    xs = np.linspace(x0, x1, length + 1)
    ys = np.linspace(y0, y1, length + 1)
    h, w = gray.shape[:2]
    xi = np.clip(xs, 0, w - 1).astype(int)
    yi = np.clip(ys, 0, h - 1).astype(int)
    return gray[yi, xi].astype(float), list(zip(xs.tolist(), ys.tolist()))


def find_transitions(values, coords, gap=3):
    transitions = []
    for i in range(1, len(values)):
        prev, curr = values[i - 1], values[i]
        if   prev > 128 >= curr:
            transitions.append({"idx": i, "x": coords[i][0], "y": coords[i][1], "dir": "W2B"})
        elif prev <= 128 < curr:
            transitions.append({"idx": i, "x": coords[i][0], "y": coords[i][1], "dir": "B2W"})
    merged = []
    for t in transitions:
        if merged and abs(t["idx"] - merged[-1]["idx"]) < gap:
            continue
        merged.append(t)
    return merged


def px_to_um(px_dist, px_per_um):
    return None if px_per_um <= 0 else px_dist * px_per_um


def gap_label(dist_px, px_per_um):
    um = px_to_um(dist_px, px_per_um)
    return f"{dist_px:.1f} px" if um is None else f"{dist_px:.1f} px  ({um:.2f} µm)"


def compute_gaps(trans_list):
    gaps = []
    if len(trans_list) >= 3:
        p2, p3 = trans_list[1], trans_list[2]
        d = np.hypot(p3["x"] - p2["x"], p3["y"] - p2["y"])
        gaps.append({"from": 1, "to": 2, "px": d,
                     "x1": p2["x"], "y1": p2["y"], "x2": p3["x"], "y2": p3["y"]})
    if len(trans_list) >= 5:
        p4, p5 = trans_list[3], trans_list[4]
        d = np.hypot(p5["x"] - p4["x"], p5["y"] - p4["y"])
        gaps.append({"from": 3, "to": 4, "px": d,
                     "x1": p4["x"], "y1": p4["y"], "x2": p5["x"], "y2": p5["y"]})
    return gaps


# -------------------------------------------------------
#  CIRCLE FITTING HELPERS
# -------------------------------------------------------

def fit_circle_3pt(p1, p2, p3):
    """Exact circumscribed circle through 3 points. Returns (cx, cy, R) or None."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-10:
        return None  # collinear
    ux = ((ax**2 + ay**2) * (by - cy) +
          (bx**2 + by**2) * (cy - ay) +
          (cx**2 + cy**2) * (ay - by)) / D
    uy = ((ax**2 + ay**2) * (cx - bx) +
          (bx**2 + by**2) * (ax - cx) +
          (cx**2 + cy**2) * (bx - ax)) / D
    R = np.hypot(ax - ux, ay - uy)
    return ux, uy, R


def fit_circle_lstsq(pts):
    """
    Algebraic least-squares circle fit (no scipy).
    Expands (x-cx)^2 + (y-cy)^2 = R^2 to:
      2x·cx + 2y·cy + (R^2 - cx^2 - cy^2) = x^2 + y^2
    Solves for [cx, cy, c] where c = R^2 - cx^2 - cy^2.
    Returns (cx, cy, R) or None on failure.
    """
    pts = np.array(pts, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(pts))])
    b = x**2 + y**2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = result
    r_sq = c + cx**2 + cy**2
    if r_sq <= 0:
        return None
    return cx, cy, float(np.sqrt(r_sq))


def fit_circle(circle_pts):
    """
    Dispatch to exact (3-pt) or least-squares fit depending on point count.
    Returns (cx, cy, R) or None.
    """
    n = len(circle_pts)
    if n < 2:
        return None
    coords = [(p["x"], p["y"]) for p in circle_pts]
    if n == 3:
        result = fit_circle_3pt(*coords)
        if result is None:
            result = fit_circle_lstsq(coords)
    else:
        result = fit_circle_lstsq(coords)
    return result


def measure_gap_to_inner(circle_center, R, dx, dy, inner_box_pts, max_dist, ppu):
    """
    Cast a ray from circle boundary outward along (dx, dy).
    Returns (gap_px, edge_x, edge_y, circ_edge_x, circ_edge_y).
    gap_px > 0  ? circle edge is inside inner rect
    gap_px < 0  ? circle edge is outside inner rect
    Returns None if no crossing found.
    """
    ccx, ccy = circle_center
    # Point on circle boundary in this direction
    cex = ccx + R * dx
    cey = ccy + R * dy

    # Check whether the circle edge point is already inside or outside inner rect
    test_pt = (float(cex), float(cey))
    inside_start = cv2.pointPolygonTest(inner_box_pts, test_pt, False) >= 0

    step = 0.5
    steps = int(max_dist / step) + 1
    found_edge_x, found_edge_y = None, None

    if inside_start:
        # Walk outward from circ_edge until we leave the inner rect
        for i in range(1, steps):
            tx = cex + dx * i * step
            ty = cey + dy * i * step
            if cv2.pointPolygonTest(inner_box_pts, (float(tx), float(ty)), False) < 0:
                found_edge_x, found_edge_y = tx, ty
                break
        if found_edge_x is None:
            return None
        gap_px = np.hypot(found_edge_x - cex, found_edge_y - cey)
    else:
        # Walk outward from circ_edge — it's already outside; gap is negative
        # Walk inward (opposite direction) to find where inner rect starts
        for i in range(1, steps):
            tx = cex - dx * i * step
            ty = cey - dy * i * step
            if cv2.pointPolygonTest(inner_box_pts, (float(tx), float(ty)), False) >= 0:
                found_edge_x, found_edge_y = tx, ty
                break
        if found_edge_x is None:
            return None
        gap_px = -np.hypot(found_edge_x - cex, found_edge_y - cey)

    gap_um = px_to_um(abs(gap_px), ppu)
    if gap_px < 0 and gap_um is not None:
        gap_um = -gap_um

    return gap_px, found_edge_x, found_edge_y, cex, cey


CORNER_ORDER = ["LEFT TOP", "RIGHT TOP", "RIGHT BOT", "LEFT BOT"]
OUTPUT_DIR   = "OUTPUT"


def _get_circle_gaps(results):
    """Extract H+, H-, V+, V- (px & um) from first result's circle data."""
    if not results:
        return {k: {"px": None, "um": None} for k in ["H+", "H-", "V+", "V-"]}
    circ = results[0].get("circle") or {}
    gaps = circ.get("gaps", {})
    out = {}
    for label in ["H+", "H-", "V+", "V-"]:
        entry = gaps.get(label, {})
        out[label] = {"px": entry.get("px"), "um": entry.get("um")}
    return out


def save_job_to_csv(lot, opt_id, corners_data, file_path):
    """Save all 4 corners to a single CSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "lot", "opt_id", "corner",
            "H+_px", "H+_um", "H-_px", "H-_um",
            "V+_px", "V+_um", "V-_px", "V-_um",
        ])
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for corner_name, gap_data in corners_data.items():
            def g(label, field):
                return gap_data.get(label, {}).get(field)
            writer.writerow([
                ts, lot, opt_id, corner_name,
                g("H+","px"), g("H+","um"),
                g("H-","px"), g("H-","um"),
                g("V+","px"), g("V+","um"),
                g("V-","px"), g("V-","um"),
            ])


def save_job_image(frames_dict, file_path):
    """
    Compose a 2×2 grid image from 4 corner frames and save as JPG.
    frames_dict: { corner_name: cv2_frame }
    Order: LEFT TOP | RIGHT TOP
           LEFT BOT | RIGHT BOT
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    order = [["LEFT TOP", "RIGHT TOP"], ["LEFT BOT", "RIGHT BOT"]]
    # Determine tile size from first available frame
    sample = next(iter(frames_dict.values()), None)
    if sample is None:
        return
    th, tw = sample.shape[:2]
    # Scale down so grid fits reasonably
    scale = min(1.0, 1200 / (tw * 2), 900 / (th * 2))
    nw, nh = int(tw * scale), int(th * scale)

    grid = np.zeros((nh * 2, nw * 2, 3), dtype=np.uint8)
    label_colors = {
        "LEFT TOP":  (0, 200, 255),
        "RIGHT TOP": (0, 255, 150),
        "RIGHT BOT": (255, 180, 0),
        "LEFT BOT":  (180, 100, 255),
    }
    for row_i, row in enumerate(order):
        for col_i, name in enumerate(row):
            frame = frames_dict.get(name)
            if frame is None:
                tile = np.zeros((nh, nw, 3), dtype=np.uint8)
            else:
                tile = cv2.resize(frame, (nw, nh))
            # Corner label overlay
            color = label_colors.get(name, (255,255,255))
            cv2.rectangle(tile, (0,0), (nw-1, 28), (0,0,0), -1)
            cv2.putText(tile, name, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            y0, x0 = row_i * nh, col_i * nw
            grid[y0:y0+nh, x0:x0+nw] = tile

    cv2.imwrite(file_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])


# -------------------------------------------------------
#  ALIGNMENT ANALYSIS ENGINE
# -------------------------------------------------------
def compute_alignment(corners_gaps):
    """
    Compute X offset, Y offset, and Rotation from 4-corner gap data.

    Convention (all values in µm):
      H+ = RIGHT gap   H- = LEFT gap
      V+ = TOP gap     V- = BOT gap

    X offset  > 0  → circle shifted RIGHT  → move LEFT
    Y offset  > 0  → circle shifted UP     → move DOWN
    Rotation  > 0  → TOP tilted RIGHT (clockwise when viewed from front)

    Returns dict with keys:
      x_offset, y_offset, rotation  (µm)
      adj_right, adj_bl, adj_br     (µm, how much each adjuster moves)
      valid                          (bool)
      details                        (per-corner H/V balance)
    """
    required = ["LEFT TOP", "RIGHT TOP", "RIGHT BOT", "LEFT BOT"]
    # Check all corners have data
    for c in required:
        if c not in corners_gaps:
            return {"valid": False}
        for key in ["H+", "H-", "V+", "V-"]:
            if corners_gaps[c].get(key, {}).get("um") is None:
                return {"valid": False}

    def um(corner, key):
        return corners_gaps[corner][key]["um"]

    # ── X offset ──────────────────────────────────────
    # H+ (right gap) large → circle near left side → shifted LEFT
    # mean(H+) - mean(H-) > 0 → right gap bigger → circle shifted LEFT
    h_plus_avg  = sum(um(c, "H+") for c in required) / 4
    h_minus_avg = sum(um(c, "H-") for c in required) / 4
    # x_offset: positive = circle shifted RIGHT (H- bigger than H+)
    x_offset = (h_minus_avg - h_plus_avg) / 2

    # ── Y offset ──────────────────────────────────────
    v_plus_avg  = sum(um(c, "V+") for c in required) / 4
    v_minus_avg = sum(um(c, "V-") for c in required) / 4
    # y_offset: positive = circle shifted UP (V- bigger than V+)
    y_offset = (v_minus_avg - v_plus_avg) / 2

    # ── Rotation ──────────────────────────────────────
    # Compare H balance (H- - H+) between TOP corners vs BOT corners
    # If TOP has more H- → circle top leans LEFT → rotate clockwise
    top_h_balance = ((um("LEFT TOP","H-")  - um("LEFT TOP","H+")) +
                     (um("RIGHT TOP","H-") - um("RIGHT TOP","H+"))) / 2
    bot_h_balance = ((um("LEFT BOT","H-")  - um("LEFT BOT","H+")) +
                     (um("RIGHT BOT","H-") - um("RIGHT BOT","H+"))) / 2
    # rotation > 0 → top leans LEFT relative to bottom → need CW rotation
    rotation = (top_h_balance - bot_h_balance) / 2

    # ── Adjuster recommendations ───────────────────────
    # RIGHT adjuster → X axis only
    #   positive adj_right = push right adjuster IN = move glass LEFT
    adj_right = -x_offset   # move opposite to offset

    # BL + BR equal → Y shift
    #   positive adj_bl/br = raise both = move glass UP
    adj_y = -y_offset       # move opposite to offset

    # BL vs BR differential → rotation
    #   rotation > 0 (top leans left) → raise BL, lower BR
    adj_rot = rotation / 2  # split equally between BL and BR

    adj_bl = adj_y + adj_rot   # positive = raise
    adj_br = adj_y - adj_rot   # positive = raise

    # ── Per-corner details ─────────────────────────────
    details = {}
    for c in required:
        h_bal = um(c, "H-") - um(c, "H+")   # > 0 → leans left
        v_bal = um(c, "V-") - um(c, "V+")   # > 0 → leans down
        details[c] = {"h_balance": h_bal, "v_balance": v_bal}

    return {
        "valid":      True,
        "x_offset":   round(x_offset,  2),
        "y_offset":   round(y_offset,  2),
        "rotation":   round(rotation,  2),
        "adj_right":  round(adj_right, 2),
        "adj_bl":     round(adj_bl,    2),
        "adj_br":     round(adj_br,    2),
        "details":    details,
    }


def scan_diagonal(gray, cx, cy, hl, hs, lx, ly, sx, sy, gap):
    """
    Scan along both diagonals of the rectangle through center.
    D1: top-left → bottom-right  (lx+sx direction)
    D2: top-right → bottom-left  (lx-sx direction)
    Returns (d1_trans, d2_trans)
    """
    length = np.hypot(hl, hs)

    # D1: (lx+sx) normalized
    d1x = lx + sx;  d1y = ly + sy
    n = np.hypot(d1x, d1y)
    if n > 0: d1x /= n; d1y /= n

    # บังคับ D1 start บนซ้าย (y เล็กกว่า หรือ x เล็กกว่า)
    if (cy - d1y*length) > (cy + d1y*length):
        d1x, d1y = -d1x, -d1y

    d1_vals, d1_coords = sample_line_pixels(
        gray,
        cx - d1x*length, cy - d1y*length,
        cx + d1x*length, cy + d1y*length
    )
    d1_trans = find_transitions(d1_vals, d1_coords, gap)

    # D2: (lx-sx) normalized
    d2x = lx - sx;  d2y = ly - sy
    n = np.hypot(d2x, d2y)
    if n > 0: d2x /= n; d2y /= n

    # บังคับ D2 start บนขวา (y เล็กกว่า)
    if (cy - d2y*length) > (cy + d2y*length):
        d2x, d2y = -d2x, -d2y

    d2_vals, d2_coords = sample_line_pixels(
        gray,
        cx - d2x*length, cy - d2y*length,
        cx + d2x*length, cy + d2y*length
    )
    d2_trans = find_transitions(d2_vals, d2_coords, gap)

    return d1_trans, d2_trans

# -------------------------------------------------------
#  DETECTION CORE
# -------------------------------------------------------
def detect_on_frame(frame, cfg):
    h_img, w_img = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()

    _, thresh = cv2.threshold(gray, cfg["thresh_value"], 255, cv2.THRESH_BINARY_INV)
    ks = max(1, cfg["morph_size"])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    results = []

    tol, tw, th = cfg["size_tolerance"], cfg["target_w"], cfg["target_h"]
    gap, ppu    = cfg["transition_gap"], cfg["px_per_um"]

    C = dict(
        rect    = (0, 220, 255),
        line    = (255, 200, 0),
        start   = (0, 255, 128),
        end     = (0, 100, 255),
        w2b     = (0, 0, 255),
        b2w     = (255, 100, 0),
        ctr     = (255, 255, 255),
        gap     = (0, 0, 255),
        # New keys for circle
        circle  = (0, 200, 255),
        circ_pt = (255, 0, 200),
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (cfg["min_area"] <= area <= cfg["max_area"]):
            continue

        rect = cv2.minAreaRect(cnt)
        center_f, (rw, rh), angle = rect
        if rw < rh:
            rw, rh, angle = rh, rw, angle + 90

        if not ((abs(rw-tw) <= tol and abs(rh-th) <= tol) or
                (abs(rw-th) <= tol and abs(rh-tw) <= tol)):
            continue

        cx, cy = float(center_f[0]), float(center_f[1])
        box = cv2.boxPoints(rect).astype(int)
        #cv2.drawContours(out, [box], 0, C["rect"], 2)

        rad = np.deg2rad(angle)
        lx, ly =  np.cos(rad),  np.sin(rad)
        sx, sy = -np.sin(rad),  np.cos(rad)
        hl, hs = rw / 2, rh / 2

        def scan(x0, y0, x1, y1, color_start, color_end):
            #cv2.line(out, (int(x0),int(y0)), (int(x1),int(y1)), C["line"], 1)
            #cv2.circle(out, (int(x0),int(y0)), 6, color_start, -1)
            #cv2.circle(out, (int(x1),int(y1)), 6, color_end,   -1)
            vals, coords = sample_line_pixels(gray, x0, y0, x1, y1)
            trans = find_transitions(vals, coords, gap)
            for pt in trans:
                px, py = int(pt["x"]), int(pt["y"])
                col = C["w2b"] if pt["dir"] == "W2B" else C["b2w"]
                #cv2.drawMarker(out, (px,py), col, cv2.MARKER_CROSS, 14, 2)
                #cv2.putText(out, f"({px},{py})", (px+4, py-6),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
            return trans, compute_gaps(trans)

        # ═══ ส่วนที่ 1: แทนที่ 4 บรรทัด hx1..v_trans ═══
        # บังคับ H-scan: start ซ้ายเสมอ
        hx1, hy1 = cx - lx*hl, cy - ly*hl
        hx2, hy2 = cx + lx*hl, cy + ly*hl
        if hx1 > hx2:                          # swap ถ้า start อยู่ขวา
            hx1, hy1, hx2, hy2 = hx2, hy2, hx1, hy1

        # บังคับ V-scan: start บนเสมอ
        vx1, vy1 = cx - sx*hs, cy - sy*hs
        vx2, vy2 = cx + sx*hs, cy + sy*hs
        if vy1 > vy2:                          # swap ถ้า start อยู่ล่าง
            vx1, vy1, vx2, vy2 = vx2, vy2, vx1, vy1

        h_trans, h_gaps = scan(hx1, hy1, hx2, hy2, C["start"], C["end"])
        v_trans, v_gaps = scan(vx1, vy1, vx2, vy2, C["start"], C["end"])

        # gap overlays
        for g, pfx in [(g, f"H{g['from']+1}?{g['to']+1}:") for g in h_gaps] + \
                      [(g, f"V{g['from']+1}?{g['to']+1}:") for g in v_gaps]:
            x1,y1 = int(g["x1"]),int(g["y1"])
            x2,y2 = int(g["x2"]),int(g["y2"])
            mx,my = (x1+x2)//2,(y1+y2)//2
            #cv2.line(out,(x1,y1),(x2,y2),C["gap"],2)
            um = px_to_um(g["px"], ppu)
            #cv2.putText(out, f"{pfx}{g['px']:.1f}px",
                        #(mx+4,my-4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, C["gap"],1,cv2.LINE_AA)
            #if um is not None:
                #cv2.putText(out, f"={um:.2f}µm",
                            #(mx+4,my+12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, C["gap"],1,cv2.LINE_AA)

        #cv2.drawMarker(out,(int(cx),int(cy)),C["ctr"],cv2.MARKER_TILTED_CROSS,16,2)
        #cv2.putText(out, f"C({int(cx)},{int(cy)})", (int(cx)+8,int(cy)-8),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["ctr"], 1, cv2.LINE_AA)

        # -- Inner rect (assumed already computed; replicate minAreaRect on inner contours) --
        # Find inner contours (RETR_LIST gives all; pick second-largest inside outer box)
        inner_rect_data = None
        inner_box_pts   = None
        irw = irh = 0.0
        icx = cx
        icy = cy

        # Search for inner contour: find all contours inside the outer box
        all_contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        best_inner_area = 0
        best_inner_cnt  = None
        outer_box_f32 = cv2.boxPoints(rect).astype(np.float32).reshape(-1, 1, 2)

        for ic in all_contours:
            ia = cv2.contourArea(ic)
            if ia < 100:
                continue
            # Must be smaller than outer rect and mostly inside it
            im = cv2.moments(ic)
            if im["m00"] == 0:
                continue
            icx_t = im["m10"] / im["m00"]
            icy_t = im["m01"] / im["m00"]
            if cv2.pointPolygonTest(outer_box_f32, (icx_t, icy_t), False) < 0:
                continue
            if ia >= area:
                continue
            if ia > best_inner_area:
                best_inner_area = ia
                best_inner_cnt  = ic

        if best_inner_cnt is not None:
            inner_rect = cv2.minAreaRect(best_inner_cnt)
            inner_center_f, (irw_r, irh_r), iangle = inner_rect
            if irw_r < irh_r:
                irw, irh = irh_r, irw_r
            else:
                irw, irh = irw_r, irh_r
            icx, icy = float(inner_center_f[0]), float(inner_center_f[1])
            # float32 array for pointPolygonTest
            inner_box_pts = cv2.boxPoints(inner_rect).astype(np.float32).reshape(-1, 1, 2)
            inner_rect_data = {
                "center": (round(icx, 1), round(icy, 1)),
                "wh":     (round(irw, 1), round(irh, 1)),
                "angle":  round(iangle % 180, 2),
            }
            # Draw inner rect
            inner_box_draw = cv2.boxPoints(inner_rect).astype(int)
            cv2.drawContours(out, [inner_box_draw], 0, (180, 255, 100), 2)

        # ═══ ส่วนที่ 2: แทนที่ STEP A ทั้งหมด ═══
        # ══════════════════════════════════════════════
        #  STEP A — Collect candidate circle points
        #  (with gap-side detection for 1-gap case)
        # ══════════════════════════════════════════════
        circle_pts = []

        # ── H-scan ──
        if len(h_trans) >= 5:
            # 2 gaps → ใช้ index [2] และ [3] แบบเดิม
            circle_pts.append(h_trans[2])
            circle_pts.append(h_trans[3])
        elif len(h_trans) >= 3 and h_gaps:
            # 1 gap → เช็คว่า gap midpoint ใกล้ขอบซ้ายหรือขวาของ rect
            g = h_gaps[0]
            mid_x = (g["x1"] + g["x2"]) / 2
            dist_left  = mid_x - (cx - hl)   # ระยะจาก midpoint ถึงขอบซ้าย
            dist_right = (cx + hl) - mid_x   # ระยะจาก midpoint ถึงขอบขวา
            if dist_left <= dist_right:       # gap ใกล้ซ้าย
                circle_pts.append(h_trans[2])
            else:                             # gap ใกล้ขวา
                circle_pts.append(h_trans[1])

        # ── V-scan ──
        if len(v_trans) >= 5:
            # 2 gaps → ใช้ index [2] และ [3] แบบเดิม
            circle_pts.append(v_trans[2])
            circle_pts.append(v_trans[3])
        elif len(v_trans) >= 3 and v_gaps:
            # 1 gap → เช็คว่า gap midpoint ใกล้ขอบบนหรือล่างของ rect
            g = v_gaps[0]
            mid_y = (g["y1"] + g["y2"]) / 2
            dist_top    = mid_y - (cy - hs)  # ระยะจาก midpoint ถึงขอบบน
            dist_bottom = (cy + hs) - mid_y  # ระยะจาก midpoint ถึงขอบล่าง
            if dist_top <= dist_bottom:       # gap ใกล้บน
                circle_pts.append(v_trans[2])
            else:                             # gap ใกล้ล่าง
                circle_pts.append(v_trans[1])
        circle_data = None

        # if len(circle_pts) == 2:
        # ── Diagonal scans (D1, D2) — เพิ่มจุดเสมอ ──
        d1_trans, d2_trans = scan_diagonal(
            gray, cx, cy, hl, hs, lx, ly, sx, sy, gap
        )

        # D1: fixed index เหมือน H/V
        if len(d1_trans) >= 5:
            circle_pts.append(d1_trans[2])
            circle_pts.append(d1_trans[3])
        elif len(d1_trans) >= 3:
            circle_pts.append(d1_trans[2])

        # D2: fixed index เหมือน H/V
        if len(d2_trans) >= 5:
            circle_pts.append(d2_trans[2])
            circle_pts.append(d2_trans[3])
        elif len(d2_trans) >= 3:
            circle_pts.append(d2_trans[2])

        circle_data = None  # ← บรรทัดนี้ย้ายมาอยู่หลัง diagonal แล้ว


        if len(circle_pts) >= 2:
            # ------------------------------------------
            #  STEP B — Fit one circle through all points
            # ------------------------------------------
            fit = fit_circle(circle_pts)

            if fit is not None:
                ccx, ccy, R = fit

                # Validity check: reject degenerate radius
                max_dim = max(rw, rh)
                if R <= 0 or R > 2 * max_dim:
                    fit = None   # invalid — skip drawing

            if fit is not None:
                ccx, ccy, R = fit

                # --------------------------------------
                #  STEP C — Draw circle and keypoints
                # --------------------------------------
                cv2.circle(out, (int(round(ccx)), int(round(ccy))),
                           int(round(R)), C["circle"], 2)

                # Mark each circle point with diamond (thickness must be >= 1 for drawMarker)
                for pt in circle_pts:
                    cv2.drawMarker(out,
                                   (int(round(pt["x"])), int(round(pt["y"]))),
                                   C["circ_pt"],
                                   cv2.MARKER_DIAMOND, 14, 3)

                # Draw circle center cross (reuse C["ctr"] colour)
                # cv2.drawMarker(out,
                #                (int(round(ccx)), int(round(ccy))),
                #                C["ctr"],
                #                cv2.MARKER_TILTED_CROSS, 16, 2)

                # --------------------------------------
                #  STEP D — Measure gap: circle edge ? inner rect
                # --------------------------------------
                direction_info = [
                    ( lx,  ly, "H+"),
                    (-lx, -ly, "H-"),
                    ( sx,  sy, "V+"),
                    (-sx, -sy, "V-"),
                ]

                circle_gaps = {}

                if inner_box_pts is not None:
                    max_dist = max(irw, irh) if (irw > 0 or irh > 0) else max(rw, rh)

                    for dx, dy, label in direction_info:
                        gap_result = measure_gap_to_inner(
                            (ccx, ccy), R, dx, dy,
                            inner_box_pts, max_dist, ppu
                        )
                        if gap_result is None:
                            circle_gaps[label] = {"px": None, "um": None}
                            continue

                        gap_px, edge_x, edge_y, cex, cey = gap_result
                        gap_um = px_to_um(abs(gap_px), ppu)
                        if gap_px < 0 and gap_um is not None:
                            gap_um = -gap_um

                        circle_gaps[label] = {
                            "px": round(gap_px, 2),
                            "um": round(gap_um, 4) if gap_um is not None else None,
                        }

                        # Draw gap line
                        cv2.line(out,
                                 (int(round(cex)), int(round(cey))),
                                 (int(round(edge_x)), int(round(edge_y))),
                                 C["gap"], 2)

                        # Text label near midpoint
                        mx = int((cex + edge_x) / 2)
                        my = int((cey + edge_y) / 2)
                        um_str = f" ({gap_um:.2f}um)" if gap_um is not None else ""
                        sign_str = "" if gap_px >= 0 else "-"
                        cv2.putText(out,
                                    f"{label}: {sign_str}{abs(gap_px):.1f}px {um_str}",
                                    (mx + 4, my - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                                    C["gap"], 1, cv2.LINE_AA)
                else:
                    # No inner rect found; gaps cannot be measured
                    for _, _, label in direction_info:
                        circle_gaps[label] = {"px": None, "um": None}

                # --------------------------------------
                #  STEP E — Store circle data
                # --------------------------------------
                circle_data = {
                    "center":       (round(ccx, 1), round(ccy, 1)),
                    "radius":       round(R, 1),
                    "points_used":  len(circle_pts),
                    "gaps":         circle_gaps,
                }

        results.append({
            "center":     (int(cx), int(cy)),
            "rect_wh":    (round(rw, 1), round(rh, 1)),
            "angle":       round(angle % 180, 2),
            "h_line":     {"start": (round(hx1,1), round(hy1,1)),
                           "end":   (round(hx2,1), round(hy2,1)),
                           "transitions": h_trans, "gaps": h_gaps},
            "v_line":     {"start": (round(vx1,1), round(vy1,1)),
                           "end":   (round(vx2,1), round(vy2,1)),
                           "transitions": v_trans, "gaps": v_gaps},
            "ppu":         ppu,
            "inner_rect":  inner_rect_data,
            "circle":      circle_data,
        })

    cv2.putText(out, "? Start(grn) ? End(blu)  +W?B(red) +B?W(org)  --Gap(cyan)",
                (8, h_img-8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160,160,160),1,cv2.LINE_AA)
    return out, results


# -------------------------------------------------------
#  ZOOM CANVAS  (unchanged)
# -------------------------------------------------------
class ZoomCanvas(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, bg="#0d0d0d", **kw)
        self.canvas = tk.Canvas(self, bg="#0d0d0d", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._pil_img = None
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0
        self._drag  = None

        self.canvas.bind("<ButtonPress-1>",   lambda e: setattr(self, "_drag", (e.x,e.y)))
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<MouseWheel>",      self._on_wheel)
        self.canvas.bind("<Button-4>",        self._on_wheel)
        self.canvas.bind("<Button-5>",        self._on_wheel)
        self.canvas.bind("<Configure>",       lambda e: self._redraw())

        self._zoom_lbl = tk.Label(self, bg="#16213e", fg="#7eb8f7",
                                   font=("Consolas",9,"bold"), text="100%")
        self._zoom_lbl.place(relx=1.0, rely=0.0, anchor="ne", x=-6, y=6)
        tk.Button(self, text="? Reset View", command=self.reset_view,
                  bg="#16213e", fg="#a0c4ff", font=("Consolas",8),
                  relief="flat", cursor="hand2", pady=2
                  ).place(relx=0.0, rely=0.0, anchor="nw", x=6, y=6)

    def set_image(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._pil_img = Image.fromarray(rgb)
        self.reset_view()

    def update_live(self, bgr):
        """Update display without resetting zoom/pan (for live stream)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._pil_img = Image.fromarray(rgb)
        self._redraw()

    def reset_view(self):
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0
        self._redraw()

    def _redraw(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()  or 640
        ch = self.canvas.winfo_height() or 480
        if self._pil_img is None:
            self.canvas.create_text(cw//2, ch//2, text="No signal",
                                    fill="#333", font=("Consolas",13))
            return
        base  = min(cw/self._pil_img.width, ch/self._pil_img.height, 1.0)
        scale = base * self._zoom
        nw    = max(1, int(self._pil_img.width  * scale))
        nh    = max(1, int(self._pil_img.height * scale))
        resized = self._pil_img.resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(resized)
        ox = (cw-nw)//2 + self._pan_x
        oy = (ch-nh)//2 + self._pan_y
        self.canvas.create_image(ox, oy, anchor="nw", image=self._tk_img)
        self._zoom_lbl.config(text=f"{int(scale*100)}%")

    def _on_drag(self, e):
        if self._drag:
            self._pan_x += e.x - self._drag[0]
            self._pan_y += e.y - self._drag[1]
            self._drag = (e.x, e.y)
            self._redraw()

    def _on_wheel(self, e):
        f = 1.15 if (e.num == 4 or e.delta > 0) else 1/1.15
        self._zoom = max(0.05, min(self._zoom * f, 20.0))
        self._redraw()


# -------------------------------------------------------
#  CAMERA THREAD  (unchanged)
# -------------------------------------------------------
class CameraThread(threading.Thread):
    """Background thread that continuously reads frames."""
    def __init__(self, camera_index=1, use_getframe=True):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.use_getframe = use_getframe
        self.frame  = None
        self.running = False
        self._lock  = threading.Lock()
        self._cap   = None
        self._cam   = None
        self.error  = None

    def run(self):
        self.running = True
        try:
            if self.use_getframe and HAS_GETFRAME:
                self._cam = Camera()
                self._cam.open(self.camera_index)
            else:
                self._cap = cv2.VideoCapture(self.camera_index)
                if not self._cap.isOpened():
                    raise RuntimeError(f"Cannot open camera index {self.camera_index}")

            while self.running:
                if self._cam is not None:
                    f = self._cam.get_frame()
                else:
                    ret, f = self._cap.read()
                    if not ret:
                        f = None
                if f is not None:
                    with self._lock:
                        self.frame = f.copy()
                time.sleep(0.01)
        except Exception as ex:
            self.error = str(ex)
        finally:
            self._cleanup()

    def get_frame(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False

    def _cleanup(self):
        try:
            if self._cap is not None:
                self._cap.release()
            if self._cam is not None and hasattr(self._cam, "close"):
                self._cam.close()
        except Exception:
            pass


# -------------------------------------------------------
#  COLOUR PALETTE
# -------------------------------------------------------
BG_DARK   = "#0f1117"
BG_PANEL  = "#1a1d27"
BG_CARD   = "#22263a"
BG_INPUT  = "#0d1020"
ACCENT    = "#00c8ff"
ACCENT2   = "#7b61ff"
GREEN     = "#00e676"
ORANGE    = "#ff9100"
RED       = "#ff3d3d"
FG_MAIN   = "#e8eaf6"
FG_DIM    = "#7986a3"
FG_LABEL  = "#90caf9"

CORNER_COLORS = {
    "LEFT TOP":  "#00c8ff",
    "RIGHT TOP": "#00e676",
    "RIGHT BOT": "#ff9100",
    "LEFT BOT":  "#c77dff",
}

GAP_LABELS = ["RIGHT (H+)", "LEFT (H-)", "TOP (V+)", "BOT (V-)"]
GAP_KEYS   = ["H+", "H-", "V+", "V-"]

STATE_IDLE    = "IDLE"
STATE_LIVE    = "LIVE"
STATE_FROZEN  = "FROZEN"
STATE_STATIC  = "STATIC"


# -------------------------------------------------------
#  MAIN APPLICATION
# -------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PT MeasStation  v2.0")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.geometry("1440x860")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.cfg = dict(DEFAULT_CONFIG)

        # Job state
        self._lot    = ""
        self._opt_id = ""
        self._step   = 0          # 0 = idle, 1-4 = measuring corner index
        self._job_started = False

        # Collected data per corner
        self._corners_gaps   = {}   # {corner_name: {H+,H-,V+,V-}}
        self._corners_frames = {}   # {corner_name: cv2_frame}

        # Camera
        self.state = STATE_IDLE
        self.raw_frame = None
        self._cam_thread: CameraThread | None = None
        self._live_after = None

        self._build_ui()
        self._refresh_ui()

    # ====================================================
    #  UI BUILD
    # ====================================================
    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────
        topbar = tk.Frame(self, bg="#111827", pady=0)
        topbar.pack(fill="x")

        # Title
        tk.Label(topbar, text="PT  MEAS STATION",
                 bg="#111827", fg=ACCENT,
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=18, pady=8)

        # Camera button (right side of topbar)
        self._btn_cam = tk.Button(
            topbar, text="📷  Open Camera",
            command=self._open_camera,
            bg="#1e293b", fg=FG_DIM,
            activebackground="#2a3f5f", activeforeground=ACCENT,
            font=("Segoe UI", 9), relief="flat", padx=10, pady=4, cursor="hand2")
        self._btn_cam.pack(side="right", padx=12, pady=6)

        self._cam_status_var = tk.StringVar(value="⬤  No Camera")
        self._cam_status_lbl = tk.Label(topbar, textvariable=self._cam_status_var,
                 bg="#111827", fg=RED,
                 font=("Segoe UI", 9))
        self._cam_status_lbl.pack(side="right", padx=4)

        # ── Main layout ──────────────────────────────────
        main = tk.Frame(self, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=10, pady=(4, 8))

        # Left: camera + input + process button
        left = tk.Frame(main, bg=BG_DARK)
        left.pack(side="left", fill="both", expand=True)

        self._build_camera_area(left)
        self._build_input_panel(left)
        self._build_process_button(left)

        # Right: results 2×2 grid + analysis panel
        right = tk.Frame(main, bg=BG_DARK, width=520)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)
        self._build_results_panel(right)
        self._build_analysis_panel(right)

    # ── Camera area ──────────────────────────────────────
    def _build_camera_area(self, parent):
        cam_frame = tk.Frame(parent, bg=BG_PANEL, bd=0)
        cam_frame.pack(fill="both", expand=True, pady=(0, 6))

        self.zoom_canvas = ZoomCanvas(cam_frame, bd=0, relief="flat")
        self.zoom_canvas.pack(fill="both", expand=True)

    # ── Input panel ──────────────────────────────────────
    def _build_input_panel(self, parent):
        panel = tk.Frame(parent, bg=BG_PANEL, pady=10, padx=14)
        panel.pack(fill="x", pady=(0, 6))

        tk.Label(panel, text="JOB INFORMATION",
                 bg=BG_PANEL, fg=FG_DIM,
                 font=("Segoe UI", 8, "bold")).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0,6))

        # LOT
        tk.Label(panel, text="LOT", bg=BG_PANEL, fg=FG_LABEL,
                 font=("Segoe UI", 10, "bold")).grid(row=1, column=0, sticky="w", padx=(0,8))
        self._lot_var = tk.StringVar()
        lot_entry = tk.Entry(panel, textvariable=self._lot_var,
                             bg=BG_INPUT, fg=FG_MAIN, insertbackground=ACCENT,
                             font=("Consolas", 11), relief="flat", width=20,
                             highlightthickness=1, highlightcolor=ACCENT,
                             highlightbackground="#2a3f5f")
        lot_entry.grid(row=1, column=1, sticky="ew", padx=(0, 20), ipady=5)

        # OPT ID
        tk.Label(panel, text="OPT ID", bg=BG_PANEL, fg=FG_LABEL,
                 font=("Segoe UI", 10, "bold")).grid(row=1, column=2, sticky="w", padx=(0,8))
        self._opt_var = tk.StringVar()
        opt_entry = tk.Entry(panel, textvariable=self._opt_var,
                             bg=BG_INPUT, fg=FG_MAIN, insertbackground=ACCENT,
                             font=("Consolas", 11), relief="flat", width=20,
                             highlightthickness=1, highlightcolor=ACCENT,
                             highlightbackground="#2a3f5f")
        opt_entry.grid(row=1, column=3, sticky="ew", ipady=5)

        panel.columnconfigure(1, weight=1)
        panel.columnconfigure(3, weight=1)

    # ── Process button + step indicator ──────────────────
    def _build_process_button(self, parent):
        frame = tk.Frame(parent, bg=BG_PANEL, pady=10, padx=14)
        frame.pack(fill="x")

        # Step indicator row
        ind_row = tk.Frame(frame, bg=BG_PANEL)
        ind_row.pack(fill="x", pady=(0, 8))

        self._step_dots = []
        for i, corner in enumerate(CORNER_ORDER):
            col = CORNER_COLORS[corner]
            dot_frame = tk.Frame(ind_row, bg=BG_PANEL)
            dot_frame.pack(side="left", padx=12)

            dot = tk.Label(dot_frame, text="◉", bg=BG_PANEL, fg="#2a3f5f",
                           font=("Segoe UI", 16))
            dot.pack()
            lbl = tk.Label(dot_frame, text=corner, bg=BG_PANEL, fg=FG_DIM,
                           font=("Segoe UI", 7, "bold"))
            lbl.pack()
            self._step_dots.append((dot, lbl, col))

        # Buttons row
        btn_row = tk.Frame(frame, bg=BG_PANEL)
        btn_row.pack(fill="x")

        self._process_label_var = tk.StringVar(value="PROCESSING")
        self._btn_process = tk.Button(
            btn_row, textvariable=self._process_label_var,
            command=self._on_process,
            bg="#1a3a5c", fg=ACCENT,
            activebackground="#0f2a4a", activeforeground=ACCENT,
            font=("Segoe UI", 13, "bold"),
            relief="flat", pady=12, cursor="hand2",
            disabledforeground="#2a3f5f")
        self._btn_process.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self._btn_newjob = tk.Button(
            btn_row, text="🔄  NEW JOB",
            command=self._new_job,
            bg="#1e2a1e", fg=GREEN,
            activebackground="#0f1f0f", activeforeground=GREEN,
            font=("Segoe UI", 11, "bold"),
            relief="flat", pady=12, cursor="hand2")
        self._btn_newjob.pack(side="right", padx=(0,0))

        # Status line
        self._info_var = tk.StringVar(value="Enter LOT and OPT ID, then press PROCESSING")
        tk.Label(frame, textvariable=self._info_var,
                 bg=BG_PANEL, fg=FG_DIM,
                 font=("Segoe UI", 9)).pack(pady=(8, 0))

    # ── Results 2×2 grid ─────────────────────────────────
    def _build_results_panel(self, parent):
        tk.Label(parent, text="MEASUREMENT RESULTS",
                 bg=BG_DARK, fg=FG_DIM,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(2, 2))

        grid = tk.Frame(parent, bg=BG_DARK)
        grid.pack(fill="x")

        # Fixed column/row sizes — ไม่ขยับตาม content
        CARD_W = 240
        CARD_H = 110
        grid.columnconfigure(0, minsize=CARD_W, weight=1)
        grid.columnconfigure(1, minsize=CARD_W, weight=1)
        grid.rowconfigure(0, minsize=CARD_H)
        grid.rowconfigure(1, minsize=CARD_H)

        positions = {
            "LEFT TOP":  (0, 0),
            "RIGHT TOP": (0, 1),
            "LEFT BOT":  (1, 0),
            "RIGHT BOT": (1, 1),
        }

        self._corner_widgets = {}
        for corner, (row, col) in positions.items():
            card = self._build_corner_card(grid, corner)
            card.grid(row=row, column=col, sticky="nsew", padx=3, pady=3)

    def _build_corner_card(self, parent, corner_name):
        color = CORNER_COLORS[corner_name]
        card = tk.Frame(parent, bg=BG_CARD, bd=0,
                        highlightthickness=2, highlightbackground="#2a3f5f")
        card.pack_propagate(False)   # ← fixed size ไม่ขยับ

        # Header — แสดง corner name + hint คลิก re-measure
        hdr = tk.Frame(card, bg=color, pady=2)
        hdr.pack(fill="x")
        tk.Label(hdr, text=corner_name, bg=color, fg="#0d1020",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=6)
        hint_lbl = tk.Label(hdr, text="", bg=color, fg="#0d1020",
                            font=("Segoe UI", 6, "italic"))
        hint_lbl.pack(side="right", padx=4)

        # Gap rows — µm only
        gap_vars = {}
        for gap_label, gap_key in zip(GAP_LABELS, GAP_KEYS):
            row = tk.Frame(card, bg=BG_CARD, pady=1)
            row.pack(fill="x", padx=6)

            tk.Label(row, text=gap_label, bg=BG_CARD, fg=FG_DIM,
                     font=("Segoe UI", 7), width=11, anchor="w").pack(side="left")

            var = tk.StringVar(value="—")
            val_lbl = tk.Label(row, textvariable=var,
                               bg=BG_CARD, fg=FG_MAIN,
                               font=("Consolas", 8, "bold"), anchor="e", width=14)
            val_lbl.pack(side="right")
            gap_vars[gap_key] = (var, val_lbl)

        # Status
        status_var = tk.StringVar(value="Waiting...")
        status_lbl = tk.Label(card, textvariable=status_var,
                              bg=BG_CARD, fg=FG_DIM,
                              font=("Segoe UI", 6, "italic"), pady=2)
        status_lbl.pack()

        self._corner_widgets[corner_name] = {
            "card":       card,
            "gap_vars":   gap_vars,
            "status_var": status_var,
            "status_lbl": status_lbl,
            "hdr_color":  color,
            "hint_lbl":   hint_lbl,
        }

        # Click-to-remeasure binding (ทุก widget ใน card)
        def _on_click(e, cn=corner_name):
            self._remeasure_corner(cn)

        for w in [card, hdr, hint_lbl, status_lbl]:
            w.bind("<Button-1>", _on_click)

        return card

    # ── Analysis Panel ───────────────────────────────────
    def _build_analysis_panel(self, parent):
        outer = tk.Frame(parent, bg=BG_DARK)
        outer.pack(fill="both", expand=True, pady=(6, 0))

        tk.Label(outer, text="ALIGNMENT ANALYSIS",
                 bg=BG_DARK, fg=FG_DIM,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 2))

        # ── Diagram + Offsets side by side ────────────────
        row_top = tk.Frame(outer, bg=BG_DARK)
        row_top.pack(fill="x")

        # Diagram canvas — smaller
        diag_size = 120
        self._diag_canvas = tk.Canvas(
            row_top, width=diag_size, height=diag_size,
            bg=BG_CARD, highlightthickness=1,
            highlightbackground="#2a3f5f")
        self._diag_canvas.pack(side="left", padx=(0, 8))
        self._draw_diagram_idle()

        # Offset info cards — compact single-line layout
        info_col = tk.Frame(row_top, bg=BG_DARK)
        info_col.pack(side="left", fill="both", expand=True)

        self._offset_vars = {}
        offset_defs = [
            ("X OFFSET", "x_offset", "← →", ACCENT),
            ("Y OFFSET", "y_offset", "↑ ↓", "#c77dff"),
            ("ROTATION", "rotation", "↻ ↺", ORANGE),
        ]
        for label, key, symbol, color in offset_defs:
            card = tk.Frame(info_col, bg=BG_CARD, pady=3, padx=6)
            card.pack(fill="x", pady=2)
            top = tk.Frame(card, bg=BG_CARD)
            top.pack(fill="x")
            tk.Label(top, text=f"{symbol}  {label}", bg=BG_CARD, fg=FG_DIM,
                     font=("Segoe UI", 7, "bold")).pack(side="left")
            val_var = tk.StringVar(value="—")
            tk.Label(card, textvariable=val_var,
                     bg=BG_CARD, fg=color,
                     font=("Consolas", 10, "bold")).pack(anchor="w")
            self._offset_vars[key] = val_var

        # ── Adjuster Recommendations ──────────────────────
        adj_frame = tk.Frame(outer, bg=BG_CARD, pady=6, padx=10)
        adj_frame.pack(fill="both", expand=True, pady=(6, 0))

        tk.Label(adj_frame, text="🔧  ADJUSTER RECOMMENDATION",
                 bg=BG_CARD, fg=FG_DIM,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 4))

        self._adj_vars = {}
        adj_defs = [
            ("RIGHT",     "adj_right", ACCENT,   "X axis — Horizontal shift"),
            ("BOT-LEFT",  "adj_bl",    "#c77dff", "Y shift + Rotation"),
            ("BOT-RIGHT", "adj_br",    "#c77dff", "Y shift − Rotation"),
        ]
        for adj_name, key, color, subtitle in adj_defs:
            row = tk.Frame(adj_frame, bg="#1a1d2e", pady=4, padx=8)
            row.pack(fill="x", pady=2)

            # Left: name + subtitle
            left = tk.Frame(row, bg="#1a1d2e")
            left.pack(side="left", fill="y")
            tk.Label(left, text=adj_name, bg="#1a1d2e", fg=color,
                     font=("Segoe UI", 9, "bold"), anchor="w").pack(anchor="w")
            tk.Label(left, text=subtitle, bg="#1a1d2e", fg=FG_DIM,
                     font=("Segoe UI", 6), anchor="w").pack(anchor="w")

            # Right: direction + value stacked
            right = tk.Frame(row, bg="#1a1d2e")
            right.pack(side="right")
            dir_var = tk.StringVar(value="")
            tk.Label(right, textvariable=dir_var,
                     bg="#1a1d2e", fg=color,
                     font=("Segoe UI", 8, "bold"), anchor="e", width=10).pack(anchor="e")
            val_var = tk.StringVar(value="—")
            tk.Label(right, textvariable=val_var,
                     bg="#1a1d2e", fg=FG_MAIN,
                     font=("Consolas", 12, "bold"), anchor="e").pack(anchor="e")

            self._adj_vars[key] = (val_var, dir_var)

    def _draw_diagram_idle(self):
        c = self._diag_canvas
        c.delete("all")
        s = int(c["width"])
        cx, cy = s // 2, s // 2
        pad = 14
        c.create_rectangle(pad, pad, s-pad, s-pad,
                           outline="#2a3f5f", width=2)
        c.create_line(cx-8, cy, cx+8, cy, fill="#2a3f5f", width=1)
        c.create_line(cx, cy-8, cx, cy+8, fill="#2a3f5f", width=1)
        r = 20
        c.create_oval(cx-r, cy-r, cx+r, cy+r,
                      outline="#2a3f5f", width=2)
        c.create_text(cx, cy, text="?", fill="#2a3f5f",
                     font=("Segoe UI", 7))

    def _update_analysis(self, analysis):
        """Refresh analysis panel with computed alignment data."""
        if not analysis.get("valid"):
            for var in self._offset_vars.values():
                var.set("—")
            for val_var, dir_var in self._adj_vars.values():
                val_var.set("—")
                dir_var.set("")
            self._draw_diagram_idle()
            return

        x_off = analysis["x_offset"]
        y_off = analysis["y_offset"]
        rot   = analysis["rotation"]

        # Offset labels
        def fmt_offset(val, pos_dir, neg_dir):
            if abs(val) < 0.5:
                return "≈ 0  (centered)"
            direction = pos_dir if val > 0 else neg_dir
            return f"{abs(val):.1f} µm  {direction}"

        self._offset_vars["x_offset"].set(fmt_offset(x_off, "→ RIGHT", "← LEFT"))
        self._offset_vars["y_offset"].set(fmt_offset(y_off, "↑ UP",    "↓ DOWN"))
        self._offset_vars["rotation"].set(fmt_offset(rot,   "↻ CW",    "↺ CCW"))

        # Adjuster recommendations
        adj_defs = [
            ("adj_right", analysis["adj_right"], "IN",  "OUT"),
            ("adj_bl",    analysis["adj_bl"],    "UP",  "DOWN"),
            ("adj_br",    analysis["adj_br"],    "UP",  "DOWN"),
        ]
        for key, val, pos_lbl, neg_lbl in adj_defs:
            val_var, dir_var = self._adj_vars[key]
            if abs(val) < 0.5:
                val_var.set("No change")
                dir_var.set("")
            else:
                direction = pos_lbl if val > 0 else neg_lbl
                val_var.set(f"{abs(val):.1f} µm")
                dir_var.set(f"▲ {direction}" if val > 0 else f"▼ {direction}")

        # Diagram
        self._draw_diagram_arrow(x_off, y_off, rot)

    def _draw_diagram_arrow(self, x_off, y_off, rotation):
        """Draw 2D diagram showing circle offset direction."""
        c = self._diag_canvas
        c.delete("all")
        s    = int(c["width"])
        cx   = s // 2
        cy   = s // 2
        pad  = 14
        r    = 20
        scale = 1.0

        # Glass outline
        c.create_rectangle(pad, pad, s-pad, s-pad,
                           outline="#3a4f6f", width=2)
        # Corner labels
        for tx, ty, txt, anch in [(pad+2, pad+2, "LT", "nw"),
                                   (s-pad-2, pad+2, "RT", "ne"),
                                   (pad+2, s-pad-2, "LB", "sw"),
                                   (s-pad-2, s-pad-2, "RB", "se")]:
            c.create_text(tx, ty, text=txt, fill="#3a4f6f",
                         font=("Segoe UI", 5), anchor=anch)

        # Center crosshair
        c.create_line(cx-6, cy, cx+6, cy, fill="#3a4f6f", width=1, dash=(2,2))
        c.create_line(cx, cy-6, cx, cy+6, fill="#3a4f6f", width=1, dash=(2,2))

        # Current circle position
        max_shift = s // 2 - pad - r - 2
        circ_x = cx + min(max(x_off * scale, -max_shift), max_shift)
        circ_y = cy - min(max(y_off * scale, -max_shift), max_shift)

        # Draw circle
        c.create_oval(circ_x-r, circ_y-r, circ_x+r, circ_y+r,
                      outline=ACCENT, width=2)

        # Arrow to ideal center
        if abs(x_off) > 0.5 or abs(y_off) > 0.5:
            c.create_line(circ_x, circ_y, cx, cy,
                         fill=GREEN, width=2,
                         arrow=tk.LAST, arrowshape=(7, 8, 3))

        # Rotation arc
        if abs(rotation) > 0.5:
            arc_r = r + 6
            extent = min(max(rotation * 3, -50), 50)
            c.create_arc(circ_x-arc_r, circ_y-arc_r,
                        circ_x+arc_r, circ_y+arc_r,
                        start=80, extent=extent,
                        style="arc", outline=ORANGE, width=2)
            rot_lbl = "↻" if rotation > 0 else "↺"
            c.create_text(circ_x, circ_y - arc_r - 4,
                         text=rot_lbl, fill=ORANGE,
                         font=("Segoe UI", 8, "bold"))

    # ====================================================
    #  UI STATE REFRESH
    # ====================================================
    def _refresh_ui(self):
        """Update buttons, step dots, and info label based on current state."""
        step    = self._step
        started = self._job_started
        has_cam = self._cam_thread is not None and self._cam_thread.running

        # Step dots
        for i, (dot, lbl, col) in enumerate(self._step_dots):
            if i < step:
                dot.config(fg=col)           # done
                lbl.config(fg=col)
            elif i == step and started:
                dot.config(fg=ORANGE)        # current (measuring)
                lbl.config(fg=ORANGE)
            else:
                dot.config(fg="#2a3f5f")     # pending
                lbl.config(fg=FG_DIM)

        # Process button
        if not started:
            self._btn_process.config(
                text="▶  START JOB", state="normal",
                bg="#1a3a5c", fg=ACCENT)
            self._process_label_var.set("▶  START JOB")
        elif step < 4:
            corner = CORNER_ORDER[step]
            self._process_label_var.set(f"⬤  PROCESSING  —  {corner}")
            self._btn_process.config(
                state="normal" if has_cam else "disabled",
                bg="#0f2a4a", fg=ORANGE)
        else:
            self._process_label_var.set("✓  COMPLETE")
            self._btn_process.config(state="disabled", bg="#1a2a1a", fg=GREEN)

        # Camera status
        if has_cam:
            self._cam_status_var.set("⬤  Camera ON")
            self._cam_status_lbl.config(fg=GREEN)
        else:
            self._cam_status_var.set("⬤  No Camera")
            self._cam_status_lbl.config(fg=RED)

    # ====================================================
    #  CAMERA CONTROL
    # ====================================================
    def _open_camera(self):
        idx = simpledialog.askinteger(
            "Camera", "Camera index:", initialvalue=1,
            minvalue=0, maxvalue=10, parent=self)
        if idx is None:
            return

        self._stop_camera()
        use_gf = HAS_GETFRAME
        self._cam_thread = CameraThread(camera_index=idx, use_getframe=use_gf)
        self._cam_thread.start()
        time.sleep(0.4)
        if self._cam_thread.error:
            messagebox.showerror("Camera Error", self._cam_thread.error)
            self._cam_thread = None
            return

        self._cam_status_lbl.config(fg=GREEN)
        self._cam_status_var.set("⬤  Camera ON")
        self.state = STATE_LIVE
        self._live_loop()
        self._refresh_ui()

    def _stop_camera(self):
        if self._live_after is not None:
            self.after_cancel(self._live_after)
            self._live_after = None
        if self._cam_thread is not None:
            self._cam_thread.stop()
            self._cam_thread.join(timeout=1.5)
            self._cam_thread = None

    def _live_loop(self):
        if self.state != STATE_LIVE:
            return
        if self._cam_thread is None or not self._cam_thread.running:
            self.state = STATE_IDLE
            return
        f = self._cam_thread.get_frame()
        if f is not None:
            self.zoom_canvas.update_live(f)
        self._live_after = self.after(33, self._live_loop)

    # ====================================================
    #  JOB FLOW
    # ====================================================
    def _on_process(self):
        if not self._job_started:
            self._start_job()
        else:
            self._do_measure()

    def _start_job(self):
        lot    = self._lot_var.get().strip()
        opt_id = self._opt_var.get().strip()
        if not lot:
            messagebox.showwarning("Missing Input", "Please enter LOT number.")
            return
        if not opt_id:
            messagebox.showwarning("Missing Input", "Please enter OPT ID.")
            return
        if self._cam_thread is None or not self._cam_thread.running:
            messagebox.showwarning("No Camera", "Please open a camera first.")
            return

        self._lot    = lot
        self._opt_id = opt_id
        self._step   = 0
        self._job_started = True
        self._corners_gaps   = {}
        self._corners_frames = {}

        # Reset corner cards
        for corner, w in self._corner_widgets.items():
            for key, (var, lbl) in w["gap_vars"].items():
                var.set("—")
                lbl.config(fg=FG_MAIN)
            w["status_var"].set("Waiting...")
            w["status_lbl"].config(fg=FG_DIM)
            w["card"].config(highlightbackground="#2a3f5f")

        self._info_var.set(f"LOT: {lot}  |  OPT ID: {opt_id}  →  Press PROCESSING to measure corner 1/4")
        self._lock_inputs(True)
        self._update_analysis({"valid": False})
        self._refresh_ui()

    def _do_measure(self, corner_name=None):
        """Measure a corner. If corner_name is None, use current step."""
        if corner_name is None:
            if self._step >= 4:
                return
            corner_name = CORNER_ORDER[self._step]

        if self._cam_thread is None or not self._cam_thread.running:
            messagebox.showwarning("No Camera", "Camera not running.")
            return

        w = self._corner_widgets[corner_name]
        w["status_var"].set("Measuring...")
        w["status_lbl"].config(fg=ORANGE)
        w["card"].config(highlightbackground=ORANGE)
        self.update_idletasks()

        # Capture + detect
        frame = self._cam_thread.get_frame()
        if frame is None:
            self._beep(fail=True)
            w["status_var"].set("✗ No frame — click to retry")
            w["status_lbl"].config(fg=RED)
            w["card"].config(highlightbackground=RED)
            return

        out_frame, results = detect_on_frame(frame, self.cfg)
        gaps = _get_circle_gaps(results)
        has_data = any(v.get("um") is not None for v in gaps.values())

        # ── Detection failed → block, do NOT advance step ──
        if not has_data:
            self._beep(fail=True)
            for key, (var, lbl) in w["gap_vars"].items():
                var.set("N/A")
                lbl.config(fg=RED)
            w["status_var"].set("✗ Not detected — click to retry")
            w["status_lbl"].config(fg=RED)
            w["card"].config(highlightbackground=RED)
            return

        # ── Detection OK ─────────────────────────────────
        self._corners_gaps[corner_name]   = gaps
        self._corners_frames[corner_name] = out_frame

        for key, (var, lbl) in w["gap_vars"].items():
            entry = gaps.get(key, {})
            um = entry.get("um")
            if um is None:
                var.set("N/A")
                lbl.config(fg=RED)
            else:
                var.set(f"{um:+.2f} µm")
                lbl.config(fg=GREEN if um >= 0 else ORANGE)

        w["status_var"].set("✓ Done  (click to re-measure)")
        w["status_lbl"].config(fg=GREEN)
        w["card"].config(highlightbackground=CORNER_COLORS[corner_name])
        w["hint_lbl"].config(text="🔁 re-measure")

        self._beep(fail=False)

        # Advance step only if this was the current sequential step
        if corner_name == CORNER_ORDER[self._step]:
            self._step += 1
            self._refresh_ui()

        if self._step == 4:
            self._finish_job()
        elif corner_name == CORNER_ORDER[self._step - 1] and self._step < 4:
            next_corner = CORNER_ORDER[self._step]
            self._info_var.set(
                f"✓ {corner_name} done  →  Press PROCESSING for {next_corner}  ({self._step + 1}/4)")

    def _remeasure_corner(self, corner_name):
        """Re-measure a specific corner (card click handler)."""
        if not self._job_started:
            return
        if self._cam_thread is None or not self._cam_thread.running:
            messagebox.showwarning("No Camera", "Camera not running.")
            return
        # Only allow re-measure on corners already attempted or current step
        corner_idx = CORNER_ORDER.index(corner_name)
        if corner_idx > self._step:
            # Not reached yet — ignore click
            return
        self._do_measure(corner_name=corner_name)

    @staticmethod
    def _beep(fail=False):
        """Cross-platform beep. fail=True → double beep."""
        try:
            import winsound
            if fail:
                winsound.Beep(400, 300)
                time.sleep(0.1)
                winsound.Beep(400, 300)
            else:
                winsound.Beep(880, 150)
        except Exception:
            try:
                import subprocess
                if fail:
                    subprocess.Popen(["aplay", "-q", "/usr/share/sounds/alsa/Front_Left.wav"])
                else:
                    subprocess.Popen(["aplay", "-q", "/usr/share/sounds/alsa/Front_Right.wav"])
            except Exception:
                pass  # silent fallback

    def _lock_inputs(self, locked: bool):
        state = "disabled" if locked else "normal"
        for widget in self.winfo_children():
            self._set_entry_state(widget, state)

    def _set_entry_state(self, widget, state):
        """Recursively find Entry widgets and set their state."""
        try:
            if isinstance(widget, tk.Entry):
                widget.config(state=state)
        except Exception:
            pass
        for child in widget.winfo_children():
            self._set_entry_state(child, state)
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        lot     = self._lot
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        csv_path = os.path.join(OUTPUT_DIR, f"{lot}_{ts}.csv")
        jpg_path = os.path.join(OUTPUT_DIR, f"{lot}_{ts}.jpg")

        save_job_to_csv(lot, self._opt_id, self._corners_gaps, csv_path)
        save_job_image(self._corners_frames, jpg_path)

        # Compute and display alignment analysis
        analysis = compute_alignment(self._corners_gaps)
        self._update_analysis(analysis)

        self._info_var.set(f"✅  All 4 corners complete!  Saved → {lot}_{ts}.csv / .jpg")

    def _new_job(self):
        # ── ถามยืนยันถ้ายังวัดไม่ครบ ──
        if self._job_started and self._step < 4:
            if not messagebox.askyesno(
                "New Job",
                f"Job is not complete ({self._step}/4 corners done).\nStart a new job anyway?",
                icon="warning", parent=self):
                return

        self._job_started = False
        self._step = 0
        self._lot = ""
        self._opt_id = ""
        self._lot_var.set("")
        self._opt_var.set("")
        self._corners_gaps   = {}
        self._corners_frames = {}

        self._lock_inputs(False)

        for corner, w in self._corner_widgets.items():
            for key, (var, lbl) in w["gap_vars"].items():
                var.set("—")
                lbl.config(fg=FG_MAIN)
            w["status_var"].set("Waiting...")
            w["status_lbl"].config(fg=FG_DIM)
            w["card"].config(highlightbackground="#2a3f5f")
            w["hint_lbl"].config(text="")

        self._info_var.set("Enter LOT and OPT ID, then press PROCESSING")
        self._update_analysis({"valid": False})
        self._refresh_ui()

    # ====================================================
    #  CLEANUP
    # ====================================================
    def _on_close(self):
        self._stop_camera()
        self.destroy()


# -------------------------------------------------------
#  ENTRY POINT
# -------------------------------------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()

