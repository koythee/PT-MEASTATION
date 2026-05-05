"""
Rectangle Detector – DB Alignment
==================================
Modes
-----
  LIVE    : camera streams to canvas in real-time (no detection)
  FROZEN  : single frame captured; detection overlay drawn
  DETECTED: same frame, detection result locked + shown in results panel

Buttons
-------
  [?? Open Camera]   ? open camera index selector, start streaming
  [? Freeze / Detect] ? capture current frame & run detection
  [? Live (Reset)]   ? discard result, back to live stream
  [?? Open Image]    ? load static image file (no camera needed)
  [?? Re-detect]     ? re-run detection on current frozen frame
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


def save_gap_to_csv(results, file_path="gap_results.csv"):
    if not results:
        return

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "rect_id",
                "center_x",
                "center_y",
                "gap_h_px",
                "gap_v_px",
                "gap_h_um",
                "gap_v_um",
                # Circle columns
                "circle_cx",
                "circle_cy",
                "circle_r_px",
                "circle_gap_Hpos_px",
                "circle_gap_Hpos_um",
                "circle_gap_Hneg_px",
                "circle_gap_Hneg_um",
                "circle_gap_Vpos_px",
                "circle_gap_Vpos_um",
                "circle_gap_Vneg_px",
                "circle_gap_Vneg_um",
            ])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        for i, r in enumerate(results):
            gap = r.get("gap", {})
            cx, cy = r.get("center", (None, None))

            # Circle data
            circ = r.get("circle", {})
            circ_cx = circ.get("center", (None, None))[0] if circ else None
            circ_cy = circ.get("center", (None, None))[1] if circ else None
            circ_r  = circ.get("radius") if circ else None
            circ_gaps = circ.get("gaps", {}) if circ else {}

            def cg(label, field):
                entry = circ_gaps.get(label, {})
                return entry.get(field)

            writer.writerow([
                timestamp,
                i + 1,
                cx,
                cy,
                gap.get("horizontal_px"),
                gap.get("vertical_px"),
                gap.get("horizontal_um"),
                gap.get("vertical_um"),
                circ_cx,
                circ_cy,
                circ_r,
                cg("H+", "px"),
                cg("H+", "um"),
                cg("H-", "px"),
                cg("H-", "um"),
                cg("V+", "px"),
                cg("V+", "um"),
                cg("V-", "px"),
                cg("V-", "um"),
            ])


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
#  ZOOM CANVAS
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
#  CAMERA THREAD
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
#  MAIN APPLICATION
# -------------------------------------------------------
# App states
STATE_IDLE     = "IDLE"       # no camera, no image
STATE_LIVE     = "LIVE"       # camera streaming
STATE_FROZEN   = "FROZEN"     # frame captured, detection done
STATE_STATIC   = "STATIC"     # static image loaded


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rectangle Detector – DB Alignment")
        self.configure(bg="#1a1a2e")
        self.resizable(True, True)
        self.geometry("1360x820")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.cfg       = dict(DEFAULT_CONFIG)
        self.state     = STATE_IDLE
        self.raw_frame = None          # frozen / static frame
        self._cam_thread: CameraThread | None = None
        self._live_after = None        # after() id for live loop

        self._build_ui()
        self._refresh_buttons()

    # -------------------------------------------------
    #  UI BUILD
    # -------------------------------------------------
    def _build_ui(self):
        # -- Toolbar --
        bar = tk.Frame(self, bg="#16213e", pady=6)
        bar.pack(fill="x")

        btn_kw = dict(relief="flat", font=("Consolas",10,"bold"),
                      padx=12, pady=5, cursor="hand2")

        self._btn_cam    = tk.Button(bar, text="?? Open Camera",
                                     command=self._open_camera,
                                     bg="#0f3460", fg="#e0e0e0",
                                     activebackground="#1a6faf", **btn_kw)
        self._btn_cam.pack(side="left", padx=6)

        self._btn_freeze = tk.Button(bar, text="? Freeze / Detect",
                                     command=self._freeze_detect,
                                     bg="#1a4f2a", fg="#80ff99",
                                     activebackground="#2a7a3a", **btn_kw)
        self._btn_freeze.pack(side="left", padx=2)

        self._btn_live   = tk.Button(bar, text="? Live (Reset)",
                                     command=self._go_live,
                                     bg="#3a2a0a", fg="#ffcc44",
                                     activebackground="#5a4010", **btn_kw)
        self._btn_live.pack(side="left", padx=2)

        self._btn_img    = tk.Button(bar, text="?? Open Image",
                                     command=self._open_image,
                                     bg="#0f3460", fg="#e0e0e0",
                                     activebackground="#1a6faf", **btn_kw)
        self._btn_img.pack(side="left", padx=2)

        self._btn_redet  = tk.Button(bar, text="?? Re-detect",
                                     command=self._run_detect,
                                     bg="#2d2d54", fg="#aaaaff",
                                     activebackground="#3d3d74", **btn_kw)
        self._btn_redet.pack(side="left", padx=2)

        # status label
        self._status_var = tk.StringVar(value="? IDLE")
        tk.Label(bar, textvariable=self._status_var,
                 bg="#16213e", fg="#556677",
                 font=("Consolas",9,"bold")).pack(side="right", padx=14)

        tk.Label(bar, text="Scroll=Zoom  Drag=Pan",
                 bg="#16213e", fg="#334455",
                 font=("Consolas",8)).pack(side="right", padx=6)

        # -- Main area --
        main = tk.Frame(self, bg="#1a1a2e")
        main.pack(fill="both", expand=True, padx=8, pady=8)

        self.zoom_canvas = ZoomCanvas(main, bd=1, relief="sunken")
        self.zoom_canvas.pack(side="left", fill="both", expand=True)

        right = tk.Frame(main, bg="#1a1a2e", width=305)
        right.pack(side="right", fill="y", padx=(8,0))
        right.pack_propagate(False)

        self._build_config(right)
        self._build_results(right)

    # -------------------------------------------------
    #  CONFIG PANEL
    # -------------------------------------------------
    def _build_config(self, parent):
        lf = tk.LabelFrame(parent, text=" ? Config ", bg="#16213e", fg="#7eb8f7",
                            font=("Consolas",10,"bold"), bd=1, relief="groove")
        lf.pack(fill="x", pady=(0,6))

        params = [
            ("Threshold",      "thresh_value",   0,    255,     1),
            ("Morph Size",     "morph_size",      1,    31,      2),
            ("Min Area",       "min_area",        0,    5000,  100),
            ("Max Area",       "max_area",     1000, 1000000, 1000),
            ("Target W (A)",   "target_w",        1,    2000,    1),
            ("Target H (B)",   "target_h",        1,    2000,    1),
            ("Size Tolerance", "size_tolerance",  0,    200,     1),
            ("Trans. Gap",     "transition_gap",  1,    20,      1),
        ]
        self._sliders = {}
        for label, key, lo, hi, _ in params:
            row = tk.Frame(lf, bg="#16213e")
            row.pack(fill="x", padx=6, pady=2)
            tk.Label(row, text=label, bg="#16213e", fg="#a0c4ff",
                     font=("Consolas",8), width=15, anchor="w").pack(side="left")
            var = tk.DoubleVar(value=self.cfg[key])
            ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal",
                      length=100,
                      command=lambda v, k=key, dv=var: self._on_slider(k, dv)
                      ).pack(side="left")
            lbl = tk.Label(row, text=str(self.cfg[key]), bg="#16213e", fg="#fff",
                           font=("Consolas",8), width=7)
            lbl.pack(side="left")
            self._sliders[key] = (var, lbl)

        # px per µm
        um_row = tk.Frame(lf, bg="#16213e")
        um_row.pack(fill="x", padx=6, pady=3)
        tk.Label(um_row, text="px per µm", bg="#16213e", fg="#ffd700",
                 font=("Consolas",8), width=15, anchor="w").pack(side="left")
        self._um_var = tk.StringVar(value=str(self.cfg["px_per_um"]))
        e = tk.Entry(um_row, textvariable=self._um_var, width=9,
                     bg="#0d1117", fg="#ffd700", insertbackground="#ffd700",
                     font=("Consolas",9), relief="flat")
        e.pack(side="left", padx=4)
        e.bind("<Return>",   lambda _: self._apply_um())
        e.bind("<FocusOut>", lambda _: self._apply_um())
        tk.Label(um_row, text="(0=off)", bg="#16213e", fg="#556",
                 font=("Consolas",7)).pack(side="left")

        tk.Button(lf, text="Reset Defaults", command=self._reset_config,
                  bg="#2d2d54", fg="#aaa", font=("Consolas",8),
                  relief="flat", pady=3, cursor="hand2").pack(fill="x", padx=6, pady=4)

    # -------------------------------------------------
    #  RESULTS PANEL
    # -------------------------------------------------
    def _build_results(self, parent):
        lf = tk.LabelFrame(parent, text=" ?? Results ", bg="#16213e", fg="#7eb8f7",
                            font=("Consolas",10,"bold"), bd=1, relief="groove")
        lf.pack(fill="both", expand=True)
        self.rtxt = tk.Text(lf, bg="#0d1117", fg="#c9d1d9",
                            font=("Consolas",8), relief="flat",
                            wrap="none", state="disabled")
        sby = ttk.Scrollbar(lf, command=self.rtxt.yview)
        sbx = ttk.Scrollbar(lf, orient="horizontal", command=self.rtxt.xview)
        self.rtxt.configure(yscrollcommand=sby.set, xscrollcommand=sbx.set)
        sby.pack(side="right",  fill="y")
        sbx.pack(side="bottom", fill="x")
        self.rtxt.pack(fill="both", expand=True, padx=2, pady=2)

        for tag, (color, bold) in {
            "hdr":    ("#7eb8f7", True),
            "key":    ("#79c0ff", False),
            "val":    ("#ffa657", False),
            "w2b":    ("#ff7b7b", False),
            "b2w":    ("#79aeff", False),
            "gap":    ("#00ffcc", False),
            "ang":    ("#ffd700", False),
            "start":  ("#56d364", False),
            "end":    ("#388bfd", False),
            "sep":    ("#555555", False),
            "info":   ("#aaaaaa", False),
            "circle": ("#00c8ff", True),
            "circ_v": ("#ff55dd", False),
        }.items():
            self.rtxt.tag_config(tag, foreground=color,
                                 font=("Consolas",8,"bold") if bold else ("Consolas",8))

    # -------------------------------------------------
    #  CAMERA CONTROL
    # -------------------------------------------------
    def _open_camera(self):
        idx = simpledialog.askinteger(
            "Camera Index",
            "Enter camera index (0, 1, 2 …):",
            initialvalue=1, minvalue=0, maxvalue=10, parent=self)
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

        self._set_state(STATE_LIVE)
        self._live_loop()

    def _stop_camera(self):
        if self._live_after is not None:
            self.after_cancel(self._live_after)
            self._live_after = None
        if self._cam_thread is not None:
            self._cam_thread.stop()
            self._cam_thread.join(timeout=1.5)
            self._cam_thread = None

    def _live_loop(self):
        """Called every ~33ms while in LIVE state."""
        if self.state != STATE_LIVE:
            return
        if self._cam_thread is None or not self._cam_thread.running:
            err = getattr(self._cam_thread, "error", "Camera disconnected")
            self._set_state(STATE_IDLE)
            messagebox.showerror("Camera Error", err or "Camera stopped unexpectedly")
            return

        f = self._cam_thread.get_frame()
        if f is not None:
            self.zoom_canvas.update_live(f)

        self._live_after = self.after(33, self._live_loop)

    # -------------------------------------------------
    #  FREEZE / DETECT / LIVE
    # -------------------------------------------------
    def _freeze_detect(self):
        if self.state == STATE_LIVE and self._cam_thread:
            f = self._cam_thread.get_frame()
            if f is None:
                messagebox.showwarning("Warning", "No frame available yet.")
                return
            if self._live_after is not None:
                self.after_cancel(self._live_after)
                self._live_after = None
            self.raw_frame = f
            self._set_state(STATE_FROZEN)
            self._run_detect()
        elif self.state in (STATE_FROZEN, STATE_STATIC):
            self._run_detect()

    def _go_live(self):
        if self._cam_thread and self._cam_thread.running:
            self.raw_frame = None
            self._set_state(STATE_LIVE)
            self._clear_results()
            self._live_loop()
        else:
            self._set_state(STATE_IDLE)

    # -------------------------------------------------
    #  STATIC IMAGE
    # -------------------------------------------------
    def _open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.bmp *.png *.jpg *.jpeg *.tif *.tiff"), ("All","*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Cannot read:\n{path}")
            return
        self._stop_camera()
        self.raw_frame = img
        self._set_state(STATE_STATIC)
        self._run_detect()

    # -------------------------------------------------
    #  DETECTION
    # -------------------------------------------------
    def _run_detect(self):
        if self.raw_frame is None:
            return
        out, results = detect_on_frame(self.raw_frame, self.cfg)
        self.zoom_canvas.set_image(out)
        self._show_results(results)
        save_gap_to_csv(results)

    # -------------------------------------------------
    #  STATE MACHINE
    # -------------------------------------------------
    def _set_state(self, new_state):
        self.state = new_state
        colors = {
            STATE_IDLE:    ("#556677", "? IDLE"),
            STATE_LIVE:    ("#00cc66", "? LIVE"),
            STATE_FROZEN:  ("#ffaa00", "? FROZEN"),
            STATE_STATIC:  ("#7eb8f7", "?? IMAGE"),
        }
        col, label = colors.get(new_state, ("#556677", "?"))
        self._status_var.set(label)
        for w in self.winfo_children():
            if isinstance(w, tk.Frame):
                for ww in w.winfo_children():
                    if isinstance(ww, tk.Label) and ww.cget("textvariable") == str(self._status_var):
                        ww.config(fg=col)
        self._refresh_buttons()

    def _refresh_buttons(self):
        s = self.state
        can_freeze = s == STATE_LIVE
        can_live   = s in (STATE_FROZEN,) and self._cam_thread is not None
        can_redet  = s in (STATE_FROZEN, STATE_STATIC)

        def state_of(active):
            return "normal" if active else "disabled"

        self._btn_freeze.config(state=state_of(can_freeze))
        self._btn_live.config(  state=state_of(can_live))
        self._btn_redet.config( state=state_of(can_redet))

    # -------------------------------------------------
    #  CONFIG HANDLERS
    # -------------------------------------------------
    def _on_slider(self, key, var):
        v = int(var.get())
        if key == "morph_size":
            v = max(1, v | 1)
        self.cfg[key] = v
        self._sliders[key][1].config(text=str(v))
        if self.state in (STATE_FROZEN, STATE_STATIC):
            self._run_detect()

    def _apply_um(self):
        try:
            val = max(0.0, float(self._um_var.get()))
        except ValueError:
            val = 0.0
        self.cfg["px_per_um"] = val
        self._um_var.set(str(val))
        if self.state in (STATE_FROZEN, STATE_STATIC):
            self._run_detect()

    def _reset_config(self):
        self.cfg = dict(DEFAULT_CONFIG)
        for key, (var, lbl) in self._sliders.items():
            var.set(self.cfg[key])
            lbl.config(text=str(self.cfg[key]))
        self._um_var.set(str(self.cfg["px_per_um"]))
        if self.state in (STATE_FROZEN, STATE_STATIC):
            self._run_detect()

    # -------------------------------------------------
    #  RESULTS RENDERER
    # -------------------------------------------------
    def _clear_results(self):
        t = self.rtxt
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.configure(state="disabled")

    def _show_results(self, results):
        t = self.rtxt
        t.configure(state="normal")
        t.delete("1.0", "end")
        ppu = self.cfg["px_per_um"]

        def ins(text, tag=""):
            t.insert("end", text, tag)

        if not results:
            ins("No rectangle matched A×B.\n", "hdr")
            t.configure(state="disabled")
            return

        for i, r in enumerate(results):
            ins(f"-- Rectangle #{i+1} ---------------------\n", "sep")
            cx, cy = r["center"]
            rw, rh = r["rect_wh"]
            ins("Center  : ", "key"); ins(f"({cx}, {cy})\n", "val")
            ins("Size    : ", "key"); ins(f"{rw} × {rh} px\n", "val")
            ins("Angle   : ", "key"); ins(f"{r['angle']}°\n", "ang")

            # Inner rect
            ir = r.get("inner_rect")
            if ir:
                ins("InnerRect: ", "key")
                ins(f"C{ir['center']}  {ir['wh'][0]}×{ir['wh'][1]}px  {ir['angle']}°\n", "val")

            for axis_key, axis_label in [("h_line","Long-axis  (H)"),
                                          ("v_line","Short-axis (V)")]:
                ln    = r[axis_key]
                trans = ln["transitions"]
                gaps  = ln["gaps"]
                gap_map = {(g["from"], g["to"]): g for g in gaps}

                ins(f"\n  {axis_label}\n", "hdr")
                sx2, sy2 = ln["start"]
                ex2, ey2 = ln["end"]
                ins("    Start : ", "key"); ins(f"({sx2:.1f}, {sy2:.1f})\n", "start")
                ins("    End   : ", "key"); ins(f"({ex2:.1f}, {ey2:.1f})\n", "end")
                ins(f"    Transitions ({len(trans)}):\n", "key")

                for n0, pt in enumerate(trans):
                    n1     = n0 + 1
                    tag    = "w2b" if pt["dir"] == "W2B" else "b2w"
                    symbol = "?" if pt["dir"] == "W2B" else "?"
                    ins(f"    {symbol} {n1:2d} ({pt['x']:8.1f}, {pt['y']:8.1f})  {pt['dir']}", tag)
                    g = gap_map.get((n0-1, n0))
                    if g:
                        um = px_to_um(g["px"], ppu)
                        suf = f"  ({um:.2f} µm)" if um is not None else ""
                        ins(f"   ? gap: {g['px']:.1f} px{suf}", "gap")
                    ins("\n")

                if gaps:
                    ins("    -- Gap summary --\n", "sep")
                    for g in gaps:
                        ins(f"    pt{g['from']+1}?pt{g['to']+1} : ", "key")
                        ins(gap_label(g["px"], ppu) + "\n", "gap")

            # -- Circle section --
            circ = r.get("circle")
            if circ:
                ins(f"\n  -- Circle Fit ({circ['points_used']} pts) --\n", "circle")
                ccx, ccy = circ["center"]
                ins("    Center : ", "key")
                ins(f"({ccx}, {ccy})\n", "circ_v")
                ins("    Radius : ", "key")
                R = circ["radius"]
                ppu_v = r["ppu"]
                um_r = px_to_um(R, ppu_v)
                r_suf = f"  ({um_r:.2f} µm)" if um_r is not None else ""
                ins(f"{R:.1f} px{r_suf}\n", "circ_v")

                cg = circ.get("gaps", {})
                if cg:
                    ins("    -- Circle?InnerRect gaps --\n", "sep")
                    for label in ["H+", "H-", "V+", "V-"]:
                        entry = cg.get(label, {})
                        gpx = entry.get("px")
                        gum = entry.get("um")
                        ins(f"    {label} : ", "key")
                        if gpx is None:
                            ins("N/A\n", "info")
                        else:
                            um_str = f"  ({gum:.4f} µm)" if gum is not None else ""
                            ins(f"{gpx:.2f} px{um_str}\n", "gap")
            else:
                ins("\n  -- Circle Fit: insufficient transitions --\n", "info")

            ins("\n")

        t.configure(state="disabled")

    # -------------------------------------------------
    #  CLEANUP
    # -------------------------------------------------
    def _on_close(self):
        self._stop_camera()
        self.destroy()


# -------------------------------------------------------
#  ENTRY POINT
# -------------------------------------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()
