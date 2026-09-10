"""
visualize_topology.py
Draws the filtered lines from the topology CSV onto the PDF page.

FIX: Instead of drawing a single diagonal line from merged start->end
(which creates wrong diagonal lines for Elbow/bent pipe groups),
this version draws each ORIGINAL SEGMENT individually, coloured by
whether its group has a matched pipe tag or not.

Colour coding:
  - Lines WITH a pipe tag  -> green,  pipe tag label shown at midpoint
  - Lines WITHOUT a tag    -> blue/orange
  - tag_start / tag_end    -> red dot + instrument label at endpoints
"""

import os
import cv2
import numpy as np
import fitz   # PyMuPDF
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
PDFS = [
    {
        "label":       "01.Marked",
        "pdf_path":    r"C:\Users\user\Desktop\Work\p&id\data\01.Marked.pdf",
        "topo_csv":    r"C:\Users\user\Desktop\Work\p&id\LE\Results\LE_csv\01.Marked_line_tag_topology.csv",
        "seg_csv":     r"C:\Users\user\Desktop\Work\p&id\LE\Results\LE_csv\01.Marked_line_coordinates_grouped.csv",
        "out_image":   r"C:\Users\user\Desktop\Work\p&id\LE\Results\LE_images\01.Marked_topology_filtered.png",
    },
    {
        "label":       "02.Unmarked",
        "pdf_path":    r"C:\Users\user\Desktop\Work\p&id\data\02.Unmarked.pdf",
        "topo_csv":    r"C:\Users\user\Desktop\Work\p&id\LE\Results\LE_csv\02.Unmarked_line_tag_topology.csv",
        "seg_csv":     r"C:\Users\user\Desktop\Work\p&id\LE\Results\LE_csv\02.Unmarked_line_coordinates_grouped.csv",
        "out_image":   r"C:\Users\user\Desktop\Work\p&id\LE\Results\LE_images\02.Unmarked_topology_filtered.png",
    },
]

DPI        = 300
LINE_W     = 4
DOT_R      = 14
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_W     = 1

COLOR_TAGGED   = (0,   200,  0)    # green  — pipe run with matched tag
COLOR_UNTAGGED = (200,  80,  0)    # orange/blue — no tag matched yet
COLOR_ENDPOINT = (0,    0,  220)   # red    — instrument at endpoint
COLOR_LABEL    = (255, 255, 255)   # white text
# ─────────────────────────────────────────────────────────────────────────────


def rasterise_pdf(pdf_path, dpi=DPI):
    doc  = fitz.open(pdf_path)
    page = doc[0]
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()
    img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img[:, :, ::-1].copy()


def draw_label(img, text, x, y, fg, bg):
    if not text or str(text) in ("nan", "None", "OPEN"):
        return
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_W)
    pad = 3
    cv2.rectangle(img, (int(x)-pad, int(y)-th-pad), (int(x)+tw+pad, int(y)+pad), bg, -1)
    cv2.putText(img, text, (int(x), int(y)), FONT, FONT_SCALE, fg, FONT_W, cv2.LINE_AA)


def visualise(cfg):
    print(f"\nRendering: {cfg['label']} ...")
    img     = rasterise_pdf(cfg["pdf_path"])
    topo_df = pd.read_csv(cfg["topo_csv"])
    seg_df  = pd.read_csv(cfg["seg_csv"])

    # ── Build group_id -> colour & tag info ────────────────────────────
    tag_lookup = {}
    for _, row in topo_df.iterrows():
        has_tag = isinstance(row["line_tag"], str) and str(row["line_tag"]) not in ("nan", "")
        tag_lookup[int(row["Line_ID"])] = {
            "color":     COLOR_TAGGED if has_tag else COLOR_UNTAGGED,
            "line_tag":  row["line_tag"] if has_tag else None,
            "tag_start": str(row.get("tag_start", "OPEN")),
            "tag_end":   str(row.get("tag_end",   "OPEN")),
            "x_start":   float(row["x_start"]),
            "y_start":   float(row["y_start"]),
            "x_end":     float(row["x_end"]),
            "y_end":     float(row["y_end"]),
        }

    # ── Draw each ORIGINAL segment (correct geometry, no diagonals) ───
    for _, seg in seg_df.iterrows():
        gid  = int(seg["Group_ID"])
        info = tag_lookup.get(gid)
        if info is None:
            continue
        p1 = (int(seg["x_start"]), int(seg["y_start"]))
        p2 = (int(seg["x_end"]),   int(seg["y_end"]))
        cv2.line(img, p1, p2, info["color"], LINE_W, cv2.LINE_AA)

    # ── Pipe tag label at centre of the overall group extent ──────────
    for gid, info in tag_lookup.items():
        if info["line_tag"]:
            mx = int((info["x_start"] + info["x_end"]) / 2)
            my = int((info["y_start"] + info["y_end"]) / 2)
            draw_label(img, str(info["line_tag"]), mx, my, COLOR_LABEL, info["color"])

        # Endpoint instrument dots
        for px, py, tag_val in [
            (info["x_start"], info["y_start"], info["tag_start"]),
            (info["x_end"],   info["y_end"],   info["tag_end"]),
        ]:
            if tag_val not in ("OPEN", "nan", "None", ""):
                cv2.circle(img, (int(px), int(py)), DOT_R, COLOR_ENDPOINT, -1, cv2.LINE_AA)
                draw_label(img, tag_val, int(px)+DOT_R+4, int(py)+4, COLOR_LABEL, COLOR_ENDPOINT)

    cv2.imwrite(cfg["out_image"], img)
    print(f"  Saved -> {cfg['out_image']}")
    h, w = img.shape[:2]
    print(f"  Image size: {w} x {h} px")


if __name__ == "__main__":
    for cfg in PDFS:
        visualise(cfg)
    print("\nDone.")
