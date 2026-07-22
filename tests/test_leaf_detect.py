"""
Temporary test script — delete after verification.

Uses SAM (Segment Anything Model) to detect individual leaves in a plant
image. Produces two outputs:
  <image_stem>_detected.png   — annotated image with coloured boxes per leaf
  <image_stem>_hierarchy.png  — leaf crops ranked by SAM confidence score

Usage:  python test_leaf_detect.py <image_path>
"""

import sys
import cv2
import numpy as np
from pathlib import Path

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
SAM_CHECKPOINT  = BASE_DIR / "sam_vit_b.pth"
SAM_MODEL_TYPE  = "vit_b"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

MIN_LEAF_FRACTION = 0.005
MAX_LEAF_FRACTION = 0.80
MIN_GREENNESS     = 0.25

THUMB_SIZE = 160        # px — size of each leaf thumbnail in the hierarchy
COLS       = 4          # thumbnails per row
HEADER_H   = 40         # px — height of the title bar
FOOTER_H   = 36         # px — height of the label bar under each thumbnail
BG_COLOUR  = (30, 30, 30)

BOX_COLOURS = [
    (0,255,0),(255,128,0),(0,200,255),(255,0,128),
    (128,0,255),(0,255,200),(255,255,0),(0,128,255),
]

HSV_GREEN_LOWER = np.array([15,  25,  25],  dtype=np.uint8)
HSV_GREEN_UPPER = np.array([100, 255, 255], dtype=np.uint8)


# ── SAM ───────────────────────────────────────────────────────────────────────
def load_sam():
    if not SAM_CHECKPOINT.exists():
        print(f"ERROR: SAM checkpoint not found at {SAM_CHECKPOINT}")
        sys.exit(1)
    print(f"Loading SAM ({SAM_MODEL_TYPE}) on {DEVICE}...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(DEVICE)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=500,
    )
    print("SAM loaded.")
    return generator


def is_green(img_rgb, mask_bool):
    img_bgr    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv        = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, HSV_GREEN_LOWER, HSV_GREEN_UPPER)
    total      = np.count_nonzero(mask_bool)
    return 0.0 if total == 0 else np.count_nonzero(green_mask[mask_bool]) / total


def detect_leaves(img_rgb, generator):
    h, w  = img_rgb.shape[:2]
    total = h * w

    print("Running SAM... (may take 10-30 seconds)")
    masks = generator.generate(img_rgb)
    print(f"SAM produced {len(masks)} raw segments — filtering for leaves...")

    leaves = []
    for m in masks:
        area_pct = m['area'] / total
        if area_pct < MIN_LEAF_FRACTION or area_pct > MAX_LEAF_FRACTION:
            continue
        mask_bool = m['segmentation']
        if is_green(img_rgb, mask_bool) < MIN_GREENNESS:
            continue
        x, y, bw, bh = (int(v) for v in m['bbox'])
        leaves.append({
            'area_px'      : m['area'],
            'area_pct'     : area_pct * 100,
            'confidence'   : float(m['predicted_iou']),
            'bbox'         : (x, y, bw, bh),
            'mask'         : mask_bool,
        })

    # Assign display ids sorted by area (for the annotated image)
    leaves.sort(key=lambda l: l['area_px'], reverse=True)
    for i, leaf in enumerate(leaves, start=1):
        leaf['id'] = i

    return leaves


# ── Annotated detection image ─────────────────────────────────────────────────
def annotate(img_rgb, leaves):
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()

    for leaf in leaves:
        colour        = BOX_COLOURS[(leaf['id'] - 1) % len(BOX_COLOURS)]
        x, y, bw, bh = leaf['bbox']

        overlay = out.copy()
        overlay[leaf['mask']] = colour
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), colour, 2)

        label = f"#{leaf['id']}  conf:{leaf['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y - th - 6, 0)
        cv2.rectangle(out, (x, ty), (x + tw + 6, ty + th + 8), colour, cv2.FILLED)
        cv2.putText(out, label, (x + 3, ty + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (out.shape[1], 32), BG_COLOUR, cv2.FILLED)
    cv2.putText(out, f"Leaves detected: {len(leaves)}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ── Hierarchy grid ranked by confidence ──────────────────────────────────────
def save_hierarchy(img_rgb, leaves, out_path):
    # Sort by confidence descending — highest confidence at top-left
    ranked = sorted(leaves, key=lambda l: l['confidence'], reverse=True)

    n_leaves = len(ranked)
    rows     = (n_leaves + COLS - 1) // COLS
    cell_h   = THUMB_SIZE + FOOTER_H
    canvas_w = COLS * THUMB_SIZE
    canvas_h = HEADER_H + rows * cell_h

    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOUR, dtype=np.uint8)

    # Title bar
    cv2.putText(canvas, "Leaves ranked by SAM confidence (highest → lowest)",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

    for rank, leaf in enumerate(ranked):
        col = rank % COLS
        row = rank // COLS

        x0 = col * THUMB_SIZE
        y0 = HEADER_H + row * cell_h

        # Crop the leaf from the original image
        bx, by, bw, bh = leaf['bbox']
        crop = img_rgb[by:by+bh, bx:bx+bw]
        if crop.size == 0:
            continue

        # Resize crop to thumbnail
        thumb = cv2.resize(crop, (THUMB_SIZE, THUMB_SIZE))
        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)

        # Draw coloured border matching the detection colour
        colour = BOX_COLOURS[(leaf['id'] - 1) % len(BOX_COLOURS)]
        cv2.rectangle(thumb_bgr, (0, 0), (THUMB_SIZE-1, THUMB_SIZE-1), colour, 3)

        # Rank badge top-left
        badge = f"#{rank+1}"
        cv2.rectangle(thumb_bgr, (0, 0), (32, 22), colour, cv2.FILLED)
        cv2.putText(thumb_bgr, badge, (3, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        canvas[y0:y0+THUMB_SIZE, x0:x0+THUMB_SIZE] = thumb_bgr

        # Footer label: leaf id + confidence score
        footer_y = y0 + THUMB_SIZE
        cv2.rectangle(canvas, (x0, footer_y), (x0+THUMB_SIZE, footer_y+FOOTER_H),
                      colour, cv2.FILLED)
        label = f"Leaf {leaf['id']}  conf: {leaf['confidence']:.3f}"
        cv2.putText(canvas, label, (x0 + 4, footer_y + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)
    print(f"Saved hierarchy → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run(image_path):
    path    = Path(image_path)
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        print(f"ERROR: cannot read {image_path}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"\nImage: {path.name}  ({img_rgb.shape[1]}x{img_rgb.shape[0]})")

    generator = load_sam()
    leaves    = detect_leaves(img_rgb, generator)

    print(f"\nLeaves detected: {len(leaves)}")
    for leaf in leaves:
        x, y, bw, bh = leaf['bbox']
        print(f"  Leaf {leaf['id']}: {bw}x{bh}px  conf:{leaf['confidence']:.3f}  {leaf['area_pct']:.1f}% of image")

    if not leaves:
        print("No leaves found. Try lowering MIN_GREENNESS or MIN_LEAF_FRACTION.")
        sys.exit(0)

    # Output 1 — annotated detection image
    annotated   = annotate(img_rgb, leaves)
    detect_path = str(path.with_stem(path.stem + "_detected"))
    cv2.imwrite(detect_path, annotated)
    print(f"Saved detected  → {detect_path}")

    # Output 2 — confidence hierarchy grid
    hier_path = str(path.with_stem(path.stem + "_hierarchy"))
    save_hierarchy(img_rgb, leaves, hier_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_leaf_detect.py <image_path>")
        sys.exit(0)
    run(sys.argv[1])
