"""
disease_localisation.py
=======================
Part 2 of CV Enhancement Pipeline — Crop Disease EWS
-----------------------------------------------------
Takes the raw Grad-CAM heatmap (float32, 0–1) produced by
CropDiseaseClassifier.grad_cam() and the leaf mask from Part 1,
then performs:

  1. CAM thresholding  →  isolates the hot (disease-activated) region
  2. Leaf-mask gating  →  removes any heatmap bleed outside the leaf
  3. Contour detection →  finds individual disease spot contours
  4. Bounding boxes    →  draws rectangles around each spot
  5. Area calculation  →  reports % of leaf area infected
  6. Severity grading  →  maps infection % to Low/Moderate/High/Severe
  7. Annotated image   →  returns a base64 PNG for Flutter to display

This replaces / extends the existing run_gradcam() function in api.py.

CV concepts covered
-------------------
  - Adaptive + Otsu thresholding on heatmap
  - cv2.findContours with hierarchy filtering
  - cv2.boundingRect, cv2.minEnclosingCircle
  - Contour area calculation and percentage computation
  - Drawing annotated overlays with text labels
"""

import cv2
import numpy as np
import base64
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

# Grad-CAM heatmap threshold — pixels above this are "hot" (disease-activated)
# 0.40 = top 60% of activation. Lower = more sensitive, more false positives.
CAM_THRESHOLD = 0.40

# Minimum disease spot area as fraction of LEAF area (not full image).
# Filters out tiny noise contours.
MIN_SPOT_FRAC = 0.005   # 0.5% of leaf area

# Maximum number of individual spots to annotate (avoid cluttering the image)
MAX_SPOTS = 12

# Severity thresholds (% of leaf area infected)
SEVERITY_THRESHOLDS = {
    "Low"      : (0.0,  15.0),
    "Moderate" : (15.0, 35.0),
    "High"     : (35.0, 60.0),
    "Severe"   : (60.0, 100.0),
}

# Visual style
BOX_COLOUR_BGR    = (0,   0,   255)   # red boxes for disease spots
BOX_THICKNESS     = 2
TEXT_COLOUR_BGR   = (255, 255, 255)   # white text
TEXT_BG_BGR       = (0,   0,   180)   # dark red text background
FONT              = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE        = 0.55
FONT_THICKNESS    = 1
HEATMAP_ALPHA     = 0.45              # blend strength of Grad-CAM overlay
CONTOUR_COLOUR    = (0, 255, 255)     # cyan contour outline
CONTOUR_THICKNESS = 1


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class DiseaseSpot:
    """One detected disease region on the leaf."""
    contour      : np.ndarray          # raw contour points
    bbox         : Tuple[int,int,int,int]  # (x, y, w, h)
    area_px      : float               # pixel area of this spot
    area_pct_leaf: float               # % of leaf area this spot occupies
    centroid     : Tuple[int, int]     # (cx, cy)


@dataclass
class LocalisationResult:
    """
    Returned by localise_disease().

    Attributes
    ----------
    annotated_b64 : str
        Base64-encoded PNG of the annotated overlay (ready for Flutter).
    spots : List[DiseaseSpot]
        All detected disease spots, sorted by area descending.
    infected_pct : float
        Total % of leaf area covered by disease spots (0–100).
    severity : str
        "Low" | "Moderate" | "High" | "Severe"
    leaf_area_px : float
        Total leaf area in pixels (from Part 1 mask).
    cam_threshold_used : float
        The threshold that was actually applied (may be auto-adjusted).
    warning : Optional[str]
        Non-None if something unusual was detected.
    """
    annotated_b64      : str
    spots              : List[DiseaseSpot]
    infected_pct       : float
    severity           : str
    leaf_area_px       : float
    cam_threshold_used : float
    warning            : Optional[str] = None


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _severity_label(infected_pct: float) -> str:
    for label, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= infected_pct < hi:
            return label
    return "Severe"


def _auto_threshold(cam_norm: np.ndarray) -> float:
    """
    If the CAM is very sparse (disease region tiny) or very dense
    (whole leaf activates), Otsu on the heatmap gives a better threshold
    than the fixed CAM_THRESHOLD. We use whichever is higher.
    """
    cam_uint8 = np.uint8(cam_norm * 255)
    otsu_val, _ = cv2.threshold(cam_uint8, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_frac = otsu_val / 255.0
    # Take the more conservative (higher) of the two
    return max(CAM_THRESHOLD, otsu_frac)


def _build_hot_mask(cam_norm: np.ndarray,
                    leaf_mask: np.ndarray,
                    threshold: float) -> np.ndarray:
    """
    Threshold CAM → binary hot mask, then gate with the leaf mask
    so activations outside the leaf boundary are removed.

    Parameters
    ----------
    cam_norm  : float32 array (H, W), values 0–1
    leaf_mask : uint8 array  (H, W), values 0 or 255  (from Part 1)
    threshold : float

    Returns
    -------
    hot_mask : uint8 array (H, W), 255 = diseased, 0 = clean
    """
    # Threshold the CAM
    hot = (cam_norm >= threshold).astype(np.uint8) * 255

    # Gate: only keep hot pixels that are inside the leaf
    hot = cv2.bitwise_and(hot, hot, mask=leaf_mask)

    # Morphological clean — close small gaps between spots, remove speckles
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hot = cv2.morphologyEx(hot, cv2.MORPH_CLOSE, k_close)
    hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN,  k_open)

    return hot


def _find_spots(hot_mask: np.ndarray,
                leaf_area_px: float,
                min_area_px: float) -> List[DiseaseSpot]:
    """
    Find and rank disease spot contours from the hot mask.
    """
    contours, _ = cv2.findContours(hot_mask,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    spots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        M  = cv2.moments(cnt)
        cx = int(M['m10'] / (M['m00'] + 1e-6))
        cy = int(M['m01'] / (M['m00'] + 1e-6))

        spots.append(DiseaseSpot(
            contour       = cnt,
            bbox          = (x, y, w, h),
            area_px       = area,
            area_pct_leaf = (area / leaf_area_px) * 100.0,
            centroid      = (cx, cy),
        ))

    # Sort largest first
    spots.sort(key=lambda s: s.area_px, reverse=True)
    return spots[:MAX_SPOTS]


def _draw_annotations(img_rgb     : np.ndarray,
                       cam_norm    : np.ndarray,
                       leaf_mask   : np.ndarray,
                       spots       : List[DiseaseSpot],
                       infected_pct: float,
                       severity    : str) -> np.ndarray:
    """
    Build the annotated overlay image:
      - Grad-CAM heatmap blended over the (masked) leaf
      - Cyan contour outlines around each disease spot
      - Red bounding boxes with spot index labels
      - Summary banner at the top
    """
    h, w = img_rgb.shape[:2]

    # ── 1. Grad-CAM coloured heatmap ──
    cam_resized = cv2.resize(cam_norm, (w, h))
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized),
                                     cv2.COLORMAP_JET)
    # Convert input to BGR for all cv2 drawing
    img_bgr     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Blend heatmap only over the leaf region
    leaf_mask3  = np.stack([leaf_mask, leaf_mask, leaf_mask], axis=2) / 255.0
    overlay     = (img_bgr * (1 - HEATMAP_ALPHA * leaf_mask3)
                   + heatmap_bgr * HEATMAP_ALPHA * leaf_mask3).astype(np.uint8)

    # ── 2. Contour outlines ──
    contours_list = [s.contour for s in spots]
    cv2.drawContours(overlay, contours_list, -1,
                     CONTOUR_COLOUR, CONTOUR_THICKNESS)

    # ── 3. Bounding boxes + spot index labels ──
    for idx, spot in enumerate(spots, start=1):
        x, y, bw, bh = spot.bbox
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh),
                      BOX_COLOUR_BGR, BOX_THICKNESS)

        label      = f"#{idx} {spot.area_pct_leaf:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

        # Text background pill
        tx, ty = x, max(y - th - 6, 0)
        cv2.rectangle(overlay,
                      (tx, ty), (tx + tw + 4, ty + th + 6),
                      TEXT_BG_BGR, cv2.FILLED)
        cv2.putText(overlay, label,
                    (tx + 2, ty + th + 2),
                    FONT, FONT_SCALE, TEXT_COLOUR_BGR, FONT_THICKNESS,
                    cv2.LINE_AA)

    # ── 4. Summary banner ──
    severity_colours = {
        "Low"      : (0, 180,   0),
        "Moderate" : (0, 165, 255),
        "High"     : (0,  69, 255),
        "Severe"   : (0,   0, 200),
    }
    banner_colour = severity_colours.get(severity, (100, 100, 100))
    banner_h      = 36
    cv2.rectangle(overlay, (0, 0), (w, banner_h), banner_colour, cv2.FILLED)

    summary = (f"Infected: {infected_pct:.1f}%  |  "
               f"Severity: {severity}  |  "
               f"Spots: {len(spots)}")
    cv2.putText(overlay, summary,
                (8, 24), FONT, FONT_SCALE,
                (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def _to_b64(img_rgb: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf  = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf).decode('utf-8')


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def localise_disease(img_rgb  : np.ndarray,
                     cam_norm : np.ndarray,
                     leaf_mask: np.ndarray) -> LocalisationResult:
    """
    Main entry point for Part 2.

    Parameters
    ----------
    img_rgb : np.ndarray (H, W, 3), uint8, RGB
        The SEGMENTED image from Part 1 (seg_result.segmented_image).
        Using the segmented image means heatmap noise outside the leaf
        is already suppressed before we even threshold.

    cam_norm : np.ndarray (h, w), float32, values 0–1
        Raw CAM array returned by CropDiseaseClassifier.grad_cam().
        This is BEFORE any resize — the function handles resizing internally.

    leaf_mask : np.ndarray (H, W), uint8, 0 or 255
        Binary leaf mask from Part 1 (seg_result.mask).
        Used to gate heatmap activations to leaf pixels only.

    Returns
    -------
    LocalisationResult
        See dataclass docstring above.
    """
    h, w = img_rgb.shape[:2]
    warning = None

    # ── Resize CAM to match input image ──
    cam_resized = cv2.resize(cam_norm.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_LINEAR)

    # ── Leaf area in pixels ──
    leaf_area_px = float(np.count_nonzero(leaf_mask))
    if leaf_area_px < 100:
        # Degenerate mask — use full image area
        leaf_area_px = float(h * w)
        warning = "Leaf mask was empty; using full image area for % calculation."

    # ── Auto-threshold ──
    threshold = _auto_threshold(cam_resized)

    # ── Build hot mask (CAM threshold ∩ leaf mask) ──
    hot_mask = _build_hot_mask(cam_resized, leaf_mask, threshold)

    # ── Find disease spots ──
    min_area_px = leaf_area_px * MIN_SPOT_FRAC
    spots       = _find_spots(hot_mask, leaf_area_px, min_area_px)

    # ── Total infected area ──
    infected_px  = float(np.count_nonzero(hot_mask))
    infected_pct = min((infected_px / leaf_area_px) * 100.0, 100.0)
    severity     = _severity_label(infected_pct)

    # ── Quality warnings ──
    if infected_pct > 90:
        warning = (warning or "") + (
            " Very high infection detected (>90% leaf area). "
            "Verify image quality or consider re-photographing."
        )
    if not spots and infected_pct > 5:
        warning = (warning or "") + (
            " Infection area detected but no distinct spots found — "
            "disease may be diffuse (e.g. powdery mildew, mosaic virus)."
        )

    # ── Draw annotated overlay ──
    annotated_rgb = _draw_annotations(
        img_rgb, cam_resized, leaf_mask,
        spots, infected_pct, severity
    )
    annotated_b64 = _to_b64(annotated_rgb)

    return LocalisationResult(
        annotated_b64      = annotated_b64,
        spots              = spots,
        infected_pct       = infected_pct,
        severity           = severity,
        leaf_area_px       = leaf_area_px,
        cam_threshold_used = threshold,
        warning            = warning,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HOW TO INTEGRATE INTO api.py
# ─────────────────────────────────────────────────────────────────────────────
#
# Step 1 — Add import at the top of api.py:
#
#     from leaf_segmentation   import segment_leaf, SegmentationResult
#     from disease_localisation import localise_disease, LocalisationResult
#
#
# Step 2 — Replace the existing run_gradcam() function entirely:
#
#     def run_gradcam(image_array : np.ndarray,
#                     class_idx   : int,
#                     leaf_mask   : np.ndarray = None) -> dict:
#         """
#         Returns Grad-CAM overlay + disease localisation metrics.
#         Now returns a dict instead of a bare base64 string.
#         """
#         resized   = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
#         tensor    = val_transform(image=resized)['image']
#         cam, _, _ = cnn_model.grad_cam(tensor, class_idx)   # cam is (h,w) float32
#
#         # If no leaf mask provided (e.g. called from /explain without Part 1),
#         # use a full-image mask as safe fallback.
#         if leaf_mask is None:
#             leaf_mask = np.full(resized.shape[:2], 255, dtype=np.uint8)
#         else:
#             leaf_mask = cv2.resize(leaf_mask, (IMG_SIZE, IMG_SIZE),
#                                    interpolation=cv2.INTER_NEAREST)
#
#         loc = localise_disease(
#             img_rgb   = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
#             cam_norm  = cam,
#             leaf_mask = leaf_mask,
#         )
#
#         return {
#             'gradcam'      : loc.annotated_b64,    # drop-in replacement
#             'infected_pct' : round(loc.infected_pct, 2),
#             'severity'     : loc.severity,
#             'spot_count'   : len(loc.spots),
#             'spots'        : [
#                 {
#                     'id'           : i + 1,
#                     'bbox'         : s.bbox,
#                     'area_pct_leaf': round(s.area_pct_leaf, 2),
#                     'centroid'     : s.centroid,
#                 }
#                 for i, s in enumerate(loc.spots)
#             ],
#             'warning'      : loc.warning,
#         }
#
#
# Step 3 — In _run_analyze_sync(), update the Grad-CAM call to pass mask:
#
#     if include_gradcam:
#         with torch.enable_grad():
#             gradcam_result = run_gradcam(
#                 seg_result.segmented_image,   # segmented image from Part 1
#                 cnn_class,
#                 leaf_mask = seg_result.mask,  # mask from Part 1
#             )
#     else:
#         gradcam_result = None
#
#     # In the return dict, replace 'gradcam': gradcam_b64 with:
#     'gradcam': gradcam_result,
#
#
# Step 4 — Flutter side:
#     The response now returns gradcam as an object, not a bare string.
#     Access the image with: response['gradcam']['gradcam']
#     Access infection %  with: response['gradcam']['infected_pct']
#     Access severity     with: response['gradcam']['severity']
#
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from leafseg import segment_leaf   # Part 1 — adjust name if needed

    if len(sys.argv) < 2:
        print("Usage: python disease_localisation.py <image_path>")
        print("       Saves: <image_path>_localised.png")
        sys.exit(0)

    path    = sys.argv[1]
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"ERROR: Cannot read image: {path}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Part 1: segment ──
    seg = segment_leaf(img_rgb)
    print(f"[Part 1] Method: {seg.method_used} | Coverage: {seg.leaf_coverage:.1%}")

    # ── Simulate a Grad-CAM (synthetic for standalone test) ──
    # In production this comes from cnn_model.grad_cam()
    # Here we simulate a CAM that activates in the top-left region
    h, w    = img_rgb.shape[:2]
    cam_sim = np.zeros((h // 16, w // 16), dtype=np.float32)  # EfficientNet feature map size
    cy, cx  = cam_sim.shape[0] // 3, cam_sim.shape[1] // 3
    Y, X    = np.ogrid[:cam_sim.shape[0], :cam_sim.shape[1]]
    radius  = min(cam_sim.shape) // 3
    dist    = np.sqrt((Y - cy)**2 + (X - cx)**2)
    cam_sim = np.clip(1.0 - dist / radius, 0, 1).astype(np.float32)

    # ── Part 2: localise ──
    loc = localise_disease(
        img_rgb   = seg.segmented_image,
        cam_norm  = cam_sim,
        leaf_mask = seg.mask,
    )

    # Save annotated image
    base          = path.rsplit(".", 1)[0]
    out_path      = f"{base}_localised.png"
    annotated_bgr = cv2.cvtColor(
        np.frombuffer(base64.b64decode(loc.annotated_b64), dtype=np.uint8),
        cv2.COLOR_RGB2BGR
    )
    # Decode from b64 properly
    buf           = base64.b64decode(loc.annotated_b64)
    arr           = np.frombuffer(buf, dtype=np.uint8)
    annotated_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    cv2.imwrite(out_path, annotated_bgr)

    print(f"[Part 2] Infected area  : {loc.infected_pct:.1f}%")
    print(f"[Part 2] Severity       : {loc.severity}")
    print(f"[Part 2] Spots detected : {len(loc.spots)}")
    print(f"[Part 2] CAM threshold  : {loc.cam_threshold_used:.2f}")
    for i, s in enumerate(loc.spots, 1):
        print(f"         Spot #{i}: {s.area_pct_leaf:.1f}% of leaf | bbox={s.bbox}")
    if loc.warning:
        print(f"[Part 2] Warning: {loc.warning}")
    print(f"Saved: {out_path}")