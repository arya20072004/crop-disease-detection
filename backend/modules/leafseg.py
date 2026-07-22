"""
leaf_segmentation.py
====================
Part 1 of CV Enhancement Pipeline — Crop Disease EWS
-----------------------------------------------------
Segments the leaf from background noise (soil, sky, pots, hands)
before the image is forwarded to EfficientNet-B4.

Strategy (cascaded fallback):
  1. HSV green-range masking  →  fastest, works for most clean field shots
  2. GrabCut refinement       →  kicks in when HSV mask is too noisy or sparse
  3. Contour selection        →  picks the single largest connected blob
  4. Mask application         →  blends leaf onto a neutral grey background

Why grey background (not black/white)?
  ImageNet normalisation (mean=[0.485,0.456,0.406]) maps grey ~= zero
  after normalisation, so it contributes no learned signal to EfficientNet.

Usage (drop-in, standalone):
    from leaf_segmentation import segment_leaf, SegmentationResult

Integration with api.py:
    Replace the "Stage 1 — CNN" block's `resized` line with the call shown
    at the bottom of this file.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# ─────────────────────────────────────────────
# Config — tweak these without touching logic
# ─────────────────────────────────────────────

# HSV green range — covers healthy + diseased (yellow-green to dark green)
HSV_LOWER_GREEN = np.array([20,  30,  30],  dtype=np.uint8)   # hue 20° (yellow-green)
HSV_UPPER_GREEN = np.array([100, 255, 255], dtype=np.uint8)   # hue 100° (teal-green)

# Minimum fraction of image area the HSV mask must cover to be trusted.
# If below this, the leaf may be dry/brown — trigger GrabCut fallback.
HSV_MIN_COVERAGE  = 0.04   # 4 %

# GrabCut iterations — more = cleaner mask, slower
GRABCUT_ITERS = 5

# Minimum contour area as fraction of image — ignores tiny noise blobs
MIN_CONTOUR_FRAC = 0.01    # 1 %

# Background fill colour (RGB) — neutral grey, near ImageNet mean
BG_FILL_RGB = (128, 128, 128)

# Morphological kernel sizes
MORPH_OPEN_KSIZE  = 5   # removes small HSV speckles
MORPH_CLOSE_KSIZE = 15  # fills holes inside the leaf mask
MORPH_DILATE_KSIZE = 3  # slight dilation to recover leaf edges

# Confidence threshold: if leaf covers less than this fraction of the
# segmented image, warn caller (field image may be a non-leaf photo).
LEAF_COVERAGE_WARN = 0.10  # 10 %


# ─────────────────────────────────────────────
# Data class returned to the caller
# ─────────────────────────────────────────────

@dataclass
class SegmentationResult:
    """
    Returned by segment_leaf().

    Attributes
    ----------
    segmented_image : np.ndarray
        RGB image (same size as input) with background replaced by BG_FILL_RGB.
        Pass this directly into val_transform() instead of the raw image.
    mask : np.ndarray
        Binary uint8 mask (255 = leaf, 0 = background). Same HxW as input.
    leaf_coverage : float
        Fraction of image pixels identified as leaf (0.0 – 1.0).
    method_used : str
        One of "hsv", "grabcut", "grabcut_fallback_full".
        Useful for logging/debugging in production.
    bbox : Tuple[int,int,int,int]
        (x, y, w, h) of the tightest bounding box around the leaf mask.
        Can be used downstream for Disease Region Localisation (Part 2).
    warning : Optional[str]
        Non-None when segmentation quality may be poor.
    """
    segmented_image: np.ndarray
    mask:            np.ndarray
    leaf_coverage:   float
    method_used:     str
    bbox:            Tuple[int, int, int, int]
    warning:         Optional[str] = None


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _morphological_clean(mask: np.ndarray) -> np.ndarray:
    """Remove noise (open) then fill interior holes (close), then dilate."""
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_KSIZE,  MORPH_OPEN_KSIZE))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KSIZE, MORPH_CLOSE_KSIZE))
    k_dil   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_DILATE_KSIZE, MORPH_DILATE_KSIZE))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.dilate(mask, k_dil, iterations=1)
    return mask


def _largest_contour_mask(mask: np.ndarray, min_area: float) -> np.ndarray:
    """
    Keep only the single largest connected contour above `min_area` pixels.
    Returns the cleaned mask (all-255 or all-0 if nothing qualifies).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return np.zeros_like(mask)

    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return clean


def _bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (x, y, w, h) of the non-zero region, or full image if empty."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        h, w = mask.shape
        return (0, 0, w, h)
    x, y, w, h = cv2.boundingRect(coords)
    return (x, y, w, h)


def _apply_mask(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace background pixels with BG_FILL_RGB."""
    result = img_rgb.copy()
    bg = np.full_like(img_rgb, BG_FILL_RGB, dtype=np.uint8)
    mask3 = (mask[:, :, np.newaxis] == 255)        # broadcast to 3 channels
    result = np.where(mask3, result, bg).astype(np.uint8)
    return result


# ─────────────────────────────────────────────
# Stage A — HSV green masking
# ─────────────────────────────────────────────

def _hsv_leaf_mask(img_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Convert to HSV and threshold for green-range pixels.

    Returns
    -------
    mask : np.ndarray  (uint8, 0 or 255)
    coverage : float   (fraction of image that is green)
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, HSV_LOWER_GREEN, HSV_UPPER_GREEN)
    mask = _morphological_clean(mask)

    total    = mask.shape[0] * mask.shape[1]
    coverage = float(np.count_nonzero(mask)) / total
    return mask, coverage


# ─────────────────────────────────────────────
# Stage B — GrabCut refinement
# ─────────────────────────────────────────────

def _grabcut_leaf_mask(img_rgb: np.ndarray,
                       hint_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Run GrabCut to separate leaf from background.

    `hint_mask` (optional): a rough binary mask (from HSV) used to seed
    the GrabCut probable-foreground region. If None, uses a centred rect.

    Returns a uint8 binary mask (255 = leaf, 0 = background).
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]

    # Initialise GrabCut state arrays
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    if hint_mask is not None and np.count_nonzero(hint_mask) > 0:
        # ── Mask-init mode ──
        # Map our binary hint into GrabCut's 4-label system:
        #   GC_BGD=0, GC_FGD=1, GC_PR_BGD=2, GC_PR_FGD=3
        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)   # default: probably background
        gc_mask[hint_mask == 255] = cv2.GC_PR_FGD                   # HSV-green: probably foreground

        # Hard-lock a thin border as definite background (helps convergence)
        border = max(3, h // 30)
        gc_mask[:border, :]  = cv2.GC_BGD
        gc_mask[-border:, :] = cv2.GC_BGD
        gc_mask[:, :border]  = cv2.GC_BGD
        gc_mask[:, -border:] = cv2.GC_BGD

        cv2.grabCut(img_bgr, gc_mask, None,
                    bgd_model, fgd_model,
                    GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)
    else:
        # ── Rect-init mode (no hint available) ──
        # Centre rect — leaves 10% margin on each side
        margin_x = w // 10
        margin_y = h // 10
        rect = (margin_x, margin_y,
                w - 2 * margin_x,
                h - 2 * margin_y)

        gc_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.grabCut(img_bgr, gc_mask, rect,
                    bgd_model, fgd_model,
                    GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)

    # Pixels labelled as probable or definite foreground → leaf
    leaf_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        np.uint8(255), np.uint8(0)
    )
    leaf_mask = _morphological_clean(leaf_mask)
    return leaf_mask


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def segment_leaf(img_rgb: np.ndarray) -> SegmentationResult:
    """
    Main entry point. Accepts an RGB image (np.ndarray, uint8, any size).

    Pipeline
    --------
    1.  HSV green-range masking + morphological cleaning.
    2a. If HSV mask covers >= HSV_MIN_COVERAGE → keep it directly.
    2b. If HSV mask is too sparse (dry/brown/unusual leaf) →
        run GrabCut seeded with the HSV hint.
    3.  Select largest contour from the chosen mask.
    4.  Apply mask: replace background with neutral grey.
    5.  Return SegmentationResult with mask, bbox, coverage, method.

    Parameters
    ----------
    img_rgb : np.ndarray
        Input image in RGB colour space (as used in api.py after
        `cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)`).

    Returns
    -------
    SegmentationResult
        See dataclass docstring above.
    """
    h, w    = img_rgb.shape[:2]
    total   = h * w
    min_area = total * MIN_CONTOUR_FRAC
    warning  = None

    # ── Stage A: HSV ──
    hsv_mask, hsv_cov = _hsv_leaf_mask(img_rgb)

    if hsv_cov >= HSV_MIN_COVERAGE:
        # HSV mask looks good — keep largest contour and proceed
        method = "hsv"
        mask   = _largest_contour_mask(hsv_mask, min_area)
    else:
        # Leaf is not predominantly green (dried, yellowed, diseased heavily)
        # → seed GrabCut with whatever HSV found (even if sparse)
        method    = "grabcut"
        hint_mask = hsv_mask if np.count_nonzero(hsv_mask) > 0 else None
        gc_mask   = _grabcut_leaf_mask(img_rgb, hint_mask=hint_mask)
        mask      = _largest_contour_mask(gc_mask, min_area)

    # ── Safety net: if mask is empty after all that → full image pass-through ──
    leaf_coverage = float(np.count_nonzero(mask)) / total
    if leaf_coverage < 0.01:
        # Could not isolate a leaf — use full image (safe degradation)
        mask          = np.full((h, w), 255, dtype=np.uint8)
        leaf_coverage = 1.0
        method        = "grabcut_fallback_full"
        warning       = (
            "Leaf segmentation failed to isolate a clear leaf region. "
            "Full image passed to CNN. Consider retaking the photo."
        )

    # ── Coverage quality warning ──
    if warning is None and leaf_coverage < LEAF_COVERAGE_WARN:
        warning = (
            f"Low leaf coverage detected ({leaf_coverage:.1%}). "
            "Ensure the leaf fills most of the frame for best results."
        )

    # ── Apply mask ──
    segmented = _apply_mask(img_rgb, mask)
    bbox      = _bounding_box(mask)

    return SegmentationResult(
        segmented_image=segmented,
        mask=mask,
        leaf_coverage=leaf_coverage,
        method_used=method,
        bbox=bbox,
        warning=warning,
    )

# ─────────────────────────────────────────────
# Quick standalone test (run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python leaf_segmentation.py <image_path>")
        print("       Saves: <image_path>_segmented.png  and  <image_path>_mask.png")
        sys.exit(0)

    path  = sys.argv[1]
    img_b = cv2.imread(path)
    if img_b is None:
        print(f"ERROR: Cannot read image: {path}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
    result  = segment_leaf(img_rgb)

    # Save outputs
    base = path.rsplit(".", 1)[0]
    cv2.imwrite(f"{base}_segmented.png",
                cv2.cvtColor(result.segmented_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{base}_mask.png", result.mask)

    print(f"Method       : {result.method_used}")
    print(f"Leaf coverage: {result.leaf_coverage:.2%}")
    print(f"Bounding box : x={result.bbox[0]} y={result.bbox[1]} "
          f"w={result.bbox[2]} h={result.bbox[3]}")
    if result.warning:
        print(f"Warning      : {result.warning}")
    print(f"Saved        : {base}_segmented.png, {base}_mask.png")