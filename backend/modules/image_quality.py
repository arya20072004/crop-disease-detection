"""
image_quality.py
================
Part 3 of CV Enhancement Pipeline — Crop Disease EWS
-----------------------------------------------------
Runs BEFORE any ML inference. Rejects or warns about images that are:

  1. Too blurry       — Laplacian variance below threshold
  2. Too dark         — mean brightness below threshold
  3. Too bright/blown — mean brightness above threshold
  4. Low contrast     — histogram std dev too narrow
  5. Too small        — resolution insufficient for EfficientNet-B4

If an image fails a hard check, the API returns an error immediately
and asks the farmer to retake — no wasted ML inference time.

CV concepts covered
-------------------
  - Laplacian operator for blur detection (variance of second derivative)
  - Histogram analysis for brightness and contrast
  - Percentile-based blown-highlight detection
  - Resolution gating
  - Composite quality score (0–100)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────
# Config — all thresholds in one place
# ─────────────────────────────────────────────

# Blur — Laplacian variance. Below HARD = reject. Below SOFT = warn.
BLUR_HARD_THRESHOLD = 40.0    # clear reject (very blurry)
BLUR_SOFT_THRESHOLD = 80.0    # warn (slightly soft)

# Brightness — mean pixel value (0–255 on grayscale)
BRIGHTNESS_LOW_HARD  = 30.0   # too dark to see leaf detail
BRIGHTNESS_LOW_SOFT  = 60.0   # dim, may affect colour accuracy
BRIGHTNESS_HIGH_HARD = 240.0  # severely overexposed
BRIGHTNESS_HIGH_SOFT = 210.0  # highlight clipping likely

# Contrast — standard deviation of grayscale histogram
CONTRAST_HARD = 18.0          # flat/foggy image, no usable texture
CONTRAST_SOFT = 30.0          # low contrast, may reduce accuracy

# Blown highlights — fraction of pixels above 250 brightness
BLOWN_HARD = 0.25             # >25% pixels blown out → reject
BLOWN_SOFT = 0.10             # >10% pixels blown out → warn

# Resolution — minimum pixels on shortest side
MIN_RESOLUTION_HARD = 100     # genuinely too small for 380×380 resize
MIN_RESOLUTION_SOFT = 200     # will be upscaled, quality may suffer

# Quality score weights (must sum to 1.0)
SCORE_WEIGHTS = {
    "blur"       : 0.40,
    "brightness" : 0.25,
    "contrast"   : 0.20,
    "resolution" : 0.15,
}


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class QualityIssue:
    """One detected quality problem."""
    check    : str    # "blur" | "brightness" | "contrast" | "resolution" | "blown"
    level    : str    # "hard" (reject) | "soft" (warn)
    message  : str    # human-readable, shown to farmer
    value    : float  # measured value for logging/debugging
    threshold: float  # threshold that was crossed


@dataclass
class QualityResult:
    """
    Returned by check_image_quality().

    Attributes
    ----------
    passed : bool
        False if ANY hard check failed → reject image, ask farmer to retake.
    score : float
        Composite quality score 0–100. 100 = perfect.
        Shown to farmer as a confidence indicator.
    issues : List[QualityIssue]
        All detected issues (hard + soft), sorted hard-first.
    retake_reason : Optional[str]
        Set when passed=False. Single clear message for the farmer.
    suggestions : List[str]
        Actionable tips for the farmer to improve the shot.
    metrics : dict
        Raw measured values — useful for logging and debugging.
    """
    passed        : bool
    score         : float
    issues        : List[QualityIssue]
    retake_reason : Optional[str]
    suggestions   : List[str]
    metrics       : dict


# ─────────────────────────────────────────────
# Individual check functions
# ─────────────────────────────────────────────

def _check_blur(gray: np.ndarray) -> Tuple[float, List[QualityIssue]]:
    """
    Laplacian variance — the second spatial derivative amplifies edges.
    A sharp image has large, varied gradients → high variance.
    A blurry image has smoothed gradients → low variance.

    Using variance (not mean) because mean can be high even in blurry
    images if there are a few strong edges.
    """
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    issues  = []

    if lap_var < BLUR_HARD_THRESHOLD:
        issues.append(QualityIssue(
            check     = "blur",
            level     = "hard",
            message   = f"Image is too blurry (sharpness score: {lap_var:.1f}). Please retake.",
            value     = lap_var,
            threshold = BLUR_HARD_THRESHOLD,
        ))
    elif lap_var < BLUR_SOFT_THRESHOLD:
        issues.append(QualityIssue(
            check     = "blur",
            level     = "soft",
            message   = f"Image is slightly soft (sharpness score: {lap_var:.1f}). Results may be less accurate.",
            value     = lap_var,
            threshold = BLUR_SOFT_THRESHOLD,
        ))

    # Normalise to 0–1 for score calculation (cap at 3× soft threshold)
    blur_score = min(lap_var / (BLUR_SOFT_THRESHOLD * 3), 1.0)
    return blur_score, issues


def _check_brightness(gray: np.ndarray) -> Tuple[float, List[QualityIssue]]:
    """
    Mean pixel brightness on the grayscale image.
    Too dark → leaf colours inaccurate, disease spots invisible.
    Too bright → blown highlights, loss of texture detail.
    """
    mean_val = float(gray.mean())
    issues   = []

    if mean_val < BRIGHTNESS_LOW_HARD:
        issues.append(QualityIssue(
            check     = "brightness",
            level     = "hard",
            message   = f"Image is too dark (brightness: {mean_val:.1f}/255). Move to better light.",
            value     = mean_val,
            threshold = BRIGHTNESS_LOW_HARD,
        ))
    elif mean_val < BRIGHTNESS_LOW_SOFT:
        issues.append(QualityIssue(
            check     = "brightness",
            level     = "soft",
            message   = f"Image is dim (brightness: {mean_val:.1f}/255). Better lighting recommended.",
            value     = mean_val,
            threshold = BRIGHTNESS_LOW_SOFT,
        ))
    elif mean_val > BRIGHTNESS_HIGH_HARD:
        issues.append(QualityIssue(
            check     = "brightness",
            level     = "hard",
            message   = f"Image is severely overexposed (brightness: {mean_val:.1f}/255). Avoid direct sunlight on the lens.",
            value     = mean_val,
            threshold = BRIGHTNESS_HIGH_HARD,
        ))
    elif mean_val > BRIGHTNESS_HIGH_SOFT:
        issues.append(QualityIssue(
            check     = "brightness",
            level     = "soft",
            message   = f"Image is bright (brightness: {mean_val:.1f}/255). Some highlight detail may be lost.",
            value     = mean_val,
            threshold = BRIGHTNESS_HIGH_SOFT,
        ))

    # Score: 1.0 at mean=128, falls toward 0 at extremes
    brightness_score = 1.0 - abs(mean_val - 128.0) / 128.0
    brightness_score = max(brightness_score, 0.0)
    return brightness_score, issues


def _check_contrast(gray: np.ndarray) -> Tuple[float, List[QualityIssue]]:
    """
    Standard deviation of pixel intensities.
    Low std dev = flat image (fog, uniform background, lens smudge).
    """
    std_val = float(gray.std())
    issues  = []

    if std_val < CONTRAST_HARD:
        issues.append(QualityIssue(
            check     = "contrast",
            level     = "hard",
            message   = f"Image has extremely low contrast (std: {std_val:.1f}). Check for lens smudge or fog.",
            value     = std_val,
            threshold = CONTRAST_HARD,
        ))
    elif std_val < CONTRAST_SOFT:
        issues.append(QualityIssue(
            check     = "contrast",
            level     = "soft",
            message   = f"Image contrast is low (std: {std_val:.1f}). Disease spots may not be visible.",
            value     = std_val,
            threshold = CONTRAST_SOFT,
        ))

    contrast_score = min(std_val / (CONTRAST_SOFT * 2), 1.0)
    return contrast_score, issues


def _check_blown_highlights(gray: np.ndarray) -> List[QualityIssue]:
    """
    Fraction of pixels above 250 — blown-out highlights lose all texture.
    Separate from mean brightness because you can have a well-exposed
    image overall with a large sky patch that's completely blown.
    """
    total  = gray.size
    blown  = float(np.sum(gray > 250)) / total
    issues = []

    if blown > BLOWN_HARD:
        issues.append(QualityIssue(
            check     = "blown",
            level     = "hard",
            message   = f"{blown*100:.1f}% of image is overexposed. Shade the leaf or change angle.",
            value     = blown,
            threshold = BLOWN_HARD,
        ))
    elif blown > BLOWN_SOFT:
        issues.append(QualityIssue(
            check     = "blown",
            level     = "soft",
            message   = f"{blown*100:.1f}% of pixels are blown out. Try to avoid direct sun reflection.",
            value     = blown,
            threshold = BLOWN_SOFT,
        ))

    return issues


def _check_resolution(img: np.ndarray) -> Tuple[float, List[QualityIssue]]:
    """
    Shortest side in pixels. EfficientNet-B4 needs 380×380 — very small
    images will be aggressively upscaled, introducing artefacts.
    """
    h, w      = img.shape[:2]
    short_side = min(h, w)
    issues     = []

    if short_side < MIN_RESOLUTION_HARD:
        issues.append(QualityIssue(
            check     = "resolution",
            level     = "hard",
            message   = f"Image resolution too low ({w}×{h}px). Move closer to the leaf.",
            value     = float(short_side),
            threshold = float(MIN_RESOLUTION_HARD),
        ))
    elif short_side < MIN_RESOLUTION_SOFT:
        issues.append(QualityIssue(
            check     = "resolution",
            level     = "soft",
            message   = f"Image resolution is low ({w}×{h}px). Closer shots give better results.",
            value     = float(short_side),
            threshold = float(MIN_RESOLUTION_SOFT),
        ))

    res_score = min(short_side / 380.0, 1.0)
    return res_score, issues


# ─────────────────────────────────────────────
# Suggestions builder
# ─────────────────────────────────────────────

_SUGGESTIONS_MAP = {
    ("blur",        "hard")  : "Hold the phone steady and tap the screen to focus on the leaf before shooting.",
    ("blur",        "soft")  : "Try tapping the leaf on screen to lock focus before taking the photo.",
    ("brightness",  "hard")  : "Move to a shaded area or wait for cloud cover. Avoid shooting toward the sun.",
    ("brightness",  "soft")  : "Find a spot with even, diffused light — morning light works well.",
    ("contrast",    "hard")  : "Wipe the camera lens clean and avoid foggy/misty conditions.",
    ("contrast",    "soft")  : "Ensure the leaf fills the frame and the background is not too similar in colour.",
    ("blown",       "hard")  : "Shade the leaf with your hand or body to block direct sunlight on the camera.",
    ("blown",       "soft")  : "Slightly angle the phone to reduce sun glare on the leaf surface.",
    ("resolution",  "hard")  : "Move the phone 15–30 cm closer to the leaf and retake.",
    ("resolution",  "soft")  : "A closer shot gives the model more leaf detail to work with.",
}


def _build_suggestions(issues: List[QualityIssue]) -> List[str]:
    seen = set()
    tips = []
    for issue in issues:
        key = (issue.check, issue.level)
        if key not in seen:
            seen.add(key)
            tip = _SUGGESTIONS_MAP.get(key)
            if tip:
                tips.append(tip)
    return tips


# ─────────────────────────────────────────────
# Composite score
# ─────────────────────────────────────────────

def _compute_score(blur_s: float, bright_s: float,
                   contrast_s: float, res_s: float) -> float:
    raw = (
        blur_s       * SCORE_WEIGHTS["blur"]       +
        bright_s     * SCORE_WEIGHTS["brightness"] +
        contrast_s   * SCORE_WEIGHTS["contrast"]   +
        res_s        * SCORE_WEIGHTS["resolution"]
    )
    return round(raw * 100, 1)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def check_image_quality(img_rgb: np.ndarray) -> QualityResult:
    """
    Main entry point. Call this BEFORE any ML inference.

    Parameters
    ----------
    img_rgb : np.ndarray (H, W, 3), uint8, RGB
        Raw decoded image — NOT yet segmented or resized.
        Pass the result of cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        directly from _run_analyze_sync().

    Returns
    -------
    QualityResult
        If result.passed is False → return 400 error to Flutter immediately.
        If result.passed is True  → proceed with segmentation + ML pipeline.
    """
    gray   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    issues = []

    blur_s,    blur_issues    = _check_blur(gray)
    bright_s,  bright_issues  = _check_brightness(gray)
    contrast_s, contrast_issues = _check_contrast(gray)
    blown_issues              = _check_blown_highlights(gray)
    res_s,     res_issues     = _check_resolution(img_rgb)

    issues = blur_issues + bright_issues + contrast_issues + blown_issues + res_issues

    # Sort: hard failures first
    issues.sort(key=lambda i: 0 if i.level == "hard" else 1)

    hard_failures = [i for i in issues if i.level == "hard"]
    passed        = len(hard_failures) == 0

    retake_reason = None
    if not passed:
        # Pick the most critical failure for the top-level message
        retake_reason = hard_failures[0].message

    score       = _compute_score(blur_s, bright_s, contrast_s, res_s)
    suggestions = _build_suggestions(issues)

    metrics = {
        "laplacian_variance" : round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "mean_brightness"    : round(float(gray.mean()), 2),
        "contrast_std"       : round(float(gray.std()), 2),
        "blown_pct"          : round(float(np.sum(gray > 250)) / gray.size * 100, 2),
        "resolution"         : f"{img_rgb.shape[1]}x{img_rgb.shape[0]}",
        "short_side_px"      : int(min(img_rgb.shape[:2])),
    }

    return QualityResult(
        passed        = passed,
        score         = score,
        issues        = issues,
        retake_reason = retake_reason,
        suggestions   = suggestions,
        metrics       = metrics,
    )

# ─────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python image_quality.py <image_path>")
        sys.exit(0)

    path    = sys.argv[1]
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"ERROR: Cannot read image: {path}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result  = check_image_quality(img_rgb)

    print(f"\n{'='*50}")
    print(f"  Quality Score : {result.score} / 100")
    print(f"  Passed        : {result.passed}")
    print(f"{'='*50}")

    print(f"\nMetrics:")
    for k, v in result.metrics.items():
        print(f"  {k:<22} : {v}")

    if result.issues:
        print(f"\nIssues ({len(result.issues)}):")
        for issue in result.issues:
            tag = "❌ REJECT" if issue.level == "hard" else "⚠️  WARN  "
            print(f"  {tag} [{issue.check}] {issue.message}")
    else:
        print("\n  ✅ No issues detected.")

    if result.suggestions:
        print(f"\nSuggestions for farmer:")
        for tip in result.suggestions:
            print(f"  → {tip}")

    if result.retake_reason:
        print(f"\n🚫 Retake reason: {result.retake_reason}")

    print()