import cv2
import numpy as np
from app.services.face_service import extract_skin_regions


# ──────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────

def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """CLAHE on L-channel in LAB space for even illumination."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def check_image_quality(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
    }


def classify(value: float, low_thresh: float, high_thresh: float) -> str:
    if value < low_thresh:
        return "low"
    elif value < high_thresh:
        return "medium"
    return "high"


# ──────────────────────────────────────────────
# PER-REGION ANALYSIS
# ──────────────────────────────────────────────

def analyze_region(region: np.ndarray) -> dict:
    """
    Analyze a single skin region.
    Returns raw metric scores (not classified yet).
    """
    region = cv2.resize(region, (128, 128))
    region = normalize_lighting(region)

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # ── ACNE SCORE (multi-channel) ──────────────────
    # Red spots in HSV (inflamed acne)
    red_lo1 = np.array([0, 60, 50]);   red_hi1 = np.array([10, 255, 200])
    red_lo2 = np.array([165, 60, 50]); red_hi2 = np.array([180, 255, 200])
    mask_red = cv2.inRange(hsv, red_lo1, red_hi1) | cv2.inRange(hsv, red_lo2, red_hi2)

    # Dark spots / comedones — low brightness, low saturation
    dark_lo = np.array([0, 0, 0]);   dark_hi = np.array([180, 50, 80])
    mask_dark = cv2.inRange(hsv, dark_lo, dark_hi)

    # Combine: red pixels + dark spots
    acne_pixels = (np.sum(mask_red) + 0.5 * np.sum(mask_dark)) / 255
    total_pixels = 128 * 128
    acne_ratio = acne_pixels / total_pixels  # 0.0 – 1.0

    # ── OILINESS (specular highlight detection) ─────
    # Oily skin reflects more light → bright highlights in V-channel
    _, bright_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    highlight_ratio = np.sum(bright_mask) / (255 * total_pixels)

    # Also use mean brightness as secondary signal
    mean_brightness = float(np.mean(gray))

    oiliness_score = highlight_ratio * 100 + (mean_brightness / 255) * 30

    # ── DRYNESS (texture / edge complexity) ─────────
    edges = cv2.Canny(gray, 40, 120)
    edge_density = np.sum(edges) / (255 * total_pixels)

    # LBP-like texture variation
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    texture_diff = np.mean(np.abs(gray.astype(float) - blur.astype(float)))

    dryness_score = edge_density * 200 + texture_diff * 0.5

    # ── PIGMENTATION (color unevenness) ─────────────
    # std of A channel in LAB: high = more red/green shift = uneven tone
    _, a_ch, _ = cv2.split(lab)
    pigmentation_score = float(np.std(a_ch))

    # Also check overall gray std
    gray_std = float(np.std(gray))

    return {
        "acne_ratio": round(acne_ratio, 4),
        "oiliness_score": round(oiliness_score, 3),
        "dryness_score": round(dryness_score, 3),
        "pigmentation_score": round(pigmentation_score, 3),
        "gray_std": round(gray_std, 3),
        "mean_brightness": round(mean_brightness, 2),
    }


# ──────────────────────────────────────────────
# SENSITIVITY DETECTION
# ──────────────────────────────────────────────

def detect_sensitivity(regions_metrics: list) -> str:
    """
    Sensitivity inferred from:
    - High redness (acne_ratio) + high pigmentation = sensitive
    - Low pigmentation + low acne = not sensitive
    """
    avg_acne = np.mean([m["acne_ratio"] for m in regions_metrics])
    avg_pig = np.mean([m["pigmentation_score"] for m in regions_metrics])

    score = avg_acne * 100 + avg_pig * 0.3

    if score > 6:
        return "high"
    elif score > 3:
        return "medium"
    return "low"


# ──────────────────────────────────────────────
# MAIN ANALYZE FUNCTION
# ──────────────────────────────────────────────

def analyze_skin(face_image_path: str) -> dict:
    image = cv2.imread(face_image_path)
    if image is None:
        raise ValueError("Invalid image")

    image = normalize_lighting(image)
    warnings = []
    confidence_score = 1.0

    # ── Quality checks ──────────────────────────────
    quality = check_image_quality(image)
    blur_score = quality["blur"]
    brightness_main = quality["brightness"]
    contrast = quality["contrast"]

    if blur_score < 60:
        warnings.append("Image is blurry — please retake in better focus")
        confidence_score -= 0.35

    if brightness_main < 55:
        warnings.append("Image is too dark — use natural or bright indoor light")
        confidence_score -= 0.25
    elif brightness_main > 210:
        warnings.append("Image is overexposed — avoid direct flash or harsh light")
        confidence_score -= 0.20

    if contrast < 15:
        warnings.append("Image has low contrast — may affect accuracy")
        confidence_score -= 0.10

    # ── Region extraction ───────────────────────────
    regions_dict = extract_skin_regions(face_image_path)

    if not regions_dict:
        return _fallback_result(blur_score, brightness_main, warnings)

    # ── Per-region analysis ─────────────────────────
    metrics_by_region = {}
    all_metrics = []

    for region_name, region_img in regions_dict.items():
        if region_img is None or region_img.size == 0:
            continue
        try:
            m = analyze_region(region_img)
            metrics_by_region[region_name] = m
            all_metrics.append(m)
        except Exception:
            continue

    if not all_metrics:
        return _fallback_result(blur_score, brightness_main, warnings)

    # ── Aggregate ───────────────────────────────────
    avg_acne_ratio    = np.mean([m["acne_ratio"] for m in all_metrics])
    avg_oiliness      = np.mean([m["oiliness_score"] for m in all_metrics])
    avg_dryness       = np.mean([m["dryness_score"] for m in all_metrics])
    avg_pigmentation  = np.mean([m["pigmentation_score"] for m in all_metrics])

    # T-zone (forehead + nose) is more oily — weight it higher
    tzone_keys = ["forehead", "nose"]
    tzone_metrics = [metrics_by_region[k] for k in tzone_keys if k in metrics_by_region]
    if tzone_metrics:
        tzone_oiliness = np.mean([m["oiliness_score"] for m in tzone_metrics])
        avg_oiliness = avg_oiliness * 0.5 + tzone_oiliness * 0.5

    # Cheeks are drier — weight their dryness
    cheek_keys = ["left_cheek", "right_cheek"]
    cheek_metrics = [metrics_by_region[k] for k in cheek_keys if k in metrics_by_region]
    if cheek_metrics:
        cheek_dryness = np.mean([m["dryness_score"] for m in cheek_metrics])
        avg_dryness = avg_dryness * 0.5 + cheek_dryness * 0.5

    # ── Classification ──────────────────────────────
    acne_label         = classify(avg_acne_ratio,   0.02,  0.08)
    oiliness_label     = classify(avg_oiliness,     20.0,  45.0)
    dryness_label      = classify(avg_dryness,       0.8,   2.5)
    pigmentation_label = classify(avg_pigmentation,  8.0,  18.0)
    sensitivity_label  = detect_sensitivity(all_metrics)

    confidence_score = round(max(0.0, min(confidence_score, 1.0)), 2)
    retake_required  = confidence_score < 0.5

    return {
        "acne":         acne_label,
        "oiliness":     oiliness_label,
        "dryness":      dryness_label,
        "pigmentation": pigmentation_label,
        "sensitivity":  sensitivity_label,

        "scores": {
            "acne_ratio":    round(float(avg_acne_ratio), 4),
            "oiliness":      round(float(avg_oiliness), 2),
            "dryness":       round(float(avg_dryness), 2),
            "pigmentation":  round(float(avg_pigmentation), 2),
        },

        "regions_analyzed": list(metrics_by_region.keys()),

        "image_quality": {
            "blur_score": round(blur_score, 2),
            "brightness":  round(brightness_main, 2),
            "contrast":    round(contrast, 2),
        },

        "confidence":      confidence_score,
        "retake_required": retake_required,
        "warnings":        warnings,
    }


def _fallback_result(blur_score: float, brightness: float, warnings: list) -> dict:
    return {
        "acne": "medium", "oiliness": "medium",
        "dryness": "medium", "pigmentation": "medium", "sensitivity": "medium",
        "scores": {},
        "regions_analyzed": [],
        "image_quality": {
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "contrast": 0,
        },
        "confidence": 0.3,
        "retake_required": True,
        "warnings": warnings + ["Could not extract skin regions — try a clearer, front-facing photo"],
    }