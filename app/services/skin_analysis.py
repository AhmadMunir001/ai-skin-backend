import cv2
import numpy as np
from app.services.face_service import extract_skin_regions


def normalize_lighting(image: np.ndarray) -> np.ndarray:
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


# ── ITA — Individual Typology Angle ─────────────────
# Standard dermatology formula: ITA = arctan((L*-50)/b*) × 180/π
def compute_ita(image: np.ndarray) -> dict:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    b = lab[:, :, 2] - 128.0
    b_safe = np.where(np.abs(b) < 0.001, 0.001, b)
    ita_value = float(np.mean(np.degrees(np.arctan((L - 50.0) / b_safe))))

    if ita_value > 55:   tone = "Very Light"
    elif ita_value > 41: tone = "Light"
    elif ita_value > 28: tone = "Intermediate"
    elif ita_value > 10: tone = "Tan"
    elif ita_value > -30:tone = "Brown"
    else:                tone = "Dark"

    return {"ita_angle": round(ita_value, 2), "skin_tone": tone}


# ── Redness Index — R/G channel ratio (erythema) ────
def compute_redness_index(image: np.ndarray) -> dict:
    img_f = image.astype(np.float32)
    ri = float(np.mean(img_f[:, :, 2]) / (np.mean(img_f[:, :, 1]) + 1.0))
    level = "high" if ri > 1.15 else "medium" if ri > 1.05 else "low"
    return {"redness_index": round(ri, 4), "redness_level": level}


# ── Pore Size — LoG high-frequency variance ─────────
def compute_pore_score(gray: np.ndarray) -> dict:
    log = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F)
    score = float(np.var(log))
    label = "enlarged" if score > 800 else "moderate" if score > 300 else "minimal"
    return {"pore_score": round(score, 2), "pore_size": label}


# ── T-zone vs U-zone mapping ─────────────────────────
def analyze_zones(metrics_by_region: dict) -> dict:
    tzone = [metrics_by_region[k] for k in ["forehead", "nose"] if k in metrics_by_region]
    uzone = [metrics_by_region[k] for k in ["left_cheek", "right_cheek", "chin"] if k in metrics_by_region]

    if not tzone or not uzone:
        return {"skin_zone_type": "unknown", "tzone_oiliness": None, "uzone_dryness": None}

    tzone_oil = float(np.mean([m["oiliness_score"] for m in tzone]))
    uzone_dry = float(np.mean([m["dryness_score"] for m in uzone]))
    uzone_oil = float(np.mean([m["oiliness_score"] for m in uzone]))

    if tzone_oil > 35 and uzone_dry > 1.5:   zone_type = "combination"
    elif tzone_oil > 35 and uzone_oil > 35:   zone_type = "oily"
    elif tzone_oil <= 35 and uzone_dry > 1.5: zone_type = "dry"
    else:                                      zone_type = "normal"

    return {"skin_zone_type": zone_type, "tzone_oiliness": round(tzone_oil, 2), "uzone_dryness": round(uzone_dry, 2)}


# ── Per-region analysis ──────────────────────────────
def analyze_region(region: np.ndarray) -> dict:
    region = cv2.resize(region, (128, 128))
    region = normalize_lighting(region)
    hsv  = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    total = 128 * 128

    # Acne: red inflamed + dark comedones
    m1 = cv2.inRange(hsv, np.array([0,60,50]),   np.array([10,255,200]))
    m2 = cv2.inRange(hsv, np.array([165,60,50]), np.array([180,255,200]))
    md = cv2.inRange(hsv, np.array([0,0,0]),     np.array([180,50,80]))
    acne_ratio = (np.sum(m1|m2) + 0.5*np.sum(md)) / 255 / total

    # Oiliness: specular highlights
    _, bm = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    mean_brightness = float(np.mean(gray))
    oiliness_score  = (np.sum(bm)/(255*total))*100 + (mean_brightness/255)*30

    # Dryness: edge density + texture diff
    edges = cv2.Canny(gray, 40, 120)
    blur  = cv2.GaussianBlur(gray, (5,5), 0)
    dryness_score = (np.sum(edges)/(255*total))*200 + np.mean(np.abs(gray.astype(float)-blur.astype(float)))*0.5

    # Pigmentation: LAB a-channel std
    _, a_ch, _ = cv2.split(lab)

    return {
        "acne_ratio":         round(float(acne_ratio), 4),
        "oiliness_score":     round(float(oiliness_score), 3),
        "dryness_score":      round(float(dryness_score), 3),
        "pigmentation_score": round(float(np.std(a_ch)), 3),
        "gray_std":           round(float(np.std(gray)), 3),
        "mean_brightness":    round(mean_brightness, 2),
    }


# ── Sensitivity ──────────────────────────────────────
def detect_sensitivity(metrics: list, redness_level: str) -> str:
    score = np.mean([m["acne_ratio"] for m in metrics])*100 + np.mean([m["pigmentation_score"] for m in metrics])*0.3
    if redness_level == "high": score += 3
    return "high" if score > 6 else "medium" if score > 3 else "low"


# ── MAIN ─────────────────────────────────────────────
def analyze_skin(face_image_path: str) -> dict:
    image = cv2.imread(face_image_path)
    if image is None:
        raise ValueError("Invalid image")

    image = normalize_lighting(image)
    warnings, confidence_score = [], 1.0

    quality = check_image_quality(image)
    blur_score, brightness_main, contrast = quality["blur"], quality["brightness"], quality["contrast"]

    if blur_score < 60:
        warnings.append("Image is blurry — retake in better focus"); confidence_score -= 0.35
    if brightness_main < 55:
        warnings.append("Too dark — use natural or bright indoor light"); confidence_score -= 0.25
    elif brightness_main > 210:
        warnings.append("Overexposed — avoid direct flash"); confidence_score -= 0.20
    if contrast < 15:
        warnings.append("Low contrast — may affect accuracy"); confidence_score -= 0.10

    # Clinical whole-face metrics
    ita_data     = compute_ita(image)
    redness_data = compute_redness_index(image)
    gray_full    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pore_data    = compute_pore_score(gray_full)

    regions_dict = extract_skin_regions(face_image_path)
    if not regions_dict:
        return _fallback(blur_score, brightness_main, contrast, warnings, ita_data, redness_data, pore_data)

    metrics_by_region, all_metrics = {}, []
    for name, img in regions_dict.items():
        if img is None or img.size == 0: continue
        try:
            m = analyze_region(img)
            metrics_by_region[name] = m
            all_metrics.append(m)
        except Exception:
            continue

    if not all_metrics:
        return _fallback(blur_score, brightness_main, contrast, warnings, ita_data, redness_data, pore_data)

    avg_acne   = np.mean([m["acne_ratio"] for m in all_metrics])
    avg_oil    = np.mean([m["oiliness_score"] for m in all_metrics])
    avg_dry    = np.mean([m["dryness_score"] for m in all_metrics])
    avg_pig    = np.mean([m["pigmentation_score"] for m in all_metrics])

    # T-zone weighted oiliness
    tz = [metrics_by_region[k] for k in ["forehead","nose"] if k in metrics_by_region]
    if tz: avg_oil = avg_oil*0.5 + np.mean([m["oiliness_score"] for m in tz])*0.5

    # Cheek weighted dryness
    ck = [metrics_by_region[k] for k in ["left_cheek","right_cheek"] if k in metrics_by_region]
    if ck: avg_dry = avg_dry*0.5 + np.mean([m["dryness_score"] for m in ck])*0.5

    zone_data = analyze_zones(metrics_by_region)
    confidence_score = round(max(0.0, min(confidence_score, 1.0)), 2)

    return {
        "acne":         classify(avg_acne, 0.02, 0.08),
        "oiliness":     classify(avg_oil,  20.0, 45.0),
        "dryness":      classify(avg_dry,   0.8,  2.5),
        "pigmentation": classify(avg_pig,   8.0, 18.0),
        "sensitivity":  detect_sensitivity(all_metrics, redness_data["redness_level"]),

        # ── Clinical metrics ──
        "skin_tone":      ita_data["skin_tone"],
        "ita_angle":      ita_data["ita_angle"],
        "redness":        redness_data["redness_level"],
        "redness_index":  redness_data["redness_index"],
        "pore_size":      pore_data["pore_size"],
        "skin_zone_type": zone_data["skin_zone_type"],

        "zones": zone_data,
        "scores": {
            "acne_ratio":    round(float(avg_acne), 4),
            "oiliness":      round(float(avg_oil), 2),
            "dryness":       round(float(avg_dry), 2),
            "pigmentation":  round(float(avg_pig), 2),
            "redness_index": redness_data["redness_index"],
            "pore_score":    pore_data["pore_score"],
            "ita_angle":     ita_data["ita_angle"],
        },
        "regions_analyzed": list(metrics_by_region.keys()),
        "image_quality": {"blur_score": round(blur_score,2), "brightness": round(brightness_main,2), "contrast": round(contrast,2)},
        "confidence":      confidence_score,
        "retake_required": confidence_score < 0.5,
        "warnings":        warnings,
    }


def _fallback(blur, brightness, contrast, warnings, ita_data, redness_data, pore_data):
    return {
        "acne":"medium","oiliness":"medium","dryness":"medium","pigmentation":"medium","sensitivity":"medium",
        "skin_tone":      ita_data["skin_tone"],
        "ita_angle":      ita_data["ita_angle"],
        "redness":        redness_data["redness_level"],
        "redness_index":  redness_data["redness_index"],
        "pore_size":      pore_data["pore_size"],
        "skin_zone_type": "unknown",
        "zones": {}, "scores": {},
        "regions_analyzed": [],
        "image_quality": {"blur_score":round(blur,2),"brightness":round(brightness,2),"contrast":round(contrast,2)},
        "confidence": 0.3, "retake_required": True,
        "warnings": warnings + ["Could not extract skin regions — try a clearer front-facing photo"],
    }