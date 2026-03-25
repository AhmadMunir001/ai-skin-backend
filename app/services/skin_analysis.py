import cv2
import numpy as np
from app.services.face_service import extract_skin_regions

def check_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_score

def normalize_lighting(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    # Apply CLAHE (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge back
    lab = cv2.merge((l, a, b))

    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return normalized

def adaptive_threshold(value, base_low, base_high, adjustment_factor):
    low = base_low * adjustment_factor
    high = base_high * adjustment_factor

    if value < low:
        return "low"
    elif value < high:
        return "medium"
    else:
        return "high"

# Image Quality Checker
def check_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)

    return {
        "blur": blur,
        "brightness": brightness
    }


def analyze_skin(face_image_path: str):
    image = cv2.imread(face_image_path)

    if image is None:
        raise ValueError("Invalid image")

    # Normalize lighting
    image = normalize_lighting(image)

    warnings = []
    confidence_score = 1.0

    # Step 1: Image quality
    quality = check_image_quality(image)
    blur_score = quality["blur"]
    brightness_main = quality["brightness"]

    # Blur penalty
    if blur_score < 50:
        warnings.append("Image is blurry, please retake a clearer photo")
        confidence_score -= 0.4

    # Dark image penalty
    if brightness_main < 60:
        warnings.append("Image is too dark")
        confidence_score -= 0.3

    # Bright image penalty
    if brightness_main > 200:
        warnings.append("Image is too bright")
        confidence_score -= 0.3

    # Step 2: Extract regions
    regions = extract_skin_regions(face_image_path)

    acne_scores = []
    brightness_scores = []
    texture_scores = []
    pigmentation_scores = []

    for region in regions:
        if region is None or region.size == 0:
            continue

        region = cv2.resize(region, (128, 128))
        region = normalize_lighting(region)

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Acne
        lower_red = np.array([0, 70, 50])
        upper_red = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        acne_scores.append(np.sum(mask_red) / 255)

        # Oiliness
        brightness_scores.append(np.mean(gray))

        # Texture
        edges = cv2.Canny(gray, 50, 150)
        texture_scores.append(np.sum(edges) / 255)

        # Pigmentation
        pigmentation_scores.append(np.std(gray))

    # Fallback
    if not acne_scores:
        return {
            "acne": "medium",
            "oiliness": "medium",
            "dryness": "medium",
            "pigmentation": "medium",
            "sensitivity": "medium",
            "confidence": 0.3,
            "retake_required": True,
            "warnings": ["Could not analyze skin properly"],
            "image_quality": {
                "blur_score": round(blur_score, 2),
                "brightness": round(brightness_main, 2)
            }
        }

    # Aggregate
    acne_score = np.mean(acne_scores)
    brightness = np.mean(brightness_scores)
    texture_score = np.mean(texture_scores)
    std_dev = np.mean(pigmentation_scores)

    # Adaptive factor
    adjustment_factor = 1 + ((brightness - 120) / 200)
    adjustment_factor = max(0.7, min(adjustment_factor, 1.3))

    # Adaptive classify
    def classify(value, base_low, base_high):
        low = base_low * adjustment_factor
        high = base_high * adjustment_factor

        if value < low:
            return "low"
        elif value < high:
            return "medium"
        else:
            return "high"

    # Final confidence clamp
    confidence_score = max(0.0, min(confidence_score, 1.0))
    retake_required = confidence_score < 0.5

    # Final return
    return {
        "acne": classify(acne_score, 300, 1500),
        "oiliness": classify(brightness, 90, 150),
        "dryness": classify(texture_score, 300, 1500),
        "pigmentation": classify(std_dev, 15, 40),
        "sensitivity": "medium",

        "image_quality": {
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness_main, 2)
        },

        "confidence": round(confidence_score, 2),
        "retake_required": retake_required,
        "warnings": warnings
    }
    confidence_score = max(0.0, min(confidence_score, 1.0))
    retake_required = confidence_score < 0.5
    return result