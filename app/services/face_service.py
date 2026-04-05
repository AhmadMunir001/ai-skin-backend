import cv2
import mediapipe as mp
import numpy as np
import os
from uuid import uuid4

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh


def detect_and_crop_face(image_path: str, output_dir: str = "uploads") -> str:
    """Detect face, crop with padding, save and return path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image: cannot read file")

    h, w, _ = image.shape

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as detector:
        results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            raise ValueError("No face detected — please use a clear front-facing photo")

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # 15% padding around face
        pad_x = int(bw * 0.15)
        pad_y = int(bh * 0.15)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        face_crop = image[y1:y2, x1:x2]

        os.makedirs(output_dir, exist_ok=True)
        filename = f"face_{uuid4().hex[:8]}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, face_crop)

        return output_path


def extract_skin_regions(face_image_path: str) -> dict:
    """
    Use MediaPipe Face Mesh landmarks to extract precise skin regions:
    forehead, left cheek, right cheek, nose, chin.
    Falls back to proportional crop if mesh fails.
    Returns dict with region name → numpy array.
    """
    image = cv2.imread(face_image_path)
    if image is None:
        raise ValueError("Invalid face image")

    h, w, _ = image.shape

    def safe_crop(img, y1_r, y2_r, x1_r, x2_r):
        """Crop by ratio, clamp to bounds, return None if too small."""
        y1 = max(0, int(y1_r * h))
        y2 = min(h, int(y2_r * h))
        x1 = max(0, int(x1_r * w))
        x2 = min(w, int(x2_r * w))
        crop = img[y1:y2, x1:x2]
        if crop.size < 100:
            return None
        return crop

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                raise RuntimeError("No landmarks")

            lm = results.multi_face_landmarks[0].landmark

            def pt(idx):
                return int(lm[idx].x * w), int(lm[idx].y * h)

            # Key landmarks (MediaPipe 468-point mesh)
            # Forehead: between brow top and hairline (approx)
            forehead_top_y = max(0, pt(10)[1] - int(0.08 * h))
            forehead_bot_y = pt(10)[1]
            forehead_l_x = pt(234)[0]
            forehead_r_x = pt(454)[0]

            # Left cheek: below left eye, above jaw
            lc_top_y = pt(116)[1]
            lc_bot_y = pt(172)[1]
            lc_l_x = pt(234)[0]
            lc_r_x = pt(116)[0]

            # Right cheek
            rc_top_y = pt(345)[1]
            rc_bot_y = pt(397)[1]
            rc_l_x = pt(345)[0]
            rc_r_x = pt(454)[0]

            # Nose bridge/tip area
            nose_top_y = pt(6)[1]
            nose_bot_y = pt(4)[1]
            nose_l_x = pt(131)[0]
            nose_r_x = pt(360)[0]

            # Chin
            chin_top_y = pt(152)[1]
            chin_bot_y = min(h, pt(152)[1] + int(0.06 * h))
            chin_l_x = pt(176)[0]
            chin_r_x = pt(400)[0]

            def crop_box(y1, y2, x1, x2):
                y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
                x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
                c = image[y1:y2, x1:x2]
                return c if c.size >= 100 else None

            regions = {
                "forehead": crop_box(forehead_top_y, forehead_bot_y, forehead_l_x, forehead_r_x),
                "left_cheek": crop_box(lc_top_y, lc_bot_y, lc_l_x, lc_r_x),
                "right_cheek": crop_box(rc_top_y, rc_bot_y, rc_l_x, rc_r_x),
                "nose": crop_box(nose_top_y, nose_bot_y, nose_l_x, nose_r_x),
                "chin": crop_box(chin_top_y, chin_bot_y, chin_l_x, chin_r_x),
            }

            # Remove None regions
            regions = {k: v for k, v in regions.items() if v is not None}

            if len(regions) < 2:
                raise RuntimeError("Too few valid regions from landmarks")

            return regions

    except Exception:
        # Proportional fallback — still better than before
        regions = {}
        r = safe_crop(image, 0.05, 0.25, 0.25, 0.75)
        if r is not None: regions["forehead"] = r
        r = safe_crop(image, 0.40, 0.70, 0.05, 0.38)
        if r is not None: regions["left_cheek"] = r
        r = safe_crop(image, 0.40, 0.70, 0.62, 0.95)
        if r is not None: regions["right_cheek"] = r
        r = safe_crop(image, 0.35, 0.60, 0.38, 0.62)
        if r is not None: regions["nose"] = r
        r = safe_crop(image, 0.75, 0.95, 0.30, 0.70)
        if r is not None: regions["chin"] = r
        return regions