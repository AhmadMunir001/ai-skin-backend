import cv2
import mediapipe as mp
import os
from uuid import uuid4

mp_face_detection = mp.solutions.face_detection


def detect_and_crop_face(image_path: str, output_dir: str = "uploads"):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Invalid image")

    h, w, _ = image.shape

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            raise ValueError("No face detected")

        # Take first detected face
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        width = min(w - x, width + padding)
        height = min(h - y, height + padding)

        face_crop = image[y:y+height, x:x+width]

        # Save cropped image
        filename = f"face_{uuid4()}.jpg"
        output_path = os.path.join(output_dir, filename)

        cv2.imwrite(output_path, face_crop)

        return output_path


# NEW FUNCTION (Skin Regions Extraction)
def extract_skin_regions(face_image_path: str):
    image = cv2.imread(face_image_path)

    if image is None:
        raise ValueError("Invalid face image")

    h, w, _ = image.shape

    # Forehead
    forehead = image[int(0.1*h):int(0.3*h), int(0.3*w):int(0.7*w)]

    # Left cheek
    left_cheek = image[int(0.4*h):int(0.7*h), int(0.1*w):int(0.4*w)]

    # Right cheek
    right_cheek = image[int(0.4*h):int(0.7*h), int(0.6*w):int(0.9*w)]

    return [forehead, left_cheek, right_cheek]