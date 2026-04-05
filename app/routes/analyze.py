from fastapi import APIRouter, UploadFile, File
from app.services.file_service import save_file
from app.services.face_service import detect_and_crop_face
from app.services.skin_analysis import analyze_skin
from app.services.routine_generator import generate_routine
from app.services.db_service import save_skin_record
from app.utils.logger import logger
import cv2
import tempfile
import os
import numpy as np

router = APIRouter()


@router.post("/analyze-skin")
async def analyze_skin_api(file: UploadFile = File(...)):
    try:
        file_path = await save_file(file)
        face_path = detect_and_crop_face(file_path)
        skin_data = analyze_skin(face_path)
        routine = generate_routine(skin_data)
        record_id = save_skin_record(skin_data, routine)
        return {"status": "success", "record_id": record_id, "skin_analysis": skin_data, "routine": routine}
    except Exception as e:
        logger.error(str(e))
        return {"status": "error", "message": str(e)}


@router.post("/analyze-video")
async def analyze_video_api(file: UploadFile = File(...)):
    tmp_video = None
    frame_paths = []
    try:
        # ── Save video to temp file (support webm/mp4/mov) ──
        suffix = ".webm"
        if file.filename:
            ext = os.path.splitext(file.filename)[-1].lower()
            if ext in [".mp4", ".mov", ".avi", ".webm"]:
                suffix = ext

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_video = tmp.name

        cap = cv2.VideoCapture(tmp_video)
        if not cap.isOpened():
            return {"status": "error", "message": "Could not open video file"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # Sample ~8 frames evenly across the video
        sample_count = min(8, max(3, total_frames // 10))
        sample_indices = set(
            int(i * total_frames / sample_count) for i in range(sample_count)
        )

        frame_results = []
        count = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if count in sample_indices:
                    frame_path = os.path.join(tmpdir, f"frame_{count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)

                    try:
                        face_path = detect_and_crop_face(frame_path, output_dir=tmpdir)
                        skin_data = analyze_skin(face_path)
                        # Only use high-confidence frames
                        if skin_data.get("confidence", 0) >= 0.4:
                            frame_results.append(skin_data)
                    except Exception:
                        pass

                count += 1

        cap.release()

        if not frame_results:
            return {"status": "error", "message": "No valid face frames detected. Please ensure your face is visible and well-lit throughout the video."}

        # ── Aggregate results ──────────────────────────────
        def majority_vote(key):
            values = [r[key] for r in frame_results if key in r]
            return max(set(values), key=values.count) if values else "medium"

        def avg_score(key):
            vals = [r.get("scores", {}).get(key, 0) for r in frame_results]
            return round(float(np.mean(vals)), 4) if vals else 0

        avg_confidence = round(float(np.mean([r.get("confidence", 1) for r in frame_results])), 2)
        all_warnings = list(set(w for r in frame_results for w in r.get("warnings", [])))

        final_skin = {
            "acne":         majority_vote("acne"),
            "oiliness":     majority_vote("oiliness"),
            "dryness":      majority_vote("dryness"),
            "pigmentation": majority_vote("pigmentation"),
            "sensitivity":  majority_vote("sensitivity"),
            "scores": {
                "acne_ratio":   avg_score("acne_ratio"),
                "oiliness":     avg_score("oiliness"),
                "dryness":      avg_score("dryness"),
                "pigmentation": avg_score("pigmentation"),
            },
            "confidence":       avg_confidence,
            "retake_required":  avg_confidence < 0.5,
            "warnings":         all_warnings,
            "frames_analyzed":  len(frame_results),
        }

        routine = generate_routine(final_skin)

        return {
            "status": "success",
            "frames_analyzed": len(frame_results),
            "skin_analysis": final_skin,
            "routine": routine
        }

    except Exception as e:
        logger.error(f"Video analysis error: {str(e)}")
        return {"status": "error", "message": str(e)}

    finally:
        # Cleanup temp video
        if tmp_video and os.path.exists(tmp_video):
            try:
                os.remove(tmp_video)
            except Exception:
                pass