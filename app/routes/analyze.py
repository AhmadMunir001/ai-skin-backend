from fastapi import APIRouter, UploadFile, File
from app.services.file_service import save_file
from app.services.face_service import detect_and_crop_face
from app.services.skin_analysis import analyze_skin
from app.services.routine_generator import generate_routine
from app.services.db_service import save_skin_record
from app.utils.logger import logger

router = APIRouter()


@router.post("/analyze-skin")
async def analyze_skin_api(file: UploadFile = File(...)):
    try:
        # Step 1: Save image
        file_path = await save_file(file)

        # Step 2: Detect face
        face_path = detect_and_crop_face(file_path)

        # Step 3: Analyze skin
        skin_data = analyze_skin(face_path)

        # Step 4: Generate routine
        routine = generate_routine(skin_data)

        # Step 5: Save to DB
        record_id = save_skin_record(skin_data, routine)

        return {
            "status": "success",
            "record_id": record_id,
            "skin_analysis": skin_data,
            "routine": routine
        }

    except Exception as e:
        logger.error(str(e))
        return {
            "status": "error",
            "message": str(e)
        }