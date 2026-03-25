import os
from uuid import uuid4
from fastapi import UploadFile, HTTPException
from app.core.config import UPLOAD_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


def validate_file(file: UploadFile):
    # Check extension
    filename = file.filename.lower()
    if "." not in filename:
        raise HTTPException(status_code=400, detail="Invalid file format")

    ext = filename.split(".")[-1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, PNG allowed")

    return ext


async def save_file(file: UploadFile):
    ext = validate_file(file)

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 5MB)")

    unique_name = f"{uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(file_path, "wb") as f:
        f.write(contents)

    return file_path