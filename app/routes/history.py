from fastapi import APIRouter
from app.services.db_service import get_all_records
import json

router = APIRouter()


@router.get("/history")
def get_history():
    records = get_all_records()

    result = []

    for r in records:
        result.append({
            "id": r.id,
            "acne": r.acne,
            "oiliness": r.oiliness,
            "dryness": r.dryness,
            "pigmentation": r.pigmentation,
            "sensitivity": r.sensitivity,
            "routine": json.loads(r.routine)
        })

    return {"data": result}