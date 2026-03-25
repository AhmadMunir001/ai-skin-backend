import json
from app.core.database import SessionLocal
from app.models.skin_record import SkinRecord


def save_skin_record(skin_data, routine_data):
    db = SessionLocal()

    record = SkinRecord(
        acne=skin_data["acne"],
        oiliness=skin_data["oiliness"],
        dryness=skin_data["dryness"],
        pigmentation=skin_data["pigmentation"],
        sensitivity=skin_data["sensitivity"],
        routine=json.dumps(routine_data)
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    db.close()

    return record.id


def get_all_records():
    db = SessionLocal()
    records = db.query(SkinRecord).all()
    db.close()

    return records