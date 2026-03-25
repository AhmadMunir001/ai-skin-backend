from sqlalchemy import Column, Integer, String, Text
from app.core.database import Base


class SkinRecord(Base):
    __tablename__ = "skin_records"

    id = Column(Integer, primary_key=True, index=True)
    acne = Column(String)
    oiliness = Column(String)
    dryness = Column(String)
    pigmentation = Column(String)
    sensitivity = Column(String)
    routine = Column(Text)