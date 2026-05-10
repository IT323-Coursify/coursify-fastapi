from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from core.database import Base

class StudentProfile(Base):
    __tablename__ = "student_profiles"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), unique=True)
    grade_level = Column(String, default="Grade 12")
    strand      = Column(String, default="STEM")
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="profile")