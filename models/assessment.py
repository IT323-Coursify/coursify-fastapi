from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from core.database import Base

class AssessmentResult(Base):
    __tablename__ = "assessment_results"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"))
    strand          = Column(String)
    riasec_scores   = Column(JSON, default=dict)
    mbti_type       = Column(String)
    academic_scores = Column(JSON, default=dict)
    top_course      = Column(String)
    match_score     = Column(Float)
    taken_at        = Column(DateTime, default=datetime.utcnow)

    user            = relationship("User", back_populates="assessments")
    recommendations = relationship("CourseRecommendation",
                                   back_populates="assessment",
                                   cascade="all, delete-orphan")

class CourseRecommendation(Base):
    __tablename__ = "course_recommendations"

    id          = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("assessment_results.id"))
    course_name = Column(String)
    match_score = Column(Float)
    reason      = Column(String)
    rank        = Column(Integer)

    assessment  = relationship("AssessmentResult", back_populates="recommendations")