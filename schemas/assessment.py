from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class CourseRecommendationSchema(BaseModel):
    rank: int
    course_name: str
    match_score: float
    reason: str

    class Config:
        from_attributes = True

class AssessmentSubmitRequest(BaseModel):
    strand: str
    riasec_scores: Dict[str, float]
    mbti_type: str
    academic_scores: Dict[str, float]
    top_course: str
    match_score: float
    recommendations: List[CourseRecommendationSchema]

class AssessmentResponse(BaseModel):
    id: int
    strand: str
    riasec_scores: Dict
    mbti_type: str
    academic_scores: Dict
    top_course: str
    match_score: float
    taken_at: datetime
    recommendations: List[CourseRecommendationSchema]

    class Config:
        from_attributes = True

class PredictRequest(BaseModel):
    strand: str
    riasec_scores: Dict[str, float]
    mbti_answers: Dict[str, str]
    academic_scores: Dict[str, float]

class PredictResponse(BaseModel):
    top_course: str
    confidence: float
    mbti_type: str
    recommendations: List[CourseRecommendationSchema]