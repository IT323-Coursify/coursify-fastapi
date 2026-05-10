from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from core.database import get_db
from core.security import get_current_user
from models.user import User
from models.assessment import AssessmentResult, CourseRecommendation
from schemas.assessment import (
    AssessmentSubmitRequest, AssessmentResponse,
    PredictRequest, PredictResponse, CourseRecommendationSchema
)
import numpy as np
import joblib, os

router = APIRouter(prefix="/assessment", tags=["Assessment"])

MODEL_PATH = "ml/coursify_model.pkl"
_model = None

def get_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    return _model

STRANDS  = ["STEM", "ABM", "HUMSS", "TVL", "GAS"]
SUBJECTS = ["Math", "Science", "Computer", "English", "Filipino", "Humanities"]

# ── Submit assessment (stores in DB) ────────────────────
@router.post("/submit", status_code=status.HTTP_201_CREATED)
def submit_assessment(
    payload: AssessmentSubmitRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    result = AssessmentResult(
        user_id        = current_user.id,
        strand         = payload.strand,
        riasec_scores  = payload.riasec_scores,
        mbti_type      = payload.mbti_type,
        academic_scores= payload.academic_scores,
        top_course     = payload.top_course,
        match_score    = payload.match_score,
    )
    db.add(result)
    db.flush()

    for rec in payload.recommendations:
        db.add(CourseRecommendation(
            assessment_id = result.id,
            course_name   = rec.course_name,
            match_score   = rec.match_score,
            reason        = rec.reason,
            rank          = rec.rank,
        ))

    db.commit()
    db.refresh(result)
    return {"message": "Assessment saved.", "id": result.id}

# ── Get all assessments for current user ────────────────
@router.get("/history", response_model=List[AssessmentResponse])
def get_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    results = db.query(AssessmentResult).filter(
        AssessmentResult.user_id == current_user.id
    ).order_by(AssessmentResult.taken_at.desc()).all()
    return results

# ── Get single assessment ────────────────────────────────
@router.get("/{assessment_id}", response_model=AssessmentResponse)
def get_assessment(
    assessment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    result = db.query(AssessmentResult).filter(
        AssessmentResult.id == assessment_id,
        AssessmentResult.user_id == current_user.id
    ).first()
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    return result

# ── Delete assessment ────────────────────────────────────
@router.delete("/{assessment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_assessment(
    assessment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    result = db.query(AssessmentResult).filter(
        AssessmentResult.id == assessment_id,
        AssessmentResult.user_id == current_user.id
    ).first()
    if not result:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    db.delete(result)
    db.commit()

# ── ML Prediction endpoint ───────────────────────────────
@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    model = get_model()

    # Build feature vector — 21 features
    strand_vec = [1 if s == payload.strand else 0 for s in STRANDS]
    riasec     = [payload.riasec_scores.get(t, 0) for t in ["R","I","A","S","E","C"]]
    mbti_vec   = [
        1 if payload.mbti_answers.get("EI") == "E" else 0,
        1 if payload.mbti_answers.get("SN") == "S" else 0,
        1 if payload.mbti_answers.get("TF") == "T" else 0,
        1 if payload.mbti_answers.get("JP") == "J" else 0,
    ]
    academic = [payload.academic_scores.get(s, 0) for s in SUBJECTS]
    features = np.array(strand_vec + riasec + mbti_vec + academic).reshape(1, -1)

    mbti_type = (
        (payload.mbti_answers.get("EI") or "I") +
        (payload.mbti_answers.get("SN") or "N") +
        (payload.mbti_answers.get("TF") or "T") +
        (payload.mbti_answers.get("JP") or "J")
    )

    if model is None:
        # Fallback — return empty prediction if model not trained yet
        return PredictResponse(
            top_course="Run ml/train.py to enable predictions",
            confidence=0.0,
            mbti_type=mbti_type,
            recommendations=[]
        )

    probs   = model.predict_proba(features)[0]
    classes = model.classes_
    top5    = np.argsort(probs)[::-1][:5]

    recommendations = [
        CourseRecommendationSchema(
            rank=i + 1,
            course_name=classes[idx],
            match_score=round(float(probs[idx]) * 100, 1),
            reason=f"Ranked #{i+1} based on your profile."
        )
        for i, idx in enumerate(top5)
    ]

    return PredictResponse(
        top_course=recommendations[0].course_name,
        confidence=recommendations[0].match_score,
        mbti_type=mbti_type,
        recommendations=recommendations,
    )