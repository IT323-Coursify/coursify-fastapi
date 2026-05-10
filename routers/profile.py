from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from core.database import get_db
from core.security import get_current_user
from models.user import User
from models.profile import StudentProfile
from schemas.profile import ProfileResponse, ProfileUpdateRequest
from datetime import datetime

router = APIRouter(prefix="/profile", tags=["Profile"])

@router.get("", response_model=ProfileResponse)
def get_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    profile = db.query(StudentProfile).filter(
        StudentProfile.user_id == current_user.id
    ).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found.")

    return ProfileResponse(
        username=current_user.username,
        email=current_user.email,
        grade_level=profile.grade_level,
        strand=profile.strand,
        created_at=current_user.created_at,
    )

@router.patch("")
def update_profile(
    payload: ProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    profile = db.query(StudentProfile).filter(
        StudentProfile.user_id == current_user.id
    ).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found.")

    if payload.grade_level is not None:
        profile.grade_level = payload.grade_level
    if payload.strand is not None:
        profile.strand = payload.strand

    profile.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(profile)

    return {"message": "Profile updated successfully."}