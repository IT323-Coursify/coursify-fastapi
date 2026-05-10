from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from core.database import get_db
from core.security import hash_password, verify_password, create_access_token
from models.user import User
from models.profile import StudentProfile
from schemas.auth import RegisterRequest, LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="Username already exists.")
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered.")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.flush()

    profile = StudentProfile(user_id=user.id)
    db.add(profile)
    db.commit()
    db.refresh(user)

    return {"message": "Registration successful.", "username": user.username}

@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password."
        )
    token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=token, username=user.username)