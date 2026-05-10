from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ProfileResponse(BaseModel):
    username: str
    email: str
    grade_level: str
    strand: str
    created_at: datetime

    class Config:
        from_attributes = True

class ProfileUpdateRequest(BaseModel):
    grade_level: Optional[str] = None
    strand: Optional[str] = None