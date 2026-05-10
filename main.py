from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.database import Base, engine
from routers import auth, profile, assessment

# ── Create all tables ────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── App ──────────────────────────────────────────────────
app = FastAPI(
    title="Coursify API",
    description="Personalized college course recommendation system backend",
    version="1.0.0",
)

# ── CORS ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────
app.include_router(auth.router,       prefix="/api/v1")
app.include_router(profile.router,    prefix="/api/v1")
app.include_router(assessment.router, prefix="/api/v1")

# ── Root ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Coursify API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
    }

@app.get("/health")
def health():
    return {"status": "ok"}