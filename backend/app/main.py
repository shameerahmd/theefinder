from fastapi import FastAPI

from app.api.classification import (
    router as classification_router,
)
from app.api.firms import (
    router as firms_router,
)


app = FastAPI(
    title="TheeFinder API",
    description=(
        "AI-enabled geospatial platform for detection, "
        "classification, and monitoring of industrial "
        "fires and persistent thermal sources."
    ),
    version="0.2.0",
)


# =========================================================
# ROUTERS
# =========================================================

app.include_router(firms_router)
app.include_router(classification_router)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "application": "TheeFinder",
        "status": "running",
        "study_area": "Chennai",
        "version": "0.2.0",
        "capabilities": [
            "NASA FIRMS thermal detection",
            "Industrial-context analysis",
            "ESA WorldCover analysis",
            "Historical persistence analysis",
            "Stage-A AI classification",
            "Stage-B industrial classification",
        ],
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
    }