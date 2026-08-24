from fastapi import FastAPI

from app.api.firms import router as firms_router


app = FastAPI(
    title="TheeFinder API",
    description=(
        "AI-enabled geospatial platform for detection, classification, "
        "and monitoring of industrial fires and persistent thermal sources."
    ),
    version="0.1.0",
)

app.include_router(firms_router)


@app.get("/")
def root():
    return {
        "application": "TheeFinder",
        "status": "running",
        "study_area": "Chennai",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }