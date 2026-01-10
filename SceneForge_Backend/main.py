from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, scenes, processing
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="SceneForge API")

# Configure CORS - MUST be first middleware
cors_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    # All current and previous Vercel URLs
    "https://scene-forge-app-abhishek-kamthes-projects.vercel.app",
    "https://scene-forge-bfcuwj0tv-abhishek-kamthes-projects.vercel.app",
    "https://scene-forge-pfmmhe923-abhishek-kamthes-projects.vercel.app",
    "https://scene-forge-7hi4yt9jy-abhishek-kamthes-projects.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

logger.info(f"CORS configured for origins: {cors_origins}")

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(scenes.router, prefix="/scenes", tags=["scenes"])
app.include_router(processing.router, prefix="/processing", tags=["processing"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.options("/{full_path:path}")
async def preflight_handler(full_path: str):
    """Handle CORS preflight requests"""
    return {"status": "ok"}