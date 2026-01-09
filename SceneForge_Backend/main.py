from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, scenes, processing

app = FastAPI(title="SceneForge API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:5173",  # Vite dev
        "https://scene-forge-app-abhishek-kamthes-projects.vercel.app",  # Old Vercel URL
        "https://scene-forge-bfcuwj0tv-abhishek-kamthes-projects.vercel.app",  # Previous Vercel URL
        "https://scene-forge-pfmmhe923-abhishek-kamthes-projects.vercel.app",  # Current Vercel URL
        "http://localhost:8000",  # Local backend
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(scenes.router, prefix="/scenes", tags=["scenes"])
app.include_router(processing.router, prefix="/processing", tags=["processing"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}