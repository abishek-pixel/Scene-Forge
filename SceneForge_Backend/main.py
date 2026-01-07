from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, scenes, processing

app = FastAPI(title="SceneForge API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(scenes.router, prefix="/scenes", tags=["scenes"])
app.include_router(processing.router, prefix="/processing", tags=["processing"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}