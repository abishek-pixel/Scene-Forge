from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, scenes
from app.core.logger import setup_logging
from app.core.config import settings

setup_logging()

app = FastAPI(title="SceneForge Backend API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:5173",  # Vite dev
        "https://scene-forge-app-abhishek-kamthes-projects.vercel.app",  # Vercel production
        "https://*.vercel.app",  # All Vercel deployments
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],  # Explicitly list methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
    max_age=3600,  # Cache preflight for 1 hour
)

# Add this before the router to ensure the root endpoint works
@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI backend!"}

# Add request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"\nIncoming request: {request.method} {request.url.path}")
    print(f"Headers: {dict(request.headers)}")
    response = await call_next(request)
    print(f"Response status: {response.status_code}")
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Include API routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(scenes.router, prefix="/scenes", tags=["scenes"])