from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import auth, scenes, processing
from app.core.logger import setup_logging
from app.core.config import settings

setup_logging()

app = FastAPI(title="SceneForge Backend API")

# Configure CORS - MUST be added first, before other middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:5173",  # Vite dev
        "https://scene-forge-app-abhishek-kamthes-projects.vercel.app",  # Old Vercel URL
        "https://scene-forge-bfcuwj0tv-abhishek-kamthes-projects.vercel.app",  # Previous Vercel URL
        "https://scene-forge-pfmmhe923-abhishek-kamthes-projects.vercel.app",  # Earlier Vercel URL
        "https://scene-forge-7hi4yt9jy-abhishek-kamthes-projects.vercel.app",  # CURRENT Vercel URL
        "http://localhost:8000",  # Local backend
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
    max_age=3600,
)

# Add explicit OPTIONS handler for all routes (preflight)
@app.options("/{full_path:path}")
async def preflight_handler(request: Request, full_path: str):
    return JSONResponse(
        content={},
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "3600",
        }
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
app.include_router(processing.router, prefix="/processing", tags=["processing"])