from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from app.api import auth, scenes, processing
from app.core.logger import setup_logging
from app.core.config import settings
import traceback
import logging

setup_logging()
logger = logging.getLogger(__name__)

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
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
        }
    )

# Global exception handler for 500 errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "Internal server error",
            "message": str(exc)[:200],  # Limit message length
            "path": str(request.url.path)
        },
        headers={
            "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
        }
    )

# Add this before the router to ensure the root endpoint works
@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI backend!"}

# Request size limiting middleware
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Limit upload size to 500MB"""
    # Set max body size for the request
    request.state.max_body_size = 1024 * 1024 * 500  # 500MB
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Request processing error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Request error: {str(e)[:100]}",
                "error": "request_failed"
            },
            headers={
                "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
            }
        )

# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests with error handling"""
    try:
        logger.info(f"Incoming: {request.method} {request.url.path}")
        response = await call_next(request)
        logger.info(f"Response: {response.status_code} {request.url.path}")
        return response
    except Exception as e:
        logger.error(f"Request error on {request.url.path}: {str(e)}", exc_info=True)
        raise

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Include API routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(scenes.router, prefix="/scenes", tags=["scenes"])
app.include_router(processing.router, prefix="/processing", tags=["processing"])