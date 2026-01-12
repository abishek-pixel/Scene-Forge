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
import os

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="SceneForge Backend API")

# ===== CRITICAL: CORS Configuration (MUST be first) =====
# Middleware executes in REVERSE order of addition, so add CORS first
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "https://scene-forge-7hi4yt9jy-abhishek-kamthes-projects.vercel.app",
]

logger.info(f"CORS Origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
# ===== END CORS Configuration =====

# Global exception handler for 500 errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "Internal server error",
            "message": str(exc)[:200],
            "path": str(request.url.path)
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