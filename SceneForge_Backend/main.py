from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import auth, scenes, processing
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

app = FastAPI(title="SceneForge API")

# Configure CORS - MUST be first middleware
# Allow all origins for development/testing
logger.info("CORS: Allowing all origins for development")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False with allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# Global exception handler
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

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(scenes.router, prefix="/scenes", tags=["scenes"])
app.include_router(processing.router, prefix="/processing", tags=["processing"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# DEBUG: Check CORS status
@app.get("/debug/status")
async def debug_status(request: Request):
    return {
        "status": "healthy",
        "backend": "online",
        "cors_enabled": True,
        "request_origin": request.headers.get('origin', 'no-origin'),
        "timestamp": datetime.now().isoformat()
    }

# DEBUG: Test CORS with POST
@app.post("/debug/test-cors")
async def test_cors(request: Request):
    return {
        "message": "CORS POST test successful!",
        "origin": request.headers.get('origin', 'no-origin'),
        "method": "POST"
    }