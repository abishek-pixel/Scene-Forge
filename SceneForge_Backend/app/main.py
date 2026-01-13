"""
NOTE: This file is NOT used by Render.
The actual entry point is SceneForge_Backend/main.py
Keep this for reference only.
"""

# ===== CRITICAL: CORS Configuration (MUST be first) =====
# Allow all origins for testing
logger.info("CORS: Allowing all origins for development/testing")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
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

# DEBUG: Endpoint to check CORS and backend status
@app.get("/debug/status")
async def debug_status(request: Request):
    from datetime import datetime
    origin = request.headers.get('origin', 'no-origin')
    return {
        "status": "healthy",
        "backend": "online",
        "request_origin": origin,
        "cors_enabled": True,
        "timestamp": datetime.now().isoformat()
    }

# DEBUG: Simple test endpoint for CORS
@app.post("/debug/test-cors")
async def test_cors(request: Request):
    from datetime import datetime
    return {
        "message": "CORS is working! POST request successful",
        "method": "POST",
        "origin": request.headers.get('origin', 'no-origin'),
        "timestamp": datetime.now().isoformat()
    }

# Include API routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(scenes.router, prefix="/scenes", tags=["scenes"])
app.include_router(processing.router, prefix="/processing", tags=["processing"])