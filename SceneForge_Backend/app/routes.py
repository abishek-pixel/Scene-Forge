from fastapi import APIRouter, HTTPException
from app.api.schemas import FrameRequest, FrameResponse

router = APIRouter()

# Lazy import to avoid loading ML models at startup
async def get_infer_frame():
    """Lazy load inference function to avoid startup errors."""
    try:
        from app.core.services.inference import infer_frame
        return infer_frame
    except ImportError:
        async def mock_infer(frame):
            return {
                "frame_id": frame.frame_id,
                "error": "Inference not available - ML packages not installed"
            }
        return mock_infer

@router.post("/api/v1/processing/process", response_model=FrameResponse)
async def process_frame(frame: FrameRequest):
    try:
        infer_frame = await get_infer_frame()
        result = await infer_frame(frame)
        return FrameResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))