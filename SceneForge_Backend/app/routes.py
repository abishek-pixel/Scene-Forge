from fastapi import APIRouter, HTTPException
from app.api.schemas import FrameRequest, FrameResponse
from app.core.services.inference import infer_frame

router = APIRouter()

@router.post("/api/v1/processing/process", response_model=FrameResponse)
async def process_frame(frame: FrameRequest):
    try:
        result = await infer_frame(frame)
        return FrameResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))