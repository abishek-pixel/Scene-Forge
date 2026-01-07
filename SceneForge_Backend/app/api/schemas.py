from pydantic import BaseModel
from typing import Optional, List

class FrameRequest(BaseModel):
    frame_id: int
    data: List[float]  # example: raw data points of frame

class FrameResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None