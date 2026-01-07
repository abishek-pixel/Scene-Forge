from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

router = APIRouter()

class Scene(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    status: str
    userId: str
    fileUrl: str
    createdAt: datetime
    updatedAt: datetime

# Dummy storage
SCENES = {}

@router.post("/upload")
async def upload_scene(file: UploadFile = File(...), description: Optional[str] = None):
    # Simulate file upload
    scene_id = str(len(SCENES) + 1)
    scene = Scene(
        id=scene_id,
        name=file.filename,
        description=description,
        status="pending",
        userId="1",  # In production, get from auth token
        fileUrl=f"http://localhost:8000/uploads/{scene_id}/{file.filename}",
        createdAt=datetime.now(),
        updatedAt=datetime.now()
    )
    
    SCENES[scene_id] = scene
    return scene

@router.get("", response_model=List[Scene])
async def get_scenes():
    return list(SCENES.values())

@router.get("/{scene_id}")
async def get_scene(scene_id: str):
    if scene_id not in SCENES:
        raise HTTPException(status_code=404, detail="Scene not found")
    return SCENES[scene_id]

@router.delete("/{scene_id}")
async def delete_scene(scene_id: str):
    if scene_id not in SCENES:
        raise HTTPException(status_code=404, detail="Scene not found")
    del SCENES[scene_id]
    return {"status": "deleted"}