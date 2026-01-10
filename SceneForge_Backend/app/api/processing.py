from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import asyncio
import os
from pathlib import Path
import aiofiles
from ..core.services.processing_service import ProcessingService
import urllib.parse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
processing_service = ProcessingService()

class ProcessingStatus(BaseModel):
    id: str
    name: str  # Scene name or file name
    status: str
    progress: int
    message: Optional[str] = None
    stage: Optional[str] = None
    eta: Optional[str] = None
    details: Optional[List[dict]] = None
    sceneId: str
    createdAt: datetime
    updatedAt: datetime

# Dummy storage - with initial test job
PROCESSING_JOBS = {
    "1": ProcessingStatus(
        id="1",
        name="Test Scene",
        status="processing",
        progress=30,
        message="Processing in progress...",
        stage="File processing",
        eta="5 minutes remaining",
        details=[
            {"step": "Initialization", "completed": True},
            {"step": "File processing", "completed": False},
            {"step": "Scene generation", "completed": False},
            {"step": "Quality checks", "completed": False},
            {"step": "Final optimization", "completed": False}
        ],
        sceneId="test-scene",
        createdAt=datetime.now(),
        updatedAt=datetime.now()
    )
}

async def save_upload_file(upload_file: UploadFile, destination: Path):
    try:
        async with aiofiles.open(destination, 'wb') as out_file:
            while content := await upload_file.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk
    finally:
        await upload_file.close()

@router.post("/files")
async def upload_files(
    files: List[UploadFile] = File(...),
    scene_name: str = Form(...),
    prompt: str = Form(None),
    quality: str = Form("high"),
):
    """
    Handle file uploads for processing.
    Files are uploaded as form-data with the following fields:
    - files: The file(s) to process
    - scene_name: Name of the scene
    - prompt: Optional description/prompt
    - quality: Processing quality (default: high)
    """
    print(f"\n=== Processing Upload Request ===")
    print(f"Files received: {[f.filename for f in files]}")
    print(f"Scene name: {scene_name}")
    print(f"Quality: {quality}")
    print(f"Prompt: {prompt}")
    print("================================\n")
    
    # Create directory for uploads if it doesn't exist
    upload_dir = Path("uploads") / datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to absolute path
    upload_dir = upload_dir.absolute()
    
    saved_files = []
    for file in files:
        file_path = upload_dir / file.filename
        await save_upload_file(file, file_path)
        # Store absolute path
        saved_files.append(str(file_path.absolute()))
    
    # Start processing job
    job_id = str(len(PROCESSING_JOBS) + 1)
    status = ProcessingStatus(
        id=job_id,
        name=scene_name,
        status="processing",
        progress=0,
        message="Starting processing...",
        stage="Initializing",
        eta="Calculating...",
        details=[
            {"step": "Initialization", "completed": True},
            {"step": "File processing", "completed": False},
            {"step": "Scene generation", "completed": False},
            {"step": "Quality checks", "completed": False},
            {"step": "Final optimization", "completed": False}
        ],
        sceneId=scene_name,
        createdAt=datetime.now(),
        updatedAt=datetime.now()
    )
    
    PROCESSING_JOBS[job_id] = status
    
    # Start processing in background
    asyncio.create_task(process_files(job_id, saved_files, prompt, scene_name))
    
    return status

async def process_files(job_id: str, file_paths: List[str], prompt: str, scene_name: str):
    try:
        output_dir = Path("outputs") / scene_name
        
        async def update_status(job_id: str, progress: int, message: str):
            if job_id in PROCESSING_JOBS:
                job = PROCESSING_JOBS[job_id]
                job.progress = progress
                job.message = message
                job.updatedAt = datetime.now()
        
        # Process the files
        result = await processing_service.process_scene(
            file_paths[0],  # Currently handling single file
            str(output_dir),
            prompt,
            job_id,
            update_status
        )
        
        # Update job status with results
        if job_id in PROCESSING_JOBS:
            job = PROCESSING_JOBS[job_id]
            job.status = "completed"
            job.progress = 100
            job.message = "Processing completed successfully"
            job.updatedAt = datetime.now()
            
    except Exception as e:
        if job_id in PROCESSING_JOBS:
            job = PROCESSING_JOBS[job_id]
            job.status = "failed"
            job.message = str(e)
            job.updatedAt = datetime.now()
        logger.error(f"Processing task {job_id} failed: {str(e)}", exc_info=True)

@router.get("/tasks")
async def list_processing_tasks():
    """Return a list of all processing jobs."""
    try:
        tasks = [job.dict() for job in PROCESSING_JOBS.values()]
        print(f"Returning {len(tasks)} tasks")  # Debug log
        return {"tasks": tasks}
    except Exception as e:
        print(f"Error in list_processing_tasks: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

@router.get("/scenes")
async def list_completed_scenes():
    """Return a list of all completed scenes."""
    try:
        completed_scenes = []
        for job in PROCESSING_JOBS.values():
            if job.status == "completed":
                # Get file information from output directory
                output_dir = Path("outputs") / job.name
                files = []
                file_size = "0 MB"
                
                if output_dir.exists():
                    # List all files in output directory
                    try:
                        for file in output_dir.glob('*'):
                            if file.is_file():
                                # Create a download URL instead of using absolute path
                                download_url = f"/processing/download/{job.id}/{urllib.parse.quote(file.name)}"
                                files.append({
                                    "name": file.name,
                                    "path": download_url,  # This is now a URL, not a file path
                                    "size": file.stat().st_size
                                })
                        
                        # Calculate total size
                        total_size = sum(f["size"] for f in files)
                        if total_size > 0:
                            file_size = f"{total_size / 1024 / 1024:.1f} MB"
                    except Exception as e:
                        print(f"Error reading output files: {e}")
                
                completed_scenes.append({
                    "id": job.id,
                    "name": job.name,
                    "date": job.createdAt.isoformat(),
                    "size": file_size,
                    "format": "GLB",
                    "quality": "High",
                    "processingTime": "25 minutes",
                    "fileCount": len(files),
                    "files": files,
                    "status": job.status
                })
        
        print(f"Returning {len(completed_scenes)} completed scenes")
        return {"scenes": completed_scenes}
    except Exception as e:
        print(f"Error in list_completed_scenes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list scenes: {str(e)}")

@router.post("/{job_id}/cancel")
async def cancel_processing(job_id: str):
    if job_id not in PROCESSING_JOBS:
        raise HTTPException(status_code=404, detail="Processing job not found")
    
    job = PROCESSING_JOBS[job_id]
    job.status = "cancelled"
    job.message = "Processing cancelled by user"
    job.updatedAt = datetime.now()
    
    return job

@router.get("/job/{job_id}")
async def get_processing_status(job_id: str):
    """Get the current status of a processing job"""
    if job_id not in PROCESSING_JOBS:
        raise HTTPException(status_code=404, detail="Processing job not found")
    
    job = PROCESSING_JOBS[job_id]
    
    # Simulate progress with more realistic updates
    if job.status == "processing":
        job.progress += 5  # Smaller increments
        
        # Update stage based on progress
        if job.progress < 20:
            job.stage = "File processing"
            job.details[1]["completed"] = False
        elif job.progress < 50:
            job.stage = "Scene generation"
            job.details[1]["completed"] = True
            job.details[2]["completed"] = False
        elif job.progress < 80:
            job.stage = "Quality checks"
            job.details[2]["completed"] = True
            job.details[3]["completed"] = False
        else:
            job.stage = "Final optimization"
            job.details[3]["completed"] = True
            job.details[4]["completed"] = job.progress >= 100
        
        # Update ETA based on progress
        remaining = 100 - job.progress
        if remaining > 0:
            job.eta = f"{(remaining // 5) + 1} minutes remaining"
        
        if job.progress >= 100:
            job.status = "completed"
            job.message = "Processing completed successfully"
            job.stage = "Completed"
            job.eta = "Done"
            job.details[4]["completed"] = True
            
        job.updatedAt = datetime.now()
    
    return job


@router.get("/download/{scene_id}/{file_name}")
async def download_scene_file(scene_id: str, file_name: str):
    """
    Download a scene file by scene ID and filename.
    The file_name should be URL-encoded to handle special characters.
    """
    try:
        # Decode the filename in case it has URL-encoded characters
        decoded_filename = urllib.parse.unquote(file_name)
        
        # Get the job to find the scene name
        if scene_id not in PROCESSING_JOBS:
            raise HTTPException(status_code=404, detail="Scene not found")
        
        job = PROCESSING_JOBS[scene_id]
        scene_name = job.name
        
        # Construct the file path
        file_path = Path("outputs") / scene_name / decoded_filename
        
        # Security check: Make sure the resolved path is within the outputs directory
        try:
            file_path.resolve().relative_to(Path("outputs").resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if file exists
        if not file_path.exists() or not file_path.is_file():
            print(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {decoded_filename}")
        
        print(f"Downloading file: {file_path}")
        
        # Determine media type based on file extension
        media_type = "application/octet-stream"
        if decoded_filename.endswith('.glb'):
            media_type = "model/gltf-binary"
        elif decoded_filename.endswith('.obj'):
            media_type = "model/obj"
        elif decoded_filename.endswith('.png'):
            media_type = "image/png"
        elif decoded_filename.endswith('.json'):
            media_type = "application/json"
        
        # Return the file with appropriate headers
        return FileResponse(
            path=file_path,
            filename=decoded_filename,
            media_type=media_type
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

