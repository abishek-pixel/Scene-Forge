import asyncio
from pathlib import Path
from app.core.services.processing_service import ProcessingService

async def update_callback(job_id, pct, msg):
    print(f"[{job_id}] {pct}% - {msg}")

async def main():
    svc = ProcessingService()
    input_path = Path('uploads/20260106_204949/chair.jpg')
    output_path = Path('outputs/chair3')
    job_id = 'reprocess_chair3'
    print('Starting reprocess...')
    result = await svc.process_scene(str(input_path), str(output_path), 'recreate 3d chair', job_id, update_callback)
    print('Reprocess result:', result)

if __name__ == '__main__':
    asyncio.run(main())