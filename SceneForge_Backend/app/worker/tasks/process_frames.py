from app.worker.worker import celery_app
import asyncio
from app.core.services.inference import infer_frame

@celery_app.task(name="worker.tasks.process_frame.process_frame_task")
def process_frame_task(frame_data):
    # Because infer_frame is async, run in event loop here
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(infer_frame(frame_data))
    return result

