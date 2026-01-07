from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch
from PIL import Image
import numpy as np

# Load model and processor (do this once, outside your function)
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

async def infer_frame(frame):
    # Assume frame.data is a path to an image or a numpy array
    image = Image.open(frame.data) if isinstance(frame.data, str) else Image.fromarray(frame.data)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    segmentation = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()
    # Save or process segmentation as needed
    return {
        "frame_id": frame.frame_id,
        "segmentation": segmentation.tolist(),
        "segmentation_shape": segmentation.shape,
    }