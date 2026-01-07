from app.core.services.segmentation import SemanticSegmenter
from PIL import Image
import numpy as np
import os

print('Initializing segmenter...')
s = SemanticSegmenter()
print('Segmenter initialized. Loading image...')
img_path = os.path.join('outputs', 'chair3', 'preview.png')
img = Image.open(img_path).convert('RGB')
arr = np.array(img)
print('Running segmentation...')
mask, conf = s.segment_image(arr)
mask_uint8 = (mask*255).astype('uint8')
mask_out = os.path.join('outputs', 'chair3', 'debug_mask.png')
overlay_out = os.path.join('outputs', 'chair3', 'debug_overlay.png')
Image.fromarray(mask_uint8).save(mask_out)
overlay = arr.copy()
overlay[mask==0] = [70, 63, 99]
Image.fromarray(overlay).save(overlay_out)
print('Saved', mask_out, overlay_out)