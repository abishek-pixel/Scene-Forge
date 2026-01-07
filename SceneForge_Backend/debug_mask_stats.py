import cv2
import numpy as np
from PIL import Image
img = Image.open('outputs/chair3/debug_mask.png')
mask = np.array(img)>0
h,w = mask.shape
areas = cv2.connectedComponentsWithStats(mask.astype('uint8'), connectivity=8)
num_labels = areas[0]
stats = areas[2]
labels = areas[1]
# compute largest non-background component area
if num_labels>1:
    comp_areas = stats[1:, cv2.CC_STAT_AREA]
    largest = comp_areas.max()
    largest_idx = comp_areas.argmax()+1
else:
    largest=0
    largest_idx=0
print('Mask size',w,h,'Total pixels',w*h)
print('Nonzero pixels',mask.sum(),'coverage %.2f'% (mask.sum()/ (w*h) *100))
print('Connected components (incl background):',num_labels)
print('Largest component area:',largest)
# bounding box of largest component
x = stats[largest_idx, cv2.CC_STAT_LEFT]
y = stats[largest_idx, cv2.CC_STAT_TOP]
w_box = stats[largest_idx, cv2.CC_STAT_WIDTH]
h_box = stats[largest_idx, cv2.CC_STAT_HEIGHT]
print('Largest component bbox:', (x,y,w_box,h_box))
# proportion of largest to mask
print('Largest fraction of mask %.2f'%(largest / (mask.sum()) * 100))
# save cropped overlay of largest bbox for visual check
from PIL import Image
overlay = Image.open('outputs/chair3/debug_overlay.png')
overlay.crop((x,y,x+w_box,y+h_box)).save('outputs/chair3/debug_overlay_crop.png')
print('Saved debug_overlay_crop.png')