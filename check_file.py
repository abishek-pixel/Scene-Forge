import os

file_path = r'SceneForge_Backend\uploads\20251210_114510\sofa.jpg'
if os.path.exists(file_path):
    size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        header = f.read(20)
    print(f'File: {file_path}')
    print(f'Size: {size} bytes')
    print(f'Header (hex): {header.hex()}')
    print(f'Header (ASCII): {header}')
    print(f'Is JPEG: {header[:2].hex() == "ffd8"}')
    print(f'Is PNG: {header[:4].hex() == "89504e47"}')
else:
    print(f'File not found at: {file_path}')
