import os
import torch
from pathlib import Path

def download_sam_model():
    """Downloads the SAM model checkpoint if it doesn't exist"""
    model_path = Path(__file__).parent / "sam_vit_h_4b8939.pth"
    
    if not model_path.exists():
        print("Downloading SAM model checkpoint...")
        # Using torch hub to download the model
        torch.hub.download_url_to_file(
            'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            str(model_path)
        )
        print("SAM model checkpoint downloaded successfully")
    else:
        print("SAM model checkpoint already exists")

if __name__ == "__main__":
    download_sam_model()