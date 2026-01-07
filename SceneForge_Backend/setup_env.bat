@echo off
echo Setting up SceneForge Backend environment...

:: Create and activate virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install base requirements
echo Installing base requirements...
pip install -r requirements.txt

:: Create necessary directories
echo Creating directories...
mkdir uploads 2>nul
mkdir outputs 2>nul
mkdir checkpoints 2>nul

:: Download model checkpoints
echo Downloading model checkpoints...
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam-vit-huge', 'sam_vit_h_4b8939.pth', local_dir='checkpoints')"

echo Setup completed!

:: Print dependency versions for verification
echo Verifying installations...
pip list

echo.
echo Setup complete! You can now start the server with:
echo uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause