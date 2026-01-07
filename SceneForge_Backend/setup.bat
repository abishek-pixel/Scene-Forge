@echo off
echo Setting up SceneForge Backend environment...

:: Create and activate virtual environment
python -m venv venv
call venv\Scripts\activate

:: Install main requirements
echo Installing main requirements...
pip install -r requirements.txt

:: Clone and install GET3D
echo Setting up GET3D...
git clone https://github.com/nv-tlabs/GET3D.git
cd GET3D
pip install -r requirements.txt
python setup.py develop
cd ..

:: Download SAM checkpoint
echo Downloading SAM checkpoint...
mkdir -p checkpoints
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o checkpoints/sam_vit_h_4b8939.pth

echo Setup completed successfully!
pause