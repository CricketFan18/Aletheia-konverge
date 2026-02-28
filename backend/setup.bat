@echo off
echo ===================================================
echo 🚀 Authenticity Verifier - Backend Setup Engine
echo ===================================================
echo.
echo WARNING: Ensure you have activated your Conda environment first!
echo (e.g., conda activate hackathon)
echo.
pause

echo.
echo [1/3] 📦 Installing core API dependencies...
pip install -r requirements.txt

echo.
echo [2/3] 🧠 Installing PyTorch (GPU CUDA 11.8 version)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo [3/3] 📁 Creating necessary local directories...
mkdir uploads 2>nul
mkdir test_images 2>nul

echo.
echo ===================================================
echo ✅ SETUP COMPLETE!
echo ===================================================
echo 🏃 To start the development server, run:
echo uvicorn main:app --reload
echo.
pause