@echo off
setlocal
echo ===================================================
echo 🐍 Miniconda ^& Environment Setup Engine
echo ===================================================

:: Check if conda is already in the system
where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Conda is already installed on this machine!
    goto create_env
)

echo.
echo [1/3] 📥 Downloading Miniconda (this may take a minute)...
:: Using built-in Windows curl to grab the latest 64-bit installer
curl -o miniconda_installer.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

echo.
echo [2/3] ⚙️ Installing Miniconda silently...
:: Installs silently without requiring admin rights (JustMe)
start /wait "" miniconda_installer.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%USERPROFILE%\Miniconda3

echo.
echo 🧹 Cleaning up installer...
del miniconda_installer.exe

:create_env
echo.
echo [3/3] 🛠️ Creating the "hackathon" Python 3.10 environment...
:: We must call the activate script to use conda in this batch session
if exist "%USERPROFILE%\Miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\Miniconda3\Scripts\activate.bat"
) else (
    echo [WARNING] Could not find Conda activation script automatically. 
    echo You may need to open "Anaconda Prompt" from your Start Menu manually.
)

:: Create the environment automatically (the -y flag skips the yes/no prompt)
conda create -n hackathon python=3.10 -y

echo.
echo ===================================================
echo 🎉 ENVIRONMENT SETUP COMPLETE!
echo ===================================================
echo ➡️ YOUR NEXT STEPS:
echo 1. Open "Anaconda Prompt" from your Windows Start Menu.
echo 2. Type:  conda activate hackathon
echo 3. Run:   setup.bat  (To install FastAPI and PyTorch)
echo ===================================================
echo.
pause