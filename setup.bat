@echo off
REM Setup script for MSA AI Pipeline (Windows)
REM Creates conda environments from YAML files

set SCRIPT_DIR=%~dp0
set ENVS_DIR=%SCRIPT_DIR%envs

echo ==========================================
echo MSA AI Pipeline Setup
echo ==========================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: conda not found
    echo Install miniforge3: https://github.com/conda-forge/miniforge
    exit /b 1
)

REM Create seqTools environment
echo [1/2] Creating seqTools environment...
conda env list | findstr /B "seqTools " >nul 2>nul
if %errorlevel% equ 0 (
    echo   seqTools environment already exists
) else (
    conda env create -f "%ENVS_DIR%\seqTools.yml"
    echo   seqTools environment created
)

REM Create AI environment
echo [2/2] Creating AI environment...
conda env list | findstr /B "AI " >nul 2>nul
if %errorlevel% equ 0 (
    echo   AI environment already exists
) else (
    conda env create -f "%ENVS_DIR%\ai.yml"
    echo   AI environment created
)

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo Verify installation:
echo   conda run -n seqTools mafft --version
echo   conda run -n seqTools muscle --version
echo   conda run -n seqTools trimal --version
echo   conda run -n AI python -c "import torch; print(torch.__version__)"
echo.
