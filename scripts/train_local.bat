@echo off
title Soloquant - Local Training
cd /d "%~dp0\.."

echo ============================================================
echo    Soloquant System - Local Model Training
echo ============================================================
echo.
echo  Select training mode:
echo.
echo    [1] Quick train (8 stocks, ~5-10 min)
echo    [2] Standard train (20 stocks, ~20-40 min)
echo    [3] Full train (50 stocks, ~1-2 hours)
echo    [4] Train + LSTM deep learning
echo    [0] Exit
echo.
set /p choice=Enter option: 

if "%choice%"=="1" goto quick
if "%choice%"=="2" goto standard
if "%choice%"=="3" goto full
if "%choice%"=="4" goto lstm
if "%choice%"=="0" goto end
goto end

:quick
echo.
echo [START] Quick training...
python scripts/train_pipeline.py --quick
goto finish

:standard
echo.
echo [START] Standard training...
python scripts/train_pipeline.py
goto finish

:full
echo.
echo [START] Full training...
python scripts/train_pipeline.py --years 3
goto finish

:lstm
echo.
echo [START] Training with LSTM...
python scripts/train_pipeline.py --lstm
goto finish

:finish
echo.
echo ============================================================
echo [OK] Training complete!
echo ============================================================
echo.
echo  Reports: data\reports\
echo  Models: data\models\
echo  Strategies: data\strategies\
echo.
echo  Next: Run sync_to_cloud.bat to deploy to cloud
echo.
pause

:end
