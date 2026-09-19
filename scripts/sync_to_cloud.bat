@echo off
title Soloquant - Sync to Cloud
cd /d "%~dp0\.."

echo ============================================================
echo    Soloquant System - Sync to Cloud
echo ============================================================
echo.
echo  Select action:
echo.
echo    [1] Sync best model to cloud
echo    [2] Sync all models and strategies
echo    [3] Check cloud connection
echo    [4] List cloud models
echo    [5] Rollback to previous version
echo    [0] Exit
echo.
set /p choice=Enter option: 

if "%choice%"=="1" goto best
if "%choice%"=="2" goto all
if "%choice%"=="3" goto check
if "%choice%"=="4" goto list
if "%choice%"=="5" goto rollback
if "%choice%"=="0" goto end
goto end

:best
echo.
echo [UPLOAD] Syncing best model...
python scripts/sync_to_cloud.py
goto finish

:all
echo.
echo [UPLOAD] Syncing all models and strategies...
python scripts/sync_to_cloud.py --all
goto finish

:check
echo.
echo [CHECK] Testing cloud connection...
python scripts/sync_to_cloud.py --check
goto finish

:list
echo.
echo [LIST] Cloud models:
python scripts/sync_to_cloud.py --list
goto finish

:rollback
echo.
echo [ROLLBACK] Rolling back to previous version...
python scripts/sync_to_cloud.py --rollback
goto finish

:finish
echo.
pause

:end
