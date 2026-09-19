@echo off
title Soloquant Trading System - Launcher
color 0B

rem Switch to the directory where this bat file is located
cd /d "%~dp0"

rem Add project root to Python path so 'src' module can be found
set "PYTHONPATH=%~dp0;%PYTHONPATH%"

rem Use Python from python_env directory
set "PYTHON=.\python_env\python.exe"

rem Check if Python environment exists
if not exist "%PYTHON%" (
    echo.
    echo  [ERROR] Python environment not found: %PYTHON%
    echo  Please run install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

:menu
cls
echo.
echo  ===============================================================
echo.
echo        Soloquant Trading System - Launch Menu
echo.
echo  ===============================================================
echo.
echo  Please select an option:
echo.
echo    [1] Start Dashboard
echo    [2] Start Scheduler
echo    [3] Chat Mode
echo    [4] Auto Trading Loop
echo    [5] System Status
echo    [6] Health Check
echo    [7] Emergency Stop
echo    [8] Realtime Market Monitor
echo    [9] Initialize Local Data
echo.
echo    [0] Exit
echo.
echo  ===============================================================
echo.
set /p choice=Enter option [0-9]: 

if "%choice%"=="1" goto dashboard
if "%choice%"=="2" goto scheduler
if "%choice%"=="3" goto chat
if "%choice%"=="4" goto auto
if "%choice%"=="5" goto status
if "%choice%"=="6" goto health
if "%choice%"=="7" goto emergency
if "%choice%"=="8" goto realtime
if "%choice%"=="9" goto init
if "%choice%"=="0" goto exit

echo.
echo  Invalid option, please try again!
pause
goto menu

:dashboard
echo.
echo  Starting Streamlit Dashboard...
echo  Browser will open automatically. Press Ctrl+C to stop.
echo.
%PYTHON% main.py dashboard
pause
goto menu

:scheduler
echo.
echo  Starting Scheduler...
echo  Running in background. Press Ctrl+C to stop.
echo.
%PYTHON% main.py scheduler
pause
goto menu

:chat
echo.
echo  Entering Chat Mode...
echo  Type 'quit' to exit.
echo.
%PYTHON% main.py chat
pause
goto menu

:auto
echo.
echo  Starting Auto Trading Loop...
echo  WARNING: This will execute trades automatically!
echo  Press Ctrl+C to stop.
echo.
%PYTHON% main.py auto
pause
goto menu

:status
echo.
echo  System Status:
echo.
%PYTHON% main.py status
echo.
pause
goto menu

:health
echo.
echo  Running Health Check...
echo.
%PYTHON% main.py health
echo.
pause
goto menu

:emergency
echo.
echo  ********** EMERGENCY STOP **********
echo.
echo  This will liquidate ALL positions!
echo.
set /p confirm=Are you sure? Type YES to confirm: 
if /i not "%confirm%"=="YES" (
    echo  Cancelled.
    pause
    goto menu
)
echo.
%PYTHON% main.py emergency
echo.
pause
goto menu

:realtime
echo.
echo  Starting Realtime Monitor...
echo.
set /p symbols=Enter stock symbols (comma-separated, e.g. AAPL,MSFT,NVDA): 
if "%symbols%"=="" (
    echo  No symbols specified, using default stock pool.
    %PYTHON% main.py realtime
) else (
    %PYTHON% main.py realtime %symbols%
)
pause
goto menu

:init
echo.
echo  Initializing local data...
echo  First run may take a while to download data.
echo.
%PYTHON% main.py init
echo.
pause
goto menu

:exit
echo.
echo  Goodbye!
echo.
timeout /t 1 >nul
exit /b 0
