@echo off
title Soloquant - Local Data Init
cd /d "%~dp0\.."

echo ============================================================
echo    Soloquant System - Local Data Initialization
echo ============================================================
echo.
echo  Downloading 50 US stocks x 3 years historical data
echo  Estimated time: 5-15 minutes (depends on network)
echo.
echo  Stock pool: Tech/Finance/Consumer/Healthcare/Energy
echo  Data source: AKShare (Yahoo Finance)
echo.
pause

echo.
echo [1/3] Downloading data...
echo ============================================================
python scripts/local_setup.py --full

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Download failed, please check network connection
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [OK] Local data initialization complete!
echo ============================================================
echo.
echo  Data path: data\qlib_data\
echo  Factor cache: data\cache\alpha158_factors.parquet
echo.
echo  Next steps:
echo    Run main.py to enter the system
echo    Or run launch.bat to start dashboard
echo.
pause
