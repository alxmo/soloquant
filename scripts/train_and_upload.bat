@echo off
REM ═══════════════════════════════════════════════════════════════
REM AI量化系统 - 本地一键训练 + 上传
REM 用法: 双击运行 或 命令行 train_and_upload.bat [--quick] [--lstm]
REM ═══════════════════════════════════════════════════════════════
chcp 65001 >nul
cd /d "%~dp0\.."

echo.
echo ═══════════════════════════════════════════════════════════════
echo   AI量化系统 - 本地训练 + 云端上传 一键脚本
echo ═══════════════════════════════════════════════════════════════
echo.

REM 步骤 1: 训练模型
echo [1/2] 开始本地模型训练...
python scripts\train_pipeline.py %*
if errorlevel 1 (
    echo.
    echo ❌ 训练失败！请检查日志。
    pause
    exit /b 1
)

echo.
echo [2/2] 上传模型到云端...
python scripts\package_model.py
if errorlevel 1 (
    echo.
    echo ⚠️ 上传失败！模型已保存在本地，可稍后手动上传。
    echo    运行: python scripts\package_model.py --check 检查云端连接
    pause
    exit /b 1
)

echo.
echo ═══════════════════════════════════════════════════════════════
echo   ✅ 全流程完成！模型已上传到云端。
echo   云端将在 30 秒内自动热加载新模型。
echo ═══════════════════════════════════════════════════════════════
pause
