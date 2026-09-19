@echo off
chcp 65001 >nul 2>&1
title Soloquant Trading System - Dependency Installer
color 0B

echo.
echo  ╔═══════════════════════════════════════════════════════════════╗
echo  ║                                                               ║
echo  ║   AI 全自动美股量化交易系统 - 一键依赖安装脚本                  ║
echo  ║   Soloquant Trading System - Dependency Installer              ║
echo  ║                                                               ║
echo  ║   技术栈: Qlib + Alpaca + Finnhub + AKShare + SOLOOMO         ║
echo  ║                                                               ║
echo  ╚═══════════════════════════════════════════════════════════════╝
echo.
echo  Python 环境: 
python --version
echo.

:: ============================================
:: Step 0: Check Python version
:: ============================================
echo  [Step 0/6] 检查 Python 环境...
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo  [ERROR] 需要 Python 3.10+，当前版本过低！
    pause
    exit /b 1
)
echo  ✅ Python 版本检查通过
echo.

:: ============================================
:: Step 1: Upgrade pip
:: ============================================
echo  [Step 1/6] 升级 pip 和打包工具...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo  [WARNING] pip 升级失败，继续安装...
) else (
    echo  ✅ pip 升级完成
)
echo.

:: ============================================
:: Step 2: Install data layer packages
:: ============================================
echo  [Step 2/6] 安装数据层依赖 (AKShare + Finnhub)...
echo  ┌─ AKShare (金融数据采集)
echo  │  Finnhub (新闻/情绪/基本面)
echo  │  pandas, pyarrow, openpyxl (数据处理)
echo  └─

pip install ^
    "akshare>=1.14" ^
    "finnhub-python>=2.4.20" ^
    "pandas>=2.0,<3.0" ^
    "pyarrow>=15.0" ^
    "openpyxl>=3.1" ^
    --quiet

if errorlevel 1 (
    echo  [ERROR] 数据层依赖安装失败！
    echo  尝试逐个安装...
    pip install "akshare>=1.14" --quiet
    pip install "finnhub-python>=2.4.20" --quiet
    pip install "pandas>=2.0,<3.0" --quiet
    pip install "pyarrow>=15.0" --quiet
    pip install "openpyxl>=3.1" --quiet
)
echo  ✅ 数据层依赖安装完成
echo.

:: ============================================
:: Step 3: Install quantitative engine
:: ============================================
echo  [Step 3/6] 安装量化引擎 (Qlib + LightGBM + scikit-learn)...
echo  ┌─ Microsoft Qlib (量化研究框架)
echo  │  LightGBM (梯度提升模型)
echo  │  scikit-learn (机器学习)
echo  │  matplotlib (图表)
echo  └─

pip install ^
    "pyqlib>=0.9.6" ^
    "lightgbm>=4.3" ^
    "scikit-learn>=1.5" ^
    "matplotlib>=3.8" ^
    "plotly>=5.20" ^
    --quiet

if errorlevel 1 (
    echo  [WARNING] Qlib 安装失败，尝试不带依赖单独安装...
    pip install "pyqlib>=0.9.6" --no-deps --quiet
    pip install "lightgbm>=4.3" --quiet
    pip install "scikit-learn>=1.5" --quiet
    pip install "matplotlib>=3.8" --quiet
    pip install "plotly>=5.20" --quiet
    echo  [WARNING] Qlib 以 --no-deps 安装，可能缺少部分依赖
    echo  如遇 import 错误，请手动运行: pip install mlflow redis pymongo cvxpy gym dill fire ruamel.yaml python-redis-lock tqdm
)
echo  ✅ 量化引擎安装完成
echo.

:: ============================================
:: Step 4: Install trading execution layer
:: ============================================
echo  [Step 4/6] 安装交易执行层 (Alpaca API)...
echo  ┌─ Alpaca-py (美股交易API)
echo  │  python-dotenv (环境变量)
echo  │  schedule (定时任务)
echo  │  loguru (日志)
echo  └─

pip install ^
    "alpaca-py>=0.30" ^
    "python-dotenv>=1.0" ^
    "schedule>=1.2" ^
    "loguru>=0.7" ^
    --quiet

if errorlevel 1 (
    echo  [ERROR] 交易执行层安装失败！
    pip install "alpaca-py>=0.30" --quiet
    pip install "python-dotenv>=1.0" --quiet
    pip install "schedule>=1.2" --quiet
    pip install "loguru>=0.7" --quiet
)
echo  ✅ 交易执行层安装完成
echo.

:: ============================================
:: Step 5: Install UI and utilities
:: ============================================
echo  [Step 5/6] 安装 UI 和工具 (Streamlit + others)...
echo  ┌─ Streamlit (仪表盘)
echo  │  httpx (异步HTTP)
echo  │  pydantic (数据校验)
echo  └─

pip install ^
    "streamlit>=1.35" ^
    "httpx>=0.27" ^
    "pydantic>=2.5" ^
    --quiet

if errorlevel 1 (
    pip install "streamlit>=1.35" --quiet
    pip install "httpx>=0.27" --quiet
    pip install "pydantic>=2.5" --quiet
)
echo  ✅ UI 和工具安装完成
echo.

:: ============================================
:: Step 6: Optional - PyTorch (CPU only)
:: ============================================
echo  [Step 6/6] 安装 PyTorch (CPU版, 可选 - 用于LSTM/Transformer模型)...
echo  ℹ️  跳过此步骤可按 Ctrl+C。CPU版约 200MB，不含CUDA。
echo.

choice /C YN /M "是否安装 PyTorch CPU 版"
if errorlevel 2 (
    echo  ⏭️  跳过 PyTorch 安装
    goto verify
)

pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 (
    echo  [WARNING] PyTorch 安装失败，可稍后手动安装:
    echo  pip install torch --index-url https://download.pytorch.org/whl/cpu
) else (
    echo  ✅ PyTorch CPU 版安装完成
)

:verify
echo.
echo  ═══════════════════════════════════════════════════════
echo  🔍 正在验证安装结果...
echo  ═══════════════════════════════════════════════════════
echo.

python -c "import akshare; print(f'  ✅ AKShare {akshare.__version__}')" 2>nul || echo  ❌ AKShare 导入失败
python -c "import finnhub; print(f'  ✅ Finnhub')" 2>nul || echo  ❌ Finnhub 导入失败
python -c "import alpaca; print(f'  ✅ Alpaca-py')" 2>nul || echo  ❌ Alpaca-py 导入失败
python -c "import qlib; print(f'  ✅ Qlib')" 2>nul || echo  ❌ Qlib 导入失败
python -c "import pandas; print(f'  ✅ Pandas {pandas.__version__}')" 2>nul || echo  ❌ Pandas 导入失败
python -c "import numpy; print(f'  ✅ NumPy {numpy.__version__}')" 2>nul || echo  ❌ NumPy 导入失败
python -c "import lightgbm; print(f'  ✅ LightGBM {lightgbm.__version__}')" 2>nul || echo  ❌ LightGBM 导入失败
python -c "import sklearn; print(f'  ✅ scikit-learn {sklearn.__version__}')" 2>nul || echo  ❌ scikit-learn 导入失败
python -c "import streamlit; print(f'  ✅ Streamlit {streamlit.__version__}')" 2>nul || echo  ❌ Streamlit 导入失败
python -c "import loguru; print(f'  ✅ Loguru')" 2>nul || echo  ❌ Loguru 导入失败
python -c "import schedule; print(f'  ✅ Schedule')" 2>nul || echo  ❌ Schedule 导入失败
python -c "import httpx; print(f'  ✅ httpx')" 2>nul || echo  ❌ httpx 导入失败
python -c "import torch; print(f'  ✅ PyTorch {torch.__version__}')" 2>nul || echo  ⏭️  PyTorch 未安装 (可选)
python -c "import matplotlib; print(f'  ✅ Matplotlib {matplotlib.__version__}')" 2>nul || echo  ❌ Matplotlib 导入失败
python -c "import plotly; print(f'  ✅ Plotly {plotly.__version__}')" 2>nul || echo  ❌ Plotly 导入失败

echo.
echo  ═══════════════════════════════════════════════════════
echo.
echo  📦 安装完成！
echo.
echo  下一步:
echo    1. 复制 .env.example 为 .env，填入你的 API Key
echo    2. 运行验证脚本: python scripts/verify_integration.py
echo    3. 启动仪表盘: streamlit run src/ui/dashboard.py
echo.
pause
