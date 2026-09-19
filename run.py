#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soloquant 跨平台启动器
=====================================================
适用于 Windows / macOS / Linux, 一条命令启动仪表盘:

    python run.py

功能:
- 检查 Python 版本 (需 3.10+)
- 检查核心依赖是否安装, 缺失时给出安装提示
- 首次运行自动从 .env.example 生成 .env 配置模板
- 启动 Streamlit 仪表盘并自动打开浏览器
"""

import os
import sys
import importlib
import subprocess
import webbrowser
import threading

# 项目根目录 (run.py 所在目录)
ROOT = os.path.dirname(os.path.abspath(__file__))

# 核心依赖清单: (导入名, pip 包名)
CORE_DEPS = [
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("loguru", "loguru"),
    ("dotenv", "python-dotenv"),
    ("pydantic", "pydantic"),
    ("httpx", "httpx"),
]

# 可选依赖清单: 缺失只提示, 不阻塞启动
OPTIONAL_DEPS = [
    ("qlib", "pyqlib"),            # 量化引擎 (策略训练)
    ("lightgbm", "lightgbm"),      # 模型训练
    ("torch", "torch"),            # LSTM 深度学习 (可选)
    ("alpaca", "alpaca-py"),       # 模拟交易
    ("finnhub", "finnhub-python"), # 金融数据
    ("akshare", "akshare"),        # 行情数据
    ("plotly", "plotly"),          # 图表
]


def check_python():
    """检查 Python 版本"""
    if sys.version_info < (3, 10):
        print(f"❌ 需要 Python 3.10+, 当前版本: {sys.version.split()[0]}")
        print("   请从 https://www.python.org/downloads/ 安装新版本")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")


def check_deps():
    """检查依赖, 返回缺失的核心依赖列表"""
    missing = []
    for mod, pkg in CORE_DEPS:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"❌ 缺少核心依赖: {', '.join(missing)}")
        print("\n请先安装依赖:")
        print(f'   pip install -e ".[local]"')
        print("   或: pip install " + " ".join(missing))
        sys.exit(1)
    print("✅ 核心依赖完整")

    # 可选依赖提示
    for mod, pkg in OPTIONAL_DEPS:
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"⚠️  可选依赖未安装: {pkg} (对应功能将不可用)")


def prepare_env():
    """首次运行时生成 .env 模板"""
    env_path = os.path.join(ROOT, ".env")
    example = os.path.join(ROOT, ".env.example")
    if not os.path.exists(env_path) and os.path.exists(example):
        import shutil
        shutil.copy2(example, env_path)
        print("📝 已生成 .env 配置文件, 请编辑填入你的 API Key:")
        print(f"   {env_path}")
        print("   (Alpaca/Finnhub 免费申请, 见 README.md)\n")


def open_browser():
    """延迟打开浏览器, 等待 Streamlit 启动"""
    import time
    time.sleep(4)
    webbrowser.open("http://localhost:8501")


def main():
    print("═" * 50)
    print("  🤖 Soloquant - AI 全自动美股量化交易系统")
    print("═" * 50 + "\n")

    check_python()
    check_deps()
    prepare_env()

    # 启动 Streamlit
    dashboard = os.path.join(ROOT, "src", "ui", "dashboard.py")
    if not os.path.exists(dashboard):
        print(f"❌ 未找到仪表盘入口: {dashboard}")
        sys.exit(1)

    print("\n🚀 启动仪表盘: http://localhost:8501")
    print("   按 Ctrl+C 停止\n")

    # 后台线程打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()

    # 启动 Streamlit (阻塞运行)
    cmd = [
        sys.executable, "-m", "streamlit", "run", dashboard,
        "--server.port", "8501",
        "--server.headless", "true",
    ]
    try:
        subprocess.run(cmd, cwd=ROOT)
    except KeyboardInterrupt:
        print("\n👋 已停止")


if __name__ == "__main__":
    main()
