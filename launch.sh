#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Soloquant 一键启动脚本 (macOS / Linux)
# 用法: chmod +x launch.sh && ./launch.sh
# ═══════════════════════════════════════════════════════════════
set -e

cd "$(dirname "$0")"

echo "══════════════════════════════════════════════════"
echo "  🤖 Soloquant - AI 全自动美股量化交易系统"
echo "══════════════════════════════════════════════════"
echo ""

# 优先使用虚拟环境中的 Python
if [ -d ".venv" ]; then
    PY=".venv/bin/python"
    echo "📦 使用虚拟环境: .venv"
else
    PY="python3"
fi

# 检查 Python
if ! command -v $PY &> /dev/null; then
    echo "❌ 未找到 Python3, 请先安装 (需 3.10+)"
    exit 1
fi

# 首次运行: 创建虚拟环境并安装依赖
if [ ! -d ".venv" ]; then
    echo "🔧 首次运行, 创建虚拟环境并安装依赖 (约需几分钟)..."
    $PY -m venv .venv
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install -e ".[local]"
fi

# 启动
exec $PY run.py
