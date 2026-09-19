#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# AI量化系统 - 云端停止所有服务
# ═══════════════════════════════════════════════════════════════

echo "停止 AI量化系统云端服务..."

# 停止 API
if pgrep -f "uvicorn.*model_sync" > /dev/null 2>&1; then
    pkill -f "uvicorn.*model_sync"
    echo "  ✅ API 已停止"
else
    echo "  ⏭️  API 未运行"
fi

# 停止调度器
if pgrep -f "main.py scheduler" > /dev/null 2>&1; then
    pkill -f "main.py scheduler"
    echo "  ✅ 调度器已停止"
else
    echo "  ⏭️  调度器未运行"
fi

# 停止 Dashboard
if pgrep -f "streamlit.*dashboard" > /dev/null 2>&1; then
    pkill -f "streamlit.*dashboard"
    echo "  ✅ Dashboard 已停止"
else
    echo "  ⏭️  Dashboard 未运行"
fi

echo ""
echo "所有服务已停止。"
