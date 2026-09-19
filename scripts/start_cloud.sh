#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# AI量化系统 - 云端一键启动脚本
# 在云端服务器上运行此脚本启动所有服务
# ═══════════════════════════════════════════════════════════════

set -e

# 切换到项目目录
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)

echo "═══════════════════════════════════════════════════════════════"
echo "  AI量化系统 - 云端服务启动"
echo "═══════════════════════════════════════════════════════════════"
echo "  项目目录: $PROJECT_DIR"
echo "  Python:   $(which python3)"
echo ""

# 检查 .env 配置
if [ ! -f .env ]; then
    echo "❌ .env 文件不存在！请先从 .env.cloud.example 复制并配置"
    exit 1
fi

# 检查 DEPLOY_MODE
DEPLOY_MODE=$(grep -E "^DEPLOY_MODE=" .env | cut -d= -f2 || echo "cloud")
echo "  部署模式: $DEPLOY_MODE"
echo ""

# ─── 启动模型同步 API ───
echo "[1/3] 启动模型同步 API..."
if ! pgrep -f "uvicorn.*model_sync" > /dev/null 2>&1; then
    nohup python3 -m src.api.model_sync > logs/api.log 2>&1 &
    echo "  ✅ API 已启动 (PID: $!)"
    echo "     日志: logs/api.log"
else
    echo "  ⏭️  API 已在运行"
fi
echo ""

# ─── 启动交易调度器 ───
echo "[2/3] 启动交易调度器..."
if ! pgrep -f "scheduler" > /dev/null 2>&1; then
    nohup python3 main.py scheduler > logs/scheduler.log 2>&1 &
    echo "  ✅ 调度器已启动 (PID: $!)"
    echo "     日志: logs/scheduler.log"
else
    echo "  ⏭️  调度器已在运行"
fi
echo ""

# ─── 启动 Dashboard (可选) ───
echo "[3/3] 启动 Dashboard..."
if ! pgrep -f "streamlit.*dashboard" > /dev/null 2>&1; then
    nohup streamlit run src/ui/dashboard.py --server.port 8501 --server.headless true > logs/dashboard.log 2>&1 &
    echo "  ✅ Dashboard 已启动 (PID: $!)"
    echo "     访问: http://$(hostname -I | awk '{print $1}'):8501"
    echo "     日志: logs/dashboard.log"
else
    echo "  ⏭️  Dashboard 已在运行"
fi
echo ""

# ─── 状态汇总 ───
echo "═══════════════════════════════════════════════════════════════"
echo "  服务状态:"
echo "  ─────────────────────────────────────────────────────────"
echo "  API:        http://0.0.0.0:8000/api/health"
echo "  Dashboard:  http://0.0.0.0:8501"
echo "  日志目录:   $PROJECT_DIR/logs/"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  停止所有服务: bash scripts/stop_cloud.sh"
echo ""
