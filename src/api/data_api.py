"""
云端数据 API 服务 - FastAPI
=====================================================
为本地仪表盘提供数据接口，实现"前端本地 + 后端云端"分离架构。

接口列表:
  GET  /api/health                      → 健康检查
  GET  /api/dashboard/overview           → 账户 + 持仓概览
  GET  /api/dashboard/strategies         → 策略列表
  GET  /api/dashboard/strategies/{sid}/backtest → 策略回测结果
  GET  /api/dashboard/notifications      → 通知历史
  GET  /api/dashboard/workflows          → 可用工作流
  GET  /api/dashboard/macro              → 宏观经济数据
  GET  /api/dashboard/yield-curve         → 美债收益率曲线
  GET  /api/dashboard/vix                → VIX 恐慌指数
  GET  /api/dashboard/options/{symbol}   → 期权链数据
  GET  /api/dashboard/crypto/prices      → 加密货币价格看板
  GET  /api/dashboard/crypto/positions   → 加密货币持仓
  GET  /api/dashboard/config             → 系统配置(脱敏)
  GET  /api/dashboard/scheduler/status   → 调度器状态

启动:
  python3.11 -m src.api.data_api
  或
  uvicorn src.api.data_api:app --host 0.0.0.0 --port 8001
"""

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from src.utils.config import config


# ─── FastAPI 应用 ───

app = FastAPI(title="AI 量化交易 - 数据 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 响应模型 ───

class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: str = ""
    deploy_mode: str = ""
    alpaca_connected: bool = False
    finnhub_connected: bool = False


class ApiResponse(BaseModel):
    success: bool = True
    data: Optional[dict | list] = None
    error: Optional[str] = None


def _ok(data) -> dict:
    """成功响应"""
    return {"success": True, "data": data}


def _err(msg: str) -> dict:
    """错误响应"""
    return {"success": False, "error": msg, "data": None}


def _serialize_strategy(s) -> dict:
    """将 StrategyConfig 序列化为 JSON 安全的 dict"""
    return {
        "strategy_id": s.strategy_id,
        "name": s.name,
        "model_type": s.model_type,
        "status": s.status,
        "stock_pool": list(s.stock_pool) if s.stock_pool else [],
        "factors": list(s.factors) if s.factors else [],
        "max_position": s.max_position,
        "rebalance_freq": s.rebalance_freq,
        "created_at": s.created_at.strftime("%Y-%m-%d %H:%M") if s.created_at else "",
        "updated_at": s.updated_at.strftime("%Y-%m-%d %H:%M") if s.updated_at else "",
    }


def _serialize_backtest(bt) -> dict:
    """将 BacktestResult 序列化为 JSON 安全的 dict"""
    return {
        "strategy_id": bt.strategy_id,
        "annual_return": bt.annual_return,
        "sharpe_ratio": bt.sharpe_ratio,
        "max_drawdown": bt.max_drawdown,
        "win_rate": bt.win_rate,
        "total_trades": bt.total_trades,
        "grade": bt.grade.value if bt.grade else "未回测",
        "equity_curve": list(bt.equity_curve) if bt.equity_curve else [],
        "daily_returns": list(bt.daily_returns) if bt.daily_returns else [],
    }


# ═══════════════════════════════════════════════════════
# 接口实现
# ═══════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return _ok({
        "status": "ok",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "deploy_mode": config.DEPLOY_MODE,
        "alpaca_connected": config.has_alpaca_keys(),
        "finnhub_connected": config.has_finnhub_key(),
        "fred_connected": config.has_fred_key(),
    })


@app.get("/api/dashboard/overview")
async def get_overview():
    """账户 + 持仓概览"""
    try:
        from src.execution.position_manager import position_manager

        account = position_manager.get_account()
        positions = position_manager.get_position_summary()

        account_data = None
        if account:
            account_data = {
                "equity": account.equity,
                "cash": account.cash,
                "buying_power": account.buying_power,
                "trading_mode": "模拟盘" if config.is_paper_trading() else "实盘",
            }

        return _ok({
            "account": account_data,
            "positions": positions,
        })
    except Exception as e:
        logger.error(f"获取概览失败: {e}")
        return _err(str(e))


@app.get("/api/dashboard/strategies")
async def get_strategies():
    """策略列表"""
    try:
        from src.engine.strategy_manager import strategy_manager

        strategies = strategy_manager.list_strategies()
        return _ok({
            "strategies": [_serialize_strategy(s) for s in strategies],
        })
    except Exception as e:
        logger.error(f"获取策略列表失败: {e}")
        return _err(str(e))


@app.get("/api/dashboard/strategies/{strategy_id}/backtest")
async def get_strategy_backtest(strategy_id: str):
    """策略回测结果"""
    try:
        from src.engine.strategy_manager import strategy_manager

        bt = strategy_manager.get_backtest_result(strategy_id)
        if bt:
            return _ok(_serialize_backtest(bt))
        return _ok(None)
    except Exception as e:
        logger.error(f"获取回测结果失败: {e}")
        return _err(str(e))


@app.get("/api/dashboard/notifications")
async def get_notifications(limit: int = Query(20, ge=1, le=200)):
    """通知历史"""
    try:
        from src.utils.notifier import notifier
        history = notifier.get_history(limit)
        return _ok({"notifications": history})
    except Exception as e:
        logger.error(f"获取通知失败: {e}")
        return _err(str(e))


@app.get("/api/dashboard/workflows")
async def get_workflows():
    """可用工作流列表"""
    try:
        from src.workflow.engine import workflow_engine
        workflows = workflow_engine.list_workflows()
        return _ok({"workflows": workflows})
    except Exception as e:
        logger.error(f"获取工作流失败: {e}")
        return _err(str(e))


@app.get("/api/dashboard/macro")
async def get_macro_data():
    """宏观经济仪表盘数据"""
    try:
        from src.data.collector import data_collector
        data = data_collector.get_macro_dashboard()
        return _ok(data)
    except Exception as e:
        logger.error(f"获取宏观数据失败: {e}")
        return _err(str(e))


@app.get("/api/dashboard/yield-curve")
async def get_yield_curve():
    """美债收益率曲线"""
    try:
        from src.data.collector import data_collector
        data = data_collector.get_yield_curve()
        return _ok(data)
    except Exception as e:
        logger.error(f"获取收益率曲线失败: {e}")
        return _err(str(e))


@app.get("/api/dashboard/vix")
async def get_vix():
    """VIX 恐慌指数"""
    try:
        from src.data.collector import data_collector
        data = data_collector.get_vix()
        return _ok(data)
    except Exception as e:
        logger.error(f"获取VIX失败: {e}")
        return _err(str(e))


@app.get("/api/dashboard/options/{symbol}")
async def get_options_chain(symbol: str, expiration_date: str = ""):
    """期权链数据"""
    try:
        from src.data.collector import data_collector
        import pandas as pd

        exp = expiration_date if expiration_date else None
        data = data_collector.get_options_chain(symbol, exp)

        # 将 DataFrame 序列化为 JSON
        result = {}
        for key, val in data.items():
            if isinstance(val, pd.DataFrame):
                result[key] = val.to_dict(orient="records")
            else:
                result[key] = val
        return _ok(result)
    except Exception as e:
        logger.error(f"获取期权链失败: {e}")
        return _err(str(e))


@app.get("/api/dashboard/crypto/prices")
async def get_crypto_prices():
    """加密货币价格看板"""
    try:
        from src.data.crypto_collector import crypto_collector
        from datetime import datetime, timedelta

        pairs = config.get_crypto_pairs()
        results = []
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        for pair in pairs:
            try:
                price = crypto_collector.get_latest_price(pair)
                df = crypto_collector.get_crypto_bars(
                    pair, timeframe="1Day", start=start_date, end=end_date
                )
                change_pct = 0.0
                volume_24h = 0.0
                if df is not None and len(df) >= 2:
                    prev_close = float(df.iloc[-2]["close"])
                    last_close = float(df.iloc[-1]["close"])
                    if prev_close > 0:
                        change_pct = (last_close - prev_close) / prev_close * 100
                    if "volume" in df.columns:
                        volume_24h = float(df.iloc[-1].get("volume", 0))

                results.append({
                    "symbol": pair,
                    "price": price if price and price > 0 else None,
                    "change_pct": change_pct,
                    "volume_24h": volume_24h,
                    "available": price is not None and price > 0,
                })
            except Exception:
                results.append({
                    "symbol": pair,
                    "price": None,
                    "change_pct": 0.0,
                    "volume_24h": 0.0,
                    "available": False,
                })

        return _ok({"prices": results})
    except Exception as e:
        logger.error(f"获取加密货币价格失败: {e}")
        return _err(str(e))


@app.get("/api/dashboard/crypto/positions")
async def get_crypto_positions():
    """加密货币持仓"""
    try:
        from src.execution.alpaca_client import alpaca_client
        from src.models import AssetType

        if not alpaca_client.available:
            return _ok({"positions": [], "account": None})

        account = alpaca_client.get_account()
        all_positions = alpaca_client.get_positions()
        crypto_positions = [
            p for p in all_positions
            if "/" in p.symbol or p.asset_type == AssetType.CRYPTO
        ]

        account_data = None
        if account:
            account_data = {
                "equity": account.equity,
                "cash": account.cash,
                "buying_power": account.buying_power,
                "trading_mode": "模拟盘" if config.is_paper_trading() else "实盘",
            }

        pos_data = []
        for p in crypto_positions:
            pos_data.append({
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_cost": p.avg_cost,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
                "unrealized_pnl_pct": p.unrealized_pnl_pct,
                "weight": p.weight,
            })

        return _ok({"positions": pos_data, "account": account_data})
    except Exception as e:
        logger.error(f"获取加密货币持仓失败: {e}")
        return _err(str(e))


@app.get("/api/dashboard/config")
async def get_config():
    """系统配置(脱敏)"""
    try:
        return _ok(config.to_dict())
    except Exception as e:
        return _err(str(e))


@app.get("/api/dashboard/scheduler/status")
async def get_scheduler_status():
    """调度器状态"""
    try:
        from src.scheduler.scheduler import trading_scheduler
        status = trading_scheduler.status()
        # 序列化 datetime
        if isinstance(status, dict):
            for task in status.get("tasks", []):
                if isinstance(task, dict):
                    for k, v in task.items():
                        if isinstance(v, datetime):
                            task[k] = v.strftime("%Y-%m-%d %H:%M:%S")
        return _ok(status)
    except Exception as e:
        logger.error(f"获取调度器状态失败: {e}")
        return _err(str(e))


# ─── 启动入口 ───

if __name__ == "__main__":
    import uvicorn
    port = int(config.CLOUD_API_PORT) if config.CLOUD_API_PORT else 8001
    # 数据 API 默认使用 8001 端口，与 model_sync (8000) 区分
    if port == 8000:
        port = 8001
    uvicorn.run(app, host="0.0.0.0", port=port)
