"""
模型同步 API 服务 - FastAPI
=====================================================
功能:
  - POST /api/models/upload    上传模型文件 (LightGBM .txt / 任意)
  - GET  /api/models/list      查看已部署模型列表
  - GET  /api/models/active    查看当前激活模型
  - POST /api/models/activate  切换激活模型
  - GET  /api/health           云端健康检查
  - GET  /api/status           交易状态 (持仓/盈亏/风控)

安全: API Key Header 认证 (X-API-Key)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from src.utils.config import config


# ─── 请求/响应模型 ───

class ActivateRequest(BaseModel):
    """激活模型请求"""
    model_name: str


class ModelInfo(BaseModel):
    """模型信息"""
    name: str
    size: int
    modified: str
    is_active: bool = False


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    deploy_mode: str = "cloud"
    timestamp: str = ""
    models_count: int = 0
    active_model: Optional[str] = None
    memory_usage_mb: float = 0.0


class StatusResponse(BaseModel):
    """交易状态响应"""
    deploy_mode: str = "cloud"
    account: Optional[dict] = None
    positions: list = []
    risk_status: Optional[dict] = None
    active_model: Optional[str] = None
    scheduler_running: bool = False


# ─── 工具函数 ───

def _get_api_key() -> str:
    """获取 API Key"""
    return os.getenv("MODEL_API_KEY", "")


def _verify_api_key(x_api_key: Optional[str]) -> None:
    """验证 API Key"""
    api_key = _get_api_key()
    if api_key and x_api_key != api_key:
        raise HTTPException(status_code=401, detail="API Key 无效")


def _get_models_dir() -> Path:
    """获取模型目录"""
    models_dir = config.DATA_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def _get_active_model_path() -> Path:
    """获取激活模型标记文件路径"""
    return _get_models_dir() / "best_model.json"


def _get_active_model_name() -> Optional[str]:
    """获取当前激活的模型名"""
    marker = _get_active_model_path()
    if not marker.exists():
        return None
    try:
        with open(marker) as f:
            data = json.load(f)
        # 返回策略ID或模型文件名
        return data.get("strategy_id") or data.get("model_name")
    except Exception:
        return None


def _list_models() -> list[ModelInfo]:
    """列出所有模型文件"""
    models_dir = _get_models_dir()
    active_name = _get_active_model_name()
    models = []

    for f in sorted(models_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if f.is_file() and not f.name.startswith(".") and f.name not in ("best_model.json", "sync_history.json"):
            stat = f.stat()
            is_active = False
            # 检查是否是激活模型
            if active_name and active_name in f.name:
                is_active = True
            models.append(ModelInfo(
                name=f.name,
                size=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                is_active=is_active,
            ))

    return models


def _get_memory_usage() -> float:
    """获取当前进程内存使用 (MB)"""
    try:
        import resource
        # Linux: ru_maxrss 单位是 KB
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / 1024.0 / 1024.0
        except Exception:
            return 0.0


# ─── FastAPI 应用 ───

def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="AI量化系统 - 模型同步API",
        description="接收本地训练的模型，管理云端模型热加载",
        version="1.0.0",
    )

    # CORS 允许本地跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── 健康检查 ───

    @app.get("/api/health", response_model=HealthResponse)
    async def health_check():
        """云端健康检查"""
        models = _list_models()
        return HealthResponse(
            status="ok",
            deploy_mode=os.getenv("DEPLOY_MODE", "cloud"),
            timestamp=datetime.now().isoformat(),
            models_count=len(models),
            active_model=_get_active_model_name(),
            memory_usage_mb=_get_memory_usage(),
        )

    # ─── 上传模型 ───

    @app.post("/api/models/upload")
    async def upload_model(
        file: UploadFile = File(...),
        x_api_key: Optional[str] = Header(None),
    ):
        """
        上传模型文件

        - 接收 .txt (LightGBM), .pkl, .pth 等模型文件
        - 同时接收元数据 (通过文件名或 query 参数)
        - 上传后自动标记为最优模型
        """
        _verify_api_key(x_api_key)

        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名为空")

        models_dir = _get_models_dir()
        save_path = models_dir / file.filename

        # 写入文件
        try:
            with open(save_path, "wb") as f:
                content = await file.read()
                f.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

        stat = save_path.stat()
        logger.info(f"模型已上传: {file.filename} ({stat.st_size / 1024:.1f} KB)")

        return {
            "status": "success",
            "filename": file.filename,
            "size": stat.st_size,
            "saved_at": datetime.now().isoformat(),
            "message": f"模型 {file.filename} 上传成功",
        }

    # ─── 上传模型元数据 ───

    @app.post("/api/models/metadata")
    async def upload_metadata(
        payload: dict,
        x_api_key: Optional[str] = Header(None),
    ):
        """
        上传模型元数据 (训练时间、IC、股票池等)
        写入 best_model.json，标记为当前激活模型
        """
        _verify_api_key(x_api_key)

        marker = _get_active_model_path()
        payload["received_at"] = datetime.now().isoformat()

        try:
            with open(marker, "w") as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"元数据保存失败: {e}")

        logger.info(f"模型元数据已更新: {payload.get('strategy_id', 'unknown')}")

        return {
            "status": "success",
            "message": "模型元数据已更新，已设为激活模型",
            "strategy_id": payload.get("strategy_id"),
        }

    # ─── 列出模型 ───

    @app.get("/api/models/list")
    async def list_models(x_api_key: Optional[str] = Header(None)):
        """查看已部署模型列表"""
        _verify_api_key(x_api_key)

        models = _list_models()
        active = _get_active_model_name()

        return {
            "models": [m.model_dump() for m in models],
            "total": len(models),
            "active_model": active,
        }

    # ─── 查看当前激活模型 ───

    @app.get("/api/models/active")
    async def get_active_model(x_api_key: Optional[str] = Header(None)):
        """查看当前激活模型详情"""
        _verify_api_key(x_api_key)

        marker = _get_active_model_path()
        if not marker.exists():
            return {"active_model": None, "metadata": {}}

        try:
            with open(marker) as f:
                data = json.load(f)
            return {"active_model": data.get("strategy_id"), "metadata": data}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取激活模型失败: {e}")

    # ─── 切换激活模型 ───

    @app.post("/api/models/activate")
    async def activate_model(
        req: ActivateRequest,
        x_api_key: Optional[str] = Header(None),
    ):
        """切换激活模型"""
        _verify_api_key(x_api_key)

        models_dir = _get_models_dir()
        target = models_dir / req.model_name

        if not target.exists():
            raise HTTPException(status_code=404, detail=f"模型文件不存在: {req.model_name}")

        # 写入激活标记
        marker = _get_active_model_path()
        metadata = {
            "strategy_id": req.model_name,
            "model_name": req.model_name,
            "activated_at": datetime.now().isoformat(),
        }
        with open(marker, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"激活模型已切换: {req.model_name}")

        return {
            "status": "success",
            "message": f"已激活模型: {req.model_name}",
            "activated_at": datetime.now().isoformat(),
        }

    # ─── 交易状态 ───

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status(x_api_key: Optional[str] = Header(None)):
        """获取云端交易状态"""
        _verify_api_key(x_api_key)

        account_info = None
        positions_list = []
        risk_info = None

        try:
            from src.execution.alpaca_client import alpaca_client
            if alpaca_client.available:
                account = alpaca_client.get_account()
                if account:
                    account_info = {
                        "equity": account.equity,
                        "cash": account.cash,
                        "buying_power": account.buying_power,
                        "day_trade_count": account.day_trade_count,
                        "trading_mode": account.trading_mode.value,
                    }

                positions = alpaca_client.get_positions()
                positions_list = [
                    {
                        "symbol": p.symbol,
                        "qty": p.qty,
                        "market_value": p.market_value,
                        "unrealized_pnl": p.unrealized_pnl,
                        "unrealized_pnl_pct": p.unrealized_pnl_pct,
                        "weight": p.weight,
                    }
                    for p in positions
                ]

                # 风控状态
                if account and positions:
                    from src.execution.risk_guard import risk_guard
                    risk_status = risk_guard.check_portfolio(account, positions)
                    risk_info = {
                        "is_alert": risk_status.is_alert,
                        "current_drawdown": risk_status.current_drawdown,
                        "current_daily_loss": risk_status.current_daily_loss,
                        "alerts": risk_status.alerts,
                    }
        except Exception as e:
            logger.warning(f"获取交易状态失败: {e}")

        # 调度器状态
        scheduler_running = False
        try:
            from src.scheduler.scheduler import trading_scheduler
            scheduler_running = trading_scheduler._running
        except Exception:
            pass

        return StatusResponse(
            deploy_mode=os.getenv("DEPLOY_MODE", "cloud"),
            account=account_info,
            positions=positions_list,
            risk_status=risk_info,
            active_model=_get_active_model_name(),
            scheduler_running=scheduler_running,
        )

    # ─── 删除模型 ───

    @app.delete("/api/models/{model_name}")
    async def delete_model(
        model_name: str,
        x_api_key: Optional[str] = Header(None),
    ):
        """删除指定模型文件"""
        _verify_api_key(x_api_key)

        models_dir = _get_models_dir()
        target = models_dir / model_name

        if not target.exists():
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_name}")

        target.unlink()
        logger.info(f"模型已删除: {model_name}")

        return {"status": "success", "message": f"已删除: {model_name}"}

    return app


# ─── 便捷启动 ───

# 全局应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("CLOUD_API_PORT", "8000"))
    host = os.getenv("CLOUD_API_HOST", "0.0.0.0")

    logger.info(f"启动模型同步API: {host}:{port}")
    uvicorn.run(app, host=host, port=port)
