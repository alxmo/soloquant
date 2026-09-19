"""
组合监控器 - 实时监控持仓和策略表现
=====================================================
T4.1: 组合监控 (Phase 7 T1 升级为事件驱动)
- 定时检查持仓盈亏 (轮询模式, 降级方案)
- 实时价格驱动监控 (WebSocket 模式, 主方案)
- 止损止盈自动触发
- 风控告警
- 性能报告
"""

import asyncio
from datetime import datetime

from loguru import logger

from src.execution.alpaca_client import alpaca_client
from src.execution.position_manager import position_manager
from src.execution.risk_guard import risk_guard
from src.models import Account, Position, RealtimeQuote, RiskStatus
from src.utils.config import config
from src.utils.notifier import notifier


class PortfolioMonitor:
    """组合监控器 (支持轮询 + 事件驱动双模式)"""

    def __init__(self):
        self.client = alpaca_client
        self.pos_mgr = position_manager
        self.risk_guard = risk_guard
        self._monitoring = False
        self._realtime_monitoring = False
        self._last_check = None
        self._check_interval = 60  # 秒
        # 历史净值记录
        self._equity_history: list[tuple[datetime, float]] = []
        # 实时持仓盈亏缓存: symbol -> {price, pnl, pnl_pct, market_value}
        self._realtime_pnl: dict[str, dict] = {}
        # 实时管理器 (延迟导入, 避免循环依赖)
        self._realtime_manager = None

    async def start_monitoring(self, interval_seconds: int = 60):
        """启动异步监控循环"""
        self._monitoring = True
        self._check_interval = interval_seconds
        logger.info(f"组合监控已启动 (间隔 {interval_seconds}s)")

        while self._monitoring:
            try:
                await self.check_once()
            except Exception as e:
                logger.error(f"监控检查异常: {e}")

            await asyncio.sleep(self._check_interval)

    def stop_monitoring(self):
        """停止监控"""
        self._monitoring = False
        logger.info("组合监控已停止")

    # ─── 实时事件驱动模式 (Phase 7 T1) ───

    async def start_realtime_monitoring(self, symbols: list[str] = None):
        """
        启动实时事件驱动监控 (WebSocket)

        Args:
            symbols: 要监控的股票列表, 默认使用当前持仓 + 关注列表
        """
        if self._realtime_monitoring:
            logger.warning("实时监控已在运行")
            return

        # 延迟导入, 避免循环依赖
        from src.data.realtime_manager import realtime_manager
        self._realtime_manager = realtime_manager

        # 确定监控范围
        if symbols is None:
            symbols = set()
            # 加入当前持仓
            if self.client.available:
                positions = self.client.get_positions()
                symbols.update(p.symbol for p in positions)
            # 加入关注列表
            watchlist = config.WATCHLIST or config.DEFAULT_WATCHLIST
            symbols.update(watchlist)
            symbols = list(symbols)

        if not symbols:
            logger.warning("无监控目标, 实时监控未启动")
            return

        # 注册回调
        self._realtime_manager.register_callback("on_price_update", self._on_price_update)
        self._realtime_manager.register_callback("on_price_alert", self._on_price_alert)

        # 启动实时数据流
        await self._realtime_manager.start(symbols)
        self._realtime_monitoring = True
        logger.info(f"实时监控已启动, 监控 {len(symbols)} 只股票")

    async def stop_realtime_monitoring(self):
        """停止实时监控"""
        if self._realtime_manager:
            await self._realtime_manager.stop()
        self._realtime_monitoring = False
        self._realtime_pnl.clear()
        logger.info("实时监控已停止")

    def _on_price_update(self, quote: RealtimeQuote):
        """价格更新回调 - 更新持仓实时盈亏"""
        symbol = quote.symbol
        if not self.client.available:
            return

        # 检查是否为持仓股票
        position = self.client.get_position(symbol)
        if position is None:
            return

        # 计算实时盈亏
        current_price = quote.price
        market_value = current_price * position.qty
        unrealized_pnl = (current_price - position.avg_cost) * position.qty
        unrealized_pnl_pct = (
            (current_price - position.avg_cost) / position.avg_cost
            if position.avg_cost > 0 else 0.0
        )

        self._realtime_pnl[symbol] = {
            "symbol": symbol,
            "price": current_price,
            "qty": position.qty,
            "market_value": market_value,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "timestamp": quote.timestamp.isoformat(),
            "source": quote.source,
        }

        # 检查止损止盈
        self._check_realtime_stop_loss(symbol, current_price, position)

    def _on_price_alert(self, alert):
        """价格异动告警回调"""
        notifier.warning(
            f"价格异动 - {alert.symbol}",
            alert.message,
        )
        logger.warning(f"🚨 实时告警: {alert.message}")

    def _check_realtime_stop_loss(self, symbol: str, current_price: float, position):
        """实时止损止盈检查"""
        # 止损检查
        if position.stop_loss and current_price <= position.stop_loss:
            logger.warning(f"实时止损触发: {symbol} 价格  <= 止损线 ")
            notifier.critical(
                "实时止损",
                f"{symbol} 触发止损! 当前 , 止损线 ",
            )
            try:
                self.pos_mgr.execute_stop_loss()
            except Exception as e:
                logger.error(f"止损执行失败: {e}")

        # 止盈检查
        if position.take_profit and current_price >= position.take_profit:
            logger.info(f"实时止盈触发: {symbol} 价格  >= 止盈线 ")
            notifier.info(
                "实时止盈",
                f"{symbol} 触发止盈! 当前 , 止盈线 ",
            )
            try:
                self.pos_mgr.execute_take_profit()
            except Exception as e:
                logger.error(f"止盈执行失败: {e}")

    def get_realtime_positions(self) -> list[dict]:
        """
        获取带实时价格的持仓列表

        Returns:
            list[dict]: 每个持仓包含实时价格和盈亏
        """
        if not self._realtime_pnl:
            return []

        return list(self._realtime_pnl.values())

    def get_realtime_status(self) -> dict:
        """获取实时监控状态"""
        return {
            "realtime_monitoring": self._realtime_monitoring,
            "realtime_positions_count": len(self._realtime_pnl),
            "realtime_manager_status": (
                self._realtime_manager.get_status()
                if self._realtime_manager else None
            ),
            "recent_alerts": (
                [a.message for a in self._realtime_manager.get_alerts(limit=5)]
                if self._realtime_manager else []
            ),
        }

    async def check_once(self):
        """执行一次监控检查"""
        self._last_check = datetime.now()

        if not self.client.available:
            return

        # 1. 账户和持仓
        account = self.client.get_account()
        positions = self.client.get_positions()

        if not account:
            return

        # 记录净值
        self._equity_history.append((datetime.now(), account.equity))
        # 只保留最近 1000 条
        if len(self._equity_history) > 1000:
            self._equity_history = self._equity_history[-1000:]

        # 2. 风控检查
        risk_status = self.risk_guard.check_portfolio(account, positions)

        # 3. 止损检查
        stop_signals = self.pos_mgr.check_stop_loss()
        if stop_signals:
            notifier.critical(
                "自动止损",
                f"检测到 {len(stop_signals)} 只股票需要止损, 正在执行..."
            )
            self.pos_mgr.execute_stop_loss()

        # 4. 止盈检查
        tp_signals = self.pos_mgr.check_take_profit()
        if tp_signals:
            self.pos_mgr.execute_take_profit()

        # 5. 定期报告 (每 10 分钟)
        if self._equity_history and len(self._equity_history) % 10 == 0:
            self._log_status(account, positions, risk_status)

    def get_portfolio_status(self) -> dict:
        """获取当前组合状态"""
        if not self.client.available:
            return {"error": "交易客户端不可用"}

        account = self.client.get_account()
        positions = self.client.get_positions()

        if not account:
            return {"error": "无法获取账户"}

        risk_status = self.risk_guard.check_portfolio(account, positions)

        return {
            "timestamp": datetime.now().isoformat(),
            "account": {
                "equity": account.equity,
                "cash": account.cash,
                "buying_power": account.buying_power,
                "trading_mode": account.trading_mode.value,
                "daily_loss_pct": (
                    (account.last_equity - account.equity) / account.last_equity
                    if account.last_equity > 0 else 0
                ),
            },
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct,
                    "weight": p.weight,
                }
                for p in positions
            ],
            "risk": {
                "is_alert": risk_status.is_alert,
                "alerts": risk_status.alerts,
                "current_drawdown": risk_status.current_drawdown,
                "current_daily_loss": risk_status.current_daily_loss,
                "max_single_position": risk_status.max_single_position,
            },
            "monitoring": self._monitoring,
            "realtime_monitoring": self._realtime_monitoring,
            "realtime_positions": self.get_realtime_positions() if self._realtime_monitoring else [],
            "last_check": self._last_check.isoformat() if self._last_check else None,
        }

    def get_performance_report(self) -> dict:
        """生成绩效报告"""
        if not self._equity_history:
            return {"error": "无历史数据"}

        equities = [e[1] for e in self._equity_history]
        timestamps = [e[0] for e in self._equity_history]

        initial = equities[0]
        current = equities[-1]
        total_return = (current - initial) / initial if initial > 0 else 0

        # 高水位和回撤
        peak = max(equities)
        current_dd = (peak - current) / peak if peak > 0 else 0

        return {
            "period": {
                "start": timestamps[0].isoformat(),
                "end": timestamps[-1].isoformat(),
                "data_points": len(equities),
            },
            "initial_equity": initial,
            "current_equity": current,
            "peak_equity": peak,
            "total_return": total_return,
            "current_drawdown": current_dd,
            "equity_history": [
                {"time": t.isoformat(), "equity": e}
                for t, e in self._equity_history[-100:]  # 最近100条
            ],
        }

    # ─── 内部 ───

    def _log_status(
        self, account: Account,
        positions: list[Position], risk: RiskStatus,
    ):
        """日志输出组合状态"""
        total_pnl = sum(p.unrealized_pnl for p in positions)
        logger.info(
            f"📊 组合状态 | 净值 ${account.equity:,.2f} | "
            f"现金 ${account.cash:,.2f} | "
            f"持仓 {len(positions)} 只 | "
            f"浮动盈亏 ${total_pnl:,.2f} | "
            f"回撤 {risk.current_drawdown:.1%}"
        )
        if risk.is_alert:
            for alert in risk.alerts:
                logger.warning(f"  ⚠️ {alert}")


# 全局实例
portfolio_monitor = PortfolioMonitor()
