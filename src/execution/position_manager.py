"""
仓位管理器 - 持仓跟踪与调整
=====================================================
T3.2: 仓位管理器
- 持仓状态查询
- 止损止盈设置
- 自动止损执行
- 仓位再平衡
"""

from datetime import datetime
from typing import Optional

from loguru import logger

from src.execution.alpaca_client import alpaca_client
from src.execution.order_manager import order_manager
from src.execution.risk_guard import risk_guard
from src.models import Account, Direction, Position, TradeSignal
from src.utils.notifier import notifier


class PositionManager:
    """仓位管理器"""

    def __init__(self):
        self.client = alpaca_client
        self.order_mgr = order_manager
        self.risk_guard = risk_guard
        # 本地持仓缓存 (用于设置止损止盈)
        self._stop_losses: dict[str, float] = {}
        self._take_profits: dict[str, float] = {}

    def get_positions(self) -> list[Position]:
        """获取所有持仓"""
        return self.client.get_positions()

    def get_position(self, symbol: str) -> Optional[Position]:
        """获取单只持仓"""
        return self.client.get_position(symbol)

    def get_account(self) -> Optional[Account]:
        """获取账户信息"""
        return self.client.get_account()

    def set_stop_loss(self, symbol: str, price: float):
        """设置止损价"""
        self._stop_losses[symbol] = price
        logger.info(f"止损设置: {symbol} @ ${price:.2f}")

    def set_take_profit(self, symbol: str, price: float):
        """设置止盈价"""
        self._take_profits[symbol] = price
        logger.info(f"止盈设置: {symbol} @ ${price:.2f}")

    def set_stop_loss_pct(self, symbol: str, entry_price: float, pct: float = 0.05):
        """按百分比设置止损 (默认 5%)"""
        stop_price = entry_price * (1 - pct)
        self.set_stop_loss(symbol, stop_price)

    def set_take_profit_rr(
        self, symbol: str, entry_price: float,
        risk_reward: float = 2.0, stop_pct: float = 0.05,
    ):
        """按盈亏比设置止盈 (默认 2:1)"""
        stop_distance = entry_price * stop_pct
        tp_price = entry_price + stop_distance * risk_reward
        self.set_take_profit(symbol, tp_price)

    def check_stop_loss(self) -> list[TradeSignal]:
        """
        检查所有持仓的止损条件
        返回需要止损的信号列表
        """
        positions = self.get_positions()
        signals = []

        for pos in positions:
            # 检查内置风控止损 (亏损超过 5%)
            if self.risk_guard.should_stop_loss(pos):
                signal = TradeSignal(
                    signal_id=f"stoploss_{pos.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    symbol=pos.symbol,
                    direction=Direction.SELL,
                    target_weight=0.0,
                    reason=f"止损卖出: 亏损 {pos.unrealized_pnl_pct:.1%}",
                    confidence=1.0,
                )
                signals.append(signal)
                notifier.critical(
                    "止损触发",
                    f"{pos.symbol} 亏损 {pos.unrealized_pnl_pct:.1%}, 自动卖出",
                )
                continue

            # 检查自定义止损价
            stop_price = self._stop_losses.get(pos.symbol)
            if stop_price and pos.market_value / pos.qty <= stop_price:
                signal = TradeSignal(
                    signal_id=f"stoploss_custom_{pos.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    symbol=pos.symbol,
                    direction=Direction.SELL,
                    target_weight=0.0,
                    reason=f"止损卖出: 触及止损价 ${stop_price:.2f}",
                    confidence=1.0,
                )
                signals.append(signal)
                notifier.critical(
                    "止损触发",
                    f"{pos.symbol} 触及止损价 ${stop_price:.2f}",
                )

        return signals

    def check_take_profit(self) -> list[TradeSignal]:
        """
        检查所有持仓的止盈条件
        返回需要止盈的信号列表
        """
        positions = self.get_positions()
        signals = []

        for pos in positions:
            tp_price = self._take_profits.get(pos.symbol)
            if tp_price and pos.market_value / pos.qty >= tp_price:
                signal = TradeSignal(
                    signal_id=f"takeprofit_{pos.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    symbol=pos.symbol,
                    direction=Direction.SELL,
                    target_weight=0.0,
                    reason=f"止盈卖出: 触及止盈价 ${tp_price:.2f}",
                    confidence=0.9,
                )
                signals.append(signal)
                notifier.success(
                    "止盈触发",
                    f"{pos.symbol} 触及止盈价 ${tp_price:.2f}, 卖出获利",
                )

        return signals

    def execute_stop_loss(self) -> list:
        """执行止损检查并自动卖出"""
        signals = self.check_stop_loss()
        results = []
        for signal in signals:
            # 止损单优先: 直接清仓
            success = self.client.liquidate(signal.symbol)
            if success:
                self._stop_losses.pop(signal.symbol, None)
                self._take_profits.pop(signal.symbol, None)
                results.append(signal)
            else:
                # 如果清仓失败, 尝试通过订单管理器
                order = self.order_mgr.execute_signal(signal)
                if order:
                    results.append(signal)
        return results

    def execute_take_profit(self) -> list:
        """执行止盈检查并自动卖出"""
        signals = self.check_take_profit()
        results = []
        for signal in signals:
            success = self.client.liquidate(signal.symbol)
            if success:
                self._stop_losses.pop(signal.symbol, None)
                self._take_profits.pop(signal.symbol, None)
                results.append(signal)
            else:
                order = self.order_mgr.execute_signal(signal)
                if order:
                    results.append(signal)
        return results

    def get_position_summary(self) -> dict:
        """获取持仓摘要 (用于展示)"""
        account = self.get_account()
        positions = self.get_positions()

        if not account:
            return {"error": "无法获取账户信息"}

        summary = {
            "total_equity": account.equity,
            "cash": account.cash,
            "buying_power": account.buying_power,
            "position_count": len(positions),
            "total_market_value": sum(p.market_value for p in positions),
            "total_unrealized_pnl": sum(p.unrealized_pnl for p in positions),
            "positions": [],
        }

        for p in positions:
            summary["positions"].append({
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_cost": p.avg_cost,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
                "unrealized_pnl_pct": p.unrealized_pnl_pct,
                "weight": p.weight,
                "stop_loss": self._stop_losses.get(p.symbol),
                "take_profit": self._take_profits.get(p.symbol),
            })

        return summary

    def liquidate_all(self) -> int:
        """清仓所有持仓"""
        positions = self.get_positions()
        count = 0
        for p in positions:
            if self.client.liquidate(p.symbol):
                count += 1
        notifier.warning("清仓操作", f"已清仓 {count}/{len(positions)} 只股票")
        return count


# 全局实例
position_manager = PositionManager()
