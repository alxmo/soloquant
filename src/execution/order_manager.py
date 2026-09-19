"""
订单管理器 - 信号到订单的转换
=====================================================
T3.1: 订单管理器
流程: 风控检查 -> 计算数量 -> 下单 -> 记录
"""

from datetime import datetime
from typing import Optional

from loguru import logger

from src.data.akshare_collector import akshare_collector
from src.execution.alpaca_client import alpaca_client
from src.execution.risk_guard import risk_guard
from src.models import (
    Account, Direction, Order, Position, TradeSignal,
)
from src.utils.notifier import notifier


class OrderManager:
    """订单管理器"""

    def __init__(self):
        self.client = alpaca_client
        self.risk_guard = risk_guard
        # 执行历史 (本会话)
        self._execution_history: list[dict] = []

    def execute_signal(
        self, signal: TradeSignal,
    ) -> Optional[Order]:
        """
        执行交易信号
        流程: 风控检查 -> 计算数量 -> 下单 -> 记录
        """
        if not self.client.available:
            logger.error("Alpaca 客户端不可用, 无法执行交易信号")
            return None

        # 1. 获取账户和持仓
        account = self.client.get_account()
        if account is None:
            logger.error("无法获取账户信息")
            return None
        positions = self.client.get_positions()

        # 2. 风控检查
        risk_result = self.risk_guard.check_order(signal, account, positions)
        if not risk_result.passed:
            logger.warning(f"订单被风控拦截: {signal.symbol} 原因: {risk_result.reason}")
            self._record(signal, None, "rejected_by_risk", risk_result.reason)
            return None

        # 3. 计算下单数量
        qty = self._calculate_quantity(signal, account, positions)
        if qty <= 0:
            logger.info(f"{signal.symbol}: 计算数量为 0, 跳过下单")
            self._record(signal, None, "skipped_zero_qty", "计算数量为0")
            return None

        # 4. 获取当前价格 (用于日志)
        current_price = self._get_current_price(signal.symbol)

        # 5. 下单
        side = signal.direction.value
        order = self.client.submit_market_order(signal.symbol, qty, side)
        if order is None:
            logger.error(f"下单失败: {signal.symbol}")
            self._record(signal, None, "order_failed", "下单失败")
            return None

        # 6. 记录
        self.risk_guard.record_trade()
        self._record(signal, order, "submitted", "")

        notifier.info(
            "订单已提交",
            f"{side.upper()} {qty:.2f} {signal.symbol} "
            f"(~${qty * current_price:,.2f}) [{'模拟盘' if account.trading_mode.value == 'paper' else '实盘'}]"
        )

        return order

    def execute_signals(
        self, signals: list[TradeSignal],
    ) -> list[Order]:
        """批量执行交易信号"""
        results = []
        for signal in signals:
            order = self.execute_signal(signal)
            if order:
                results.append(order)
        logger.info(f"批量执行: {len(results)}/{len(signals)} 个订单成功")
        return results

    def cancel_all_pending(self) -> int:
        """撤销所有未完成订单"""
        orders = self.client.get_orders(status="open")
        count = 0
        for o in orders:
            if self.client.cancel_order(o.order_id):
                count += 1
        logger.info(f"撤销 {count} 个未完成订单")
        return count

    def get_execution_history(self) -> list[dict]:
        """获取本会话执行历史"""
        return self._execution_history

    # ─── 内部方法 ───

    def _calculate_quantity(
        self, signal: TradeSignal,
        account: Account, positions: list[Position],
    ) -> float:
        """计算下单数量"""
        current_pos = next(
            (p for p in positions if p.symbol == signal.symbol), None
        )

        # 目标价值 = 总资产 × 目标权重
        target_value = account.equity * signal.target_weight
        current_value = current_pos.market_value if current_pos else 0.0
        delta_value = target_value - current_value

        # 变动太小则跳过
        if abs(delta_value) < 1:
            return 0.0

        # 获取当前价格
        price = self._get_current_price(signal.symbol)
        if price is None or price <= 0:
            logger.warning(f"无法获取 {signal.symbol} 当前价格, 跳过")
            return 0.0

        qty = abs(delta_value) / price

        # ── 卖出时检查可用数量 (防止重复下单导致 insufficient qty) ──
        if signal.direction == Direction.SELL and current_pos:
            available_qty = getattr(current_pos, "qty_available", None)
            if available_qty is None:
                available_qty = current_pos.qty  # fallback: 假设全部可用
            if qty > available_qty:
                if available_qty <= 0.01:
                    logger.info(
                        f"{signal.symbol}: 可用数量为 {available_qty:.4f} (持仓 {current_pos.qty:.4f} "
                        f"已被挂单锁定), 跳过卖出"
                    )
                    return 0.0
                logger.warning(
                    f"{signal.symbol}: 卖单数量 {qty:.4f} > 可用数量 {available_qty:.4f} "
                    f"(持仓 {current_pos.qty:.4f}, 部分被挂单锁定), 调整为可用数量"
                )
                qty = available_qty

        # 确保数量合理 (至少 0.01 股, Alpaca 支持小数股)
        qty = round(qty, 4)
        if qty < 0.001:
            return 0.0

        return qty

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        # 先查 Alpaca 持仓中的均价, 再查 AKShare 最新价
        price = akshare_collector.get_latest_price(symbol)
        if price is None or price <= 0:
            # 尝试从 Alpaca 获取
            pos = self.client.get_position(symbol)
            if pos:
                price = pos.avg_cost
        return price

    def _record(
        self, signal: TradeSignal, order: Optional[Order],
        status: str, note: str,
    ):
        """记录执行历史"""
        self._execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "target_weight": signal.target_weight,
            "reason": signal.reason,
            "confidence": signal.confidence,
            "order_id": order.order_id if order else None,
            "status": status,
            "note": note,
        })


# 全局实例
order_manager = OrderManager()
