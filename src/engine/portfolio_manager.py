"""
多策略组合管理器
=====================================================
Phase 6 T6.2: 多策略组合权重分配
- 支持 3 种权重分配方式: 等权/风险平价/绩效加权
- 多策略信号聚合与冲突处理
- 组合再平衡
"""

import uuid
from datetime import datetime
from typing import Optional

import numpy as np
from loguru import logger

from src.models import (
    AggregatedSignal,
    AllocationMethod,
    BacktestResult,
    Direction,
    PortfolioConfig,
    PortfolioStatus,
    TradeSignal,
)


class PortfolioManager:
    """多策略组合管理器"""

    def __init__(self):
        self._portfolios: dict[str, PortfolioConfig] = {}
        self._strategy_performance: dict[str, BacktestResult] = {}
        self._last_rebalance: dict[str, datetime] = {}

    def create_portfolio(
        self,
        name: str,
        strategy_ids: list[str],
        method: AllocationMethod = AllocationMethod.EQUAL,
        risk_budget: float = 0.10,
        rebalance_freq: str = "daily",
    ) -> PortfolioConfig:
        """
        创建多策略组合

        Args:
            name: 组合名称
            strategy_ids: 策略ID列表
            method: 权重分配方式
            risk_budget: 风险预算
            rebalance_freq: 再平衡频率

        Returns:
            PortfolioConfig: 创建的组合配置
        """
        portfolio_id = f"port_{uuid.uuid4().hex[:8]}"

        # 初始等权分配
        n = len(strategy_ids)
        if n == 0:
            raise ValueError("策略列表不能为空")

        initial_weights = {sid: 1.0 / n for sid in strategy_ids}

        portfolio = PortfolioConfig(
            portfolio_id=portfolio_id,
            name=name,
            strategy_allocations=initial_weights,
            allocation_method=method,
            risk_budget=risk_budget,
            rebalance_freq=rebalance_freq,
        )

        self._portfolios[portfolio_id] = portfolio
        logger.info(f"创建组合: {name} ({portfolio_id}), 策略数: {n}, 方式: {method.value}")
        return portfolio

    def update_strategy_performance(self, strategy_id: str, result: BacktestResult):
        """更新策略绩效数据 (用于绩效加权)"""
        self._strategy_performance[strategy_id] = result

    def allocate_weights(
        self,
        portfolio: PortfolioConfig,
        method: Optional[AllocationMethod] = None,
    ) -> dict[str, float]:
        """
        计算策略权重分配

        Args:
            portfolio: 组合配置
            method: 分配方式 (None 则使用组合配置的方式)

        Returns:
            dict: {strategy_id: weight} 权重之和为 1.0
        """
        method = method or portfolio.allocation_method
        strategy_ids = list(portfolio.strategy_allocations.keys())
        n = len(strategy_ids)

        if n == 0:
            return {}

        if method == AllocationMethod.EQUAL:
            # 等权分配
            weights = {sid: 1.0 / n for sid in strategy_ids}

        elif method == AllocationMethod.RISK_PARITY:
            # 风险平价: 按策略波动率反比分配
            weights = self._risk_parity_weights(strategy_ids)

        elif method == AllocationMethod.PERFORMANCE:
            # 绩效加权: 按夏普比率加权
            weights = self._performance_weights(strategy_ids)

        else:
            logger.warning(f"未知分配方式 {method}, 使用等权")
            weights = {sid: 1.0 / n for sid in strategy_ids}

        # 归一化确保权重和为 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # 更新组合配置
        portfolio.strategy_allocations = weights
        portfolio.allocation_method = method
        portfolio.updated_at = datetime.now()

        logger.debug(f"权重分配 ({method.value}): {weights}")
        return weights

    def aggregate_signals(
        self,
        portfolio: PortfolioConfig,
        signals_by_strategy: dict[str, list[TradeSignal]],
    ) -> list[AggregatedSignal]:
        """
        聚合多策略信号

        处理逻辑:
        1. 按股票代码分组
        2. 按策略权重加权计算净方向和置信度
        3. 冲突处理: 买卖方向不一致时按加权净方向决定
        4. 目标权重为各策略加权平均

        Args:
            portfolio: 组合配置
            signals_by_strategy: {strategy_id: [TradeSignal, ...]}

        Returns:
            list[AggregatedSignal]: 聚合后的信号列表
        """
        weights = portfolio.strategy_allocations

        # 按 symbol 分组收集信号
        symbol_signals: dict[str, list[tuple[TradeSignal, float]]] = {}
        for strategy_id, signals in signals_by_strategy.items():
            strategy_weight = weights.get(strategy_id, 0)
            if strategy_weight <= 0:
                continue
            for sig in signals:
                if sig.symbol not in symbol_signals:
                    symbol_signals[sig.symbol] = []
                symbol_signals[sig.symbol].append((sig, strategy_weight))

        aggregated = []
        for symbol, weighted_sigs in symbol_signals.items():
            # 计算加权方向得分: buy=+1, sell=-1
            buy_score = 0.0
            sell_score = 0.0
            total_weight = 0.0
            weighted_target = 0.0
            confidence_sum = 0.0
            source_strategies = []
            reasons = []

            for sig, w in weighted_sigs:
                total_weight += w
                source_strategies.append(sig.strategy_id)
                confidence_sum += sig.confidence * w
                weighted_target += sig.target_weight * w
                reasons.append(f"{sig.strategy_id}:{sig.reason}")

                if sig.direction == Direction.BUY:
                    buy_score += w
                else:
                    sell_score += w

            if total_weight == 0:
                continue

            # 决定方向
            if buy_score > sell_score:
                direction = Direction.BUY
                net_score = buy_score / total_weight
            elif sell_score > buy_score:
                direction = Direction.SELL
                net_score = sell_score / total_weight
            else:
                # 完全冲突时跳过
                logger.debug(f"信号完全冲突, 跳过 {symbol}")
                continue

            # 聚合信号
            agg_sig = AggregatedSignal(
                signal_id=f"agg_{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                direction=direction,
                target_weight=min(weighted_target / total_weight, 1.0),
                source_strategies=list(set(source_strategies)),
                confidence=min(confidence_sum / total_weight * net_score, 1.0),
                reason=f"聚合{len(source_strategies)}个策略: " + "; ".join(reasons[:3]),
            )
            aggregated.append(agg_sig)

        logger.info(f"信号聚合: {sum(len(s) for s in signals_by_strategy.values())} 个原始信号 -> "
                    f"{len(aggregated)} 个聚合信号")
        return aggregated

    def rebalance(
        self,
        portfolio: PortfolioConfig,
        signals_by_strategy: dict[str, list[TradeSignal]],
    ) -> list[AggregatedSignal]:
        """
        执行组合再平衡

        1. 重新计算权重
        2. 聚合信号
        3. 记录再平衡时间
        """
        logger.info(f"组合再平衡: {portfolio.name}")

        # 重新分配权重
        self.allocate_weights(portfolio)

        # 聚合信号
        signals = self.aggregate_signals(portfolio, signals_by_strategy)

        # 记录时间
        self._last_rebalance[portfolio.portfolio_id] = datetime.now()

        return signals

    def get_portfolio(self, portfolio_id: str) -> Optional[PortfolioConfig]:
        """获取组合配置"""
        return self._portfolios.get(portfolio_id)

    def list_portfolios(self) -> list[PortfolioConfig]:
        """列出所有组合"""
        return list(self._portfolios.values())

    def get_portfolio_status(self, portfolio: PortfolioConfig) -> PortfolioStatus:
        """获取组合运行状态"""
        return PortfolioStatus(
            portfolio_id=portfolio.portfolio_id,
            name=portfolio.name,
            strategy_count=len(portfolio.strategy_allocations),
            total_weight=sum(portfolio.strategy_allocations.values()),
            last_rebalance=self._last_rebalance.get(portfolio.portfolio_id),
            allocation_method=portfolio.allocation_method.value,
            strategy_weights=portfolio.strategy_allocations.copy(),
        )

    def delete_portfolio(self, portfolio_id: str) -> bool:
        """删除组合"""
        if portfolio_id in self._portfolios:
            del self._portfolios[portfolio_id]
            self._last_rebalance.pop(portfolio_id, None)
            logger.info(f"删除组合: {portfolio_id}")
            return True
        return False

    # ─── 内部方法 ───

    def _risk_parity_weights(self, strategy_ids: list[str]) -> dict[str, float]:
        """风险平价权重: 波动率越低权重越高"""
        weights = {}
        default_vol = 0.15  # 默认年化波动率 15%

        for sid in strategy_ids:
            perf = self._strategy_performance.get(sid)
            if perf and perf.daily_returns:
                # 计算年化波动率
                vol = np.std(perf.daily_returns) * np.sqrt(252)
                vol = max(vol, 0.01)  # 避免除零
            else:
                vol = default_vol
            # 权重与波动率成反比
            weights[sid] = 1.0 / vol

        return weights

    def _performance_weights(self, strategy_ids: list[str]) -> dict[str, float]:
        """绩效加权: 夏普比率越高权重越高"""
        weights = {}
        default_sharpe = 0.5  # 默认夏普

        for sid in strategy_ids:
            perf = self._strategy_performance.get(sid)
            if perf:
                sharpe = max(perf.sharpe_ratio, 0.1)  # 避免负或零
            else:
                sharpe = default_sharpe
            weights[sid] = sharpe

        return weights


# 全局实例
portfolio_manager = PortfolioManager()
