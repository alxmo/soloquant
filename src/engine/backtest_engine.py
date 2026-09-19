"""
回测引擎封装
=====================================================
T2.4: 回测引擎封装 + T2.7: 绩效评估器
- 支持基于 Qlib 回测和独立回测
- 绩效指标: 年化收益/夏普/最大回撤/胜率
- 策略评级 (A/B/C/D)
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.models import BacktestResult, StrategyConfig, StrategyGrade


class BacktestEngine:
    """回测引擎"""

    def __init__(self):
        from src.engine.qlib_wrapper import qlib_wrapper
        self.qlib = qlib_wrapper

    def run_backtest(
        self,
        strategy: StrategyConfig,
        start: str,
        end: str,
        initial_capital: float = 100000,
    ) -> BacktestResult:
        """
        运行回测

        优先使用 Qlib 回测, 不可用则降级为独立回测
        """
        logger.info(f"回测开始: {strategy.name} ({strategy.model_type})")
        logger.info(f"  时间范围: {start} ~ {end}")
        logger.info(f"  初始资金: ${initial_capital:,.2f}")
        logger.info(f"  股票池: {strategy.stock_pool}")

        if self.qlib.available:
            return self._backtest_with_qlib(strategy, start, end, initial_capital)
        else:
            return self._backtest_standalone(strategy, start, end, initial_capital)

    def run_simple_backtest(
        self,
        signals: list,  # list of (date, symbol, direction, weight)
        stock_pool: list[str],
        start: str,
        end: str,
        initial_capital: float = 100000,
        strategy_id: str = "simple",
    ) -> BacktestResult:
        """
        简化回测: 给定信号列表, 模拟执行

        Args:
            signals: [(date_str, symbol, "buy"/"sell", weight), ...]
        """
        from src.data.akshare_collector import akshare_collector
        from src.utils.parallel import parallel_map

        logger.info(f"简化回测: {len(signals)} 个信号")

        # 并行获取行情数据
        def _fetch_prices(symbol: str) -> tuple[str, Optional[pd.Series]]:
            try:
                df = akshare_collector.get_us_stock_daily(symbol, start=start, end=end)
                if len(df) > 0:
                    return symbol, df.set_index("date")["close"]
            except Exception:
                pass
            return symbol, None

        results = parallel_map(
            _fetch_prices, stock_pool,
            max_workers=min(4, len(stock_pool)),
            desc="并行获取回测行情",
            timeout=120,
        )

        price_data = {}
        for result in results:
            if result is None:
                continue
            symbol, price_series = result
            if price_series is not None and len(price_series) > 0:
                price_data[symbol] = price_series

        if not price_data:
            return BacktestResult(strategy_id=strategy_id)

        # 所有交易日
        all_dates = sorted(set().union(*[set(p.index) for p in price_data.values()]))

        # 模拟交易
        cash = initial_capital
        holdings: dict[str, float] = {}  # symbol -> qty
        equity_curve = []
        daily_returns = []

        prev_equity = initial_capital

        for date in all_dates:
            # 执行信号
            for sig_date, symbol, direction, weight in signals:
                if str(sig_date)[:10] != str(date)[:10]:
                    continue
                if symbol not in price_data:
                    continue

                price = price_data[symbol].get(date)
                if price is None or price <= 0:
                    continue

                total_equity = cash + sum(
                    holdings.get(s, 0) * price_data[s].get(date, 0)
                    for s in holdings
                )

                if direction == "buy":
                    target_value = total_equity * weight
                    current_value = holdings.get(symbol, 0) * price
                    delta = target_value - current_value
                    if delta > 0:
                        shares = delta / price
                        cash -= shares * price
                        holdings[symbol] = holdings.get(symbol, 0) + shares

                elif direction == "sell":
                    if symbol in holdings:
                        cash += holdings[symbol] * price
                        del holdings[symbol]

            # 计算当日总资产
            equity = cash + sum(
                holdings.get(s, 0) * price_data[s].get(date, 0)
                for s in holdings
            )
            equity_curve.append(equity)

            # 日收益率
            if prev_equity > 0:
                daily_returns.append((equity - prev_equity) / prev_equity)
            prev_equity = equity

        # 计算绩效指标
        return self._calc_performance(
            equity_curve, daily_returns, initial_capital,
            start, end, strategy_id
        )

    # ─── 绩效评估 ───

    def calc_risk_metrics(self, daily_returns: list[float]) -> dict:
        """计算风险指标"""
        if not daily_returns:
            return {}

        returns = np.array(daily_returns)

        # 年化收益
        total_return = np.prod(1 + returns) - 1
        n_days = len(returns)
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # 夏普比率 (假设无风险利率 0)
        std = np.std(returns, ddof=1) if len(returns) > 1 else 0
        sharpe = (np.mean(returns) / std * np.sqrt(252)) if std > 0 else 0

        # Sortino 比率 (只考虑下行波动)
        downside = returns[returns < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0
        sortino = (np.mean(returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0

        # 最大回撤
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0

        # 胜率
        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
        }

    # ─── 内部方法 ───

    def _backtest_with_qlib(
        self, strategy: StrategyConfig,
        start: str, end: str, initial_capital: float,
    ) -> BacktestResult:
        """使用 Qlib 回测"""
        try:
            # 确保 Qlib 已初始化
            if not self.qlib.available:
                self.qlib.init()

            from qlib.data.dataset import DatasetH
            from qlib.contrib.data.handler import Alpha158

            # 使用 Qlib 实际交易日历分割 (回测用 70% 训练, 30% 测试)
            calendar = self.qlib.D.calendar(start_time=start, end_time=end)
            calendar_list = [str(d)[:10] for d in calendar]
            split_idx = int(len(calendar_list) * 0.7)
            train_end = calendar_list[split_idx]
            test_start = train_end
            # 回测结束日提前一天，避免最后一个交易日索引越界
            actual_end = calendar_list[-2] if len(calendar_list) > 2 else calendar_list[-1]

            # 创建数据集
            data_handler_config = {
                "start_time": start,
                "end_time": actual_end,
                "fit_start_time": start,
                "fit_end_time": train_end,
                "instruments": strategy.stock_pool,
            }

            handler = Alpha158(**data_handler_config)
            dataset = DatasetH(
                handler=handler,
                segments={
                    "train": (start, train_end),
                    "test": (test_start, actual_end),
                }
            )
            dataset.prepare(["train", "test"])

            # 直接训练模型 (和 model_manager 保持一致)
            if strategy.model_type == "lightgbm":
                from qlib.contrib.model.gbdt import LGBModel
                params = {
                    "loss": "mse",
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "num_leaves": 32,
                    "num_threads": 4,
                    "early_stopping_rounds": 20,
                    "num_boost_round": 100,
                }
                if strategy.hyper_params:
                    params.update(strategy.hyper_params)
                model = LGBModel(**params)
                model.fit(dataset)
            else:
                from qlib.contrib.model.linear import LinearModel
                model = LinearModel()
                model.fit(dataset)

            # 运行回测 (使用测试集时间段)
            return self.qlib.run_backtest(
                model, dataset, strategy,
                test_start, actual_end, initial_capital,
            )

        except Exception as e:
            logger.error(f"Qlib 回测失败, 降级到独立回测: {e}")
            return self._backtest_standalone(strategy, start, end, initial_capital)

    def _backtest_standalone(
        self, strategy: StrategyConfig,
        start: str, end: str, initial_capital: float,
    ) -> BacktestResult:
        """独立回测 (不依赖 Qlib, 并行获取行情数据)

        修复: 原实现训练了模型却未使用预测结果 (纯等权买入)。
        现在用模型预测分数构建持仓权重, 模型失败时降级为等权并告警。
        """
        from src.data.akshare_collector import akshare_collector
        from src.engine.model_manager import model_manager
        from src.utils.parallel import parallel_map

        logger.info("独立回测模式 (AKShare + LightGBM, 并行获取行情)")

        # 训练模型
        model_result = model_manager.train_model(
            strategy.model_type,
            strategy.factors,
            strategy.stock_pool,
            start, end,
            strategy.hyper_params,
        )

        # 并行获取行情数据 (原来串行约20-40s，并行约5-10s)
        def _fetch_prices(symbol: str) -> tuple[str, Optional[pd.Series]]:
            """获取单只股票收盘价序列"""
            try:
                df = akshare_collector.get_us_stock_daily(symbol, start=start, end=end)
                if len(df) > 0:
                    return symbol, df.set_index("date")["close"]
            except Exception as e:
                logger.debug(f"获取 {symbol} 行情失败: {e}")
            return symbol, None

        results = parallel_map(
            _fetch_prices, strategy.stock_pool,
            max_workers=min(4, len(strategy.stock_pool)),
            desc="并行获取回测行情",
            timeout=120,
        )

        all_prices = {}
        for result in results:
            if result is None:
                continue
            symbol, price_series = result
            if price_series is not None and len(price_series) > 0:
                all_prices[symbol] = price_series

        if not all_prices:
            logger.error("无可用行情数据")
            return BacktestResult(strategy_id=strategy.strategy_id)

        # ─── 用模型预测分数构建持仓权重 ───
        weights = self._build_weights_from_prediction(
            model_result, strategy, list(all_prices.keys()),
        )

        all_dates = sorted(set().union(*[set(p.index) for p in all_prices.values()]))

        cash = initial_capital
        holdings: dict[str, float] = {}
        equity_curve = []
        daily_returns = []
        prev_equity = initial_capital

        # 模拟交易: 第一天按权重买入, 持有到最后
        for i, date in enumerate(all_dates):
            # 第一天买入
            if i == 0:
                for symbol, weight in weights.items():
                    price = all_prices[symbol].get(date)
                    if price and price > 0:
                        target_value = initial_capital * weight
                        shares = target_value / price
                        cash -= shares * price
                        holdings[symbol] = shares

            # 计算当日总资产
            equity = cash + sum(
                holdings.get(s, 0) * all_prices[s].get(date, 0)
                for s in holdings
            )
            equity_curve.append(equity)

            if prev_equity > 0:
                daily_returns.append((equity - prev_equity) / prev_equity)
            prev_equity = equity

        # 计算绩效
        return self._calc_performance(
            equity_curve, daily_returns, initial_capital,
            start, end, strategy.strategy_id,
        )

    def _build_weights_from_prediction(
        self,
        model_result,
        strategy: StrategyConfig,
        available_symbols: list[str],
    ) -> dict[str, float]:
        """根据模型预测分数构建持仓权重

        - 模型有效 (valid_ic > 0 且有预测分数): 按分数正比分配, 受 max_position 约束
        - 模型无效: 降级为等权分配并告警

        Returns:
            {symbol: weight} 字典, 权重和 <= 1.0
        """
        n_stocks = len(available_symbols)
        equal_weight = min(1.0 / n_stocks, strategy.max_position)

        # 模型无效或无预测能力 -> 等权降级
        if (model_result is None
                or not getattr(model_result, "model_path", "")
                or getattr(model_result, "valid_ic", 0) <= 0):
            logger.warning(
                "模型预测无效 (IC<=0 或未保存), 降级为等权买入 "
                f"(weight={equal_weight:.3f})"
            )
            return {s: equal_weight for s in available_symbols}

        # 尝试用模型对股票池打分
        try:
            from src.engine.signal_generator import signal_generator

            scores = signal_generator._predict_with_model_file(
                model_result, available_symbols,
            )
            if not scores:
                logger.warning("模型预测返回空, 降级为等权买入")
                return {s: equal_weight for s in available_symbols}

            # 只保留有行情的股票, 分数截断为非负 (负分股票不持有)
            valid_scores = {
                s: max(float(scores.get(s, 0.0)), 0.0) for s in available_symbols
            }
            total = sum(valid_scores.values())
            if total <= 0:
                logger.warning("模型预测分数全为负, 降级为等权买入")
                return {s: equal_weight for s in available_symbols}

            # 分数正比分配, 单只受 max_position 约束
            raw_weights = {s: v / total for s, v in valid_scores.items()}
            weights = {s: min(w, strategy.max_position) for s, w in raw_weights.items()}

            # 截断后归一化 (保持总仓位 <= 1)
            total_w = sum(weights.values())
            if total_w > 1.0:
                weights = {s: w / total_w for s, w in weights.items()}

            logger.info(
                f"按模型预测分配权重: {len(weights)} 只, "
                f"最大 {max(weights.values()):.3f}, 最小 {min(weights.values()):.3f}"
            )
            return weights

        except Exception as e:
            logger.warning(f"模型预测加权失败 ({e}), 降级为等权买入")
            return {s: equal_weight for s in available_symbols}

    def _calc_performance(
        self, equity_curve: list[float],
        daily_returns: list[float],
        initial_capital: float,
        start: str, end: str, strategy_id: str,
    ) -> BacktestResult:
        """计算绩效指标并生成结果"""
        metrics = self.calc_risk_metrics(daily_returns)

        # 评级
        grade = self._grade(
            metrics.get("annual_return", 0),
            metrics.get("sharpe_ratio", 0),
            metrics.get("max_drawdown", 0),
        )

        final_value = equity_curve[-1] if equity_curve else initial_capital

        result = BacktestResult(
            strategy_id=strategy_id,
            period=(start, end),
            initial_capital=initial_capital,
            final_value=final_value,
            total_return=metrics.get("total_return", 0),
            annual_return=metrics.get("annual_return", 0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0),
            sortino_ratio=metrics.get("sortino_ratio", 0),
            max_drawdown=metrics.get("max_drawdown", 0),
            win_rate=metrics.get("win_rate", 0),
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            grade=grade,
            plain_summary=self._plain_summary(
                metrics.get("annual_return", 0),
                metrics.get("sharpe_ratio", 0),
                metrics.get("max_drawdown", 0),
                grade,
            ),
        )

        logger.info(f"回测结果: {grade}级 | 年化 {result.annual_return:.1%} | "
                    f"夏普 {result.sharpe_ratio:.2f} | 回撤 {result.max_drawdown:.1%} | "
                    f"胜率 {result.win_rate:.1%}")
        return result

    def _grade(
        self, annual_return: float, sharpe: float, max_dd: float,
    ) -> StrategyGrade:
        if annual_return > 0.20 and sharpe > 1.5 and max_dd < 0.10:
            return StrategyGrade.A
        elif annual_return > 0.15 and sharpe > 1.0 and max_dd < 0.15:
            return StrategyGrade.B
        elif annual_return > 0.10 and sharpe > 0.5 and max_dd < 0.20:
            return StrategyGrade.C
        else:
            return StrategyGrade.D

    def run_batch_backtest(
        self,
        strategies: list,
        start: str,
        end: str,
        initial_capital: float = 100000,
        max_workers: int = 4,
        progress_callback=None,
    ) -> list:
        """
        批量并行回测多个策略

        Args:
            strategies: StrategyConfig 列表
            start: 开始日期
            end: 结束日期
            initial_capital: 初始资金
            max_workers: 并行线程数
            progress_callback: 进度回调

        Returns:
            list[BacktestResult]: 按夏普比率降序排列
        """
        from src.engine.parallel_backtest import ParallelBacktester

        backtester = ParallelBacktester(max_workers=max_workers)
        return backtester.run_parallel(
            strategies, start, end, initial_capital, progress_callback
        )

    def _plain_summary(
        self, annual_return, sharpe, max_dd, grade,
    ) -> str:
        return (
            f"策略评级: {grade}级\n"
            f"年化收益 {annual_return:.1%} - "
            f"{'优秀' if annual_return > 0.2 else '良好' if annual_return > 0.15 else '一般' if annual_return > 0.1 else '不理想'}\n"
            f"最大回撤 {max_dd:.1%} - "
            f"{'很小' if max_dd < 0.05 else '可控' if max_dd < 0.15 else '较大'}\n"
            f"夏普比率 {sharpe:.2f} - "
            f"{'优秀' if sharpe > 1.5 else '不错' if sharpe > 1.0 else '一般' if sharpe > 0.5 else '较差'}\n"
            f"⚠️ 过去的收益不代表未来"
        )


# 全局实例
backtest_engine = BacktestEngine()
