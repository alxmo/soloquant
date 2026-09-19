"""
并行回测引擎
=====================================================
Phase 6 T6.4: 性能优化 - 多策略并行回测
- 基于 ThreadPoolExecutor 多线程并行
- 进度回调支持
- 结果汇总排序
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from loguru import logger

from src.models import BacktestResult, StrategyConfig


class ParallelBacktester:
    """多策略并行回测器"""

    def __init__(self, max_workers: int = 4):
        """
        Args:
            max_workers: 最大并行线程数
        """
        self.max_workers = max_workers

    def run_parallel(
        self,
        strategies: list[StrategyConfig],
        start: str,
        end: str,
        initial_capital: float = 100000,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[BacktestResult]:
        """
        并行运行多个策略回测

        Args:
            strategies: 策略配置列表
            start: 开始日期
            end: 结束日期
            initial_capital: 初始资金
            progress_callback: 进度回调 (completed, total, strategy_name)

        Returns:
            list[BacktestResult]: 回测结果列表 (按夏普比率降序)
        """
        from src.engine.backtest_engine import backtest_engine

        results: list[BacktestResult] = []
        total = len(strategies)
        completed = 0

        if total == 0:
            logger.warning("没有策略需要回测")
            return results

        logger.info(f"并行回测开始: {total} 个策略, {self.max_workers} 线程")

        def _run_single(strategy: StrategyConfig) -> BacktestResult:
            """运行单个策略回测 (在线程中执行)"""
            try:
                return backtest_engine.run_backtest(
                    strategy, start, end, initial_capital
                )
            except Exception as e:
                logger.error(f"策略 {strategy.name} 回测失败: {e}")
                return BacktestResult(
                    strategy_id=strategy.strategy_id,
                    period=(start, end),
                    initial_capital=initial_capital,
                )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_strategy = {
                executor.submit(_run_single, strat): strat
                for strat in strategies
            }

            # 收集结果
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"获取策略 {strategy.name} 结果失败: {e}")
                    results.append(BacktestResult(
                        strategy_id=strategy.strategy_id,
                        period=(start, end),
                    ))

                completed += 1
                if progress_callback:
                    try:
                        progress_callback(completed, total, strategy.name)
                    except Exception:
                        pass

                logger.info(f"进度: {completed}/{total} - {strategy.name} 完成")

        # 按夏普比率降序排序
        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)

        logger.info(f"并行回测完成: {len(results)} 个策略")
        if results:
            best = results[0]
            logger.info(f"最佳策略: {best.strategy_id} | "
                        f"年化 {best.annual_return:.1%} | "
                        f"夏普 {best.sharpe_ratio:.2f}")

        return results

    def run_parallel_simple(
        self,
        signals_list: list[tuple[str, list]],
        stock_pool: list[str],
        start: str,
        end: str,
        initial_capital: float = 100000,
    ) -> list[BacktestResult]:
        """
        并行运行简化回测 (给定信号列表)

        Args:
            signals_list: [(strategy_id, signals), ...]
            stock_pool: 股票池
            start: 开始日期
            end: 结束日期
            initial_capital: 初始资金
        """
        from src.engine.backtest_engine import backtest_engine

        results = []
        total = len(signals_list)

        def _run_simple(item):
            sid, signals = item
            try:
                return backtest_engine.run_simple_backtest(
                    signals, stock_pool, start, end, initial_capital, sid
                )
            except Exception as e:
                logger.error(f"简化回测 {sid} 失败: {e}")
                return BacktestResult(strategy_id=sid)

        logger.info(f"并行简化回测: {total} 个")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_run_simple, item) for item in signals_list]
            for f in as_completed(futures):
                results.append(f.result())

        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        return results


# 全局实例
parallel_backtester = ParallelBacktester()
