"""
Strategy Agent - 量化策略工程师
=====================================================
职责:
1. 接收候选股票池 (来自 Research Agent)
2. 自动因子搜索与筛选
3. 多模型竞赛训练 (LightGBM / Linear)
4. 超参数优化
5. 生成交易信号 (买入/卖出/持有)

工具链:
- factor_searcher.search() -> 因子搜索
- model_manager.train_model() -> 模型训练
- model_manager.auto_optimize() -> 超参优化
- signal_generator.generate_signals() -> 信号生成
- strategy_manager.create_strategy() -> 策略创建
"""

from typing import Any
from datetime import datetime

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.engine.factor_search import factor_searcher
from src.engine.model_manager import model_manager
from src.engine.signal_generator import signal_generator
from src.engine.strategy_manager import strategy_manager
from src.models import StrategyConfig, TradeSignal
from src.utils.config import config


class StrategyAgent(BaseAgent):
    """量化策略师 Agent"""

    def __init__(self):
        super().__init__(name="strategy", role="量化策略师")

    def _execute(
        self,
        candidate_stocks: list = None,
        stock_pool: list[str] = None,
        model_type: str = "lightgbm",
        optimize: bool = True,
        mode: str = "single",
        portfolio_id: str = "",
        **kwargs,
    ) -> dict:
        """
        执行策略生成

        Args:
            candidate_stocks: 来自 Research Agent 的候选股票列表
            stock_pool: 直接指定股票池 (优先级高于 candidate_stocks)
            model_type: 模型类型 "lightgbm" | "linear"
            optimize: 是否自动优化超参数
            mode: 执行模式
                  "single"        - 单策略生成 (默认)
                  "get_active"   - 获取所有活跃策略
                  "generate_all" - 为所有活跃策略生成信号
                  "aggregate"    - 聚合多策略信号 (需配合 portfolio_id)
                  "crypto_signals" - 加密货币信号生成 (Phase 8 新增)
            portfolio_id: 组合ID (aggregate 模式需要)

        Returns:
            dict: {
                "strategy": StrategyConfig,
                "factors": list[FactorResult],
                "model_result": ModelResult,
                "signals": list[TradeSignal],
                "plain_summary": str,
            }
            (多策略模式时返回不同的结构)
        """
        # ── 多策略模式分发 ──
        if mode == "get_active":
            return self._mode_get_active_strategies()
        elif mode == "generate_all":
            return self._mode_generate_all_signals()
        elif mode == "aggregate":
            return self._mode_aggregate_signals(portfolio_id)
        elif mode == "crypto_signals":
            return self._mode_crypto_signals(candidate_stocks, stock_pool)

        # ── 单策略模式 (原有逻辑) ──
        # 1. 确定股票池
        if stock_pool is None:
            if candidate_stocks:
                # 从 StockCandidate 对象提取 symbol
                stock_pool = []
                for c in candidate_stocks:
                    sym = c.symbol if hasattr(c, "symbol") else str(c)
                    stock_pool.append(sym)
            else:
                # 使用配置的关注列表
                stock_pool = config.WATCHLIST if config.WATCHLIST else config.DEFAULT_WATCHLIST

        logger.info(f"🧠 策略生成开始 | 股票池: {stock_pool} | 模型: {model_type}")

        # 2. 因子搜索
        factors = self._search_factors(stock_pool)
        logger.info(f"  有效因子: {len(factors)} 个")

        # 3. 模型训练
        model_result = self._train_model(model_type, factors, stock_pool)
        if model_result is None:
            logger.warning("  模型训练失败，使用默认策略")
            model_result = self._fallback_model(model_type)

        # 4. 超参数优化 (可选)
        if optimize and model_result:
            optimized_params = self._optimize_params(model_type, factors, stock_pool)
            if optimized_params:
                model_result.params.update(optimized_params)
                logger.info(f"  超参优化完成: {optimized_params}")

        # 5. 创建策略配置
        strategy = self._create_strategy(
            stock_pool, model_type, factors, model_result
        )

        # 6. 生成交易信号
        signals = self._generate_signals(strategy, model_result, stock_pool)
        logger.info(f"  交易信号: {len(signals)} 条")

        # 7. 生成小白解读 (AI 增强: 优先使用 AI 策略分析报告)
        plain_summary = self._build_plain_summary(
            strategy, factors, model_result, signals
        )
        from src.llm.ai_analyzer import ai_analyzer
        ai_report = ai_analyzer.analyze_strategy(
            strategy.name, strategy.model_type, strategy.stock_pool,
            factors, model_result, signals,
        )
        if ai_report:
            plain_summary += f"\n\n🤖 AI 策略分析:\n{'─' * 40}\n{ai_report}"

        result = {
            "strategy": strategy,
            "factors": factors,
            "model_result": model_result,
            "signals": signals,
            "plain_summary": plain_summary,
        }

        logger.info("🧠 策略生成完成")
        return result

    def _search_factors(self, stock_pool: list[str]) -> list:
        """因子搜索"""
        try:
            # 使用最近1年数据搜索因子
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now().replace(year=datetime.now().year - 1)).strftime("%Y-%m-%d")

            factors = factor_searcher.search(
                stock_pool=stock_pool,
                start=start,
                end=end,
                top_n=10,
            )
            return factors
        except Exception as e:
            logger.warning(f"因子搜索失败: {e}")
            return []

    def _train_model(self, model_type: str, factors: list, stock_pool: list[str]):
        """模型训练"""
        try:
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now().replace(year=datetime.now().year - 1)).strftime("%Y-%m-%d")

            factor_names = [f.name if hasattr(f, "name") else str(f) for f in factors]
            if not factor_names:
                factor_names = ["RSI_14", "MA5_MA20", "VOL_RATIO",
                                "MOMENTUM_5", "VOLATILITY_20"]

            result = model_manager.train_model(
                model_type=model_type,
                factors=factor_names,
                stock_pool=stock_pool,
                train_start=start,
                train_end=end,
            )
            return result
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            return None

    def _optimize_params(self, model_type: str, factors: list, stock_pool: list[str]) -> dict:
        """超参数优化"""
        try:
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now().replace(year=datetime.now().year - 1)).strftime("%Y-%m-%d")

            factor_names = [f.name if hasattr(f, "name") else str(f) for f in factors]
            if not factor_names:
                factor_names = ["RSI_14", "MA5_MA20", "VOL_RATIO"]

            base_params = model_manager.SUPPORTED_MODELS.get(
                model_type, {}
            ).get("default_params", {}).copy()

            best_params = model_manager.auto_optimize(
                model_type=model_type,
                base_params=base_params,
                factors=factor_names,
                stock_pool=stock_pool,
                train_start=start,
                train_end=end,
            )
            return best_params
        except Exception as e:
            logger.warning(f"超参优化失败: {e}")
            return {}

    def _create_strategy(self, stock_pool, model_type, factors, model_result) -> StrategyConfig:
        """创建策略配置"""
        factor_names = [f.name if hasattr(f, "name") else str(f) for f in factors]
        if not factor_names:
            factor_names = ["RSI_14", "MA5_MA20", "VOL_RATIO",
                            "MOMENTUM_5", "VOLATILITY_20"]

        params = model_result.params if model_result else {}

        strategy = strategy_manager.create_strategy(
            strategy_id=f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=f"自动策略_{model_type}",
            model_type=model_type,
            factors=factor_names,
            stock_pool=stock_pool,
            hyper_params=params,
        )
        return strategy

    def _generate_signals(self, strategy, model_result, stock_pool) -> list[TradeSignal]:
        """生成交易信号"""
        try:
            model_path = model_result.model_path if model_result else ""
            if not model_path:
                logger.warning("无模型路径，跳过信号生成")
                return []

            signals = signal_generator.generate_signals(
                strategy=strategy,
                model_path=model_path,
                stock_pool=stock_pool,
                top_k=config.TOP_K_SIGNALS,
            )
            return signals
        except Exception as e:
            logger.warning(f"信号生成失败: {e}")
            return []

    def _fallback_model(self, model_type: str):
        """模型训练失败时的降级策略"""
        from src.models import ModelResult
        return ModelResult(
            model_type=model_type,
            train_ic=0.0,
            valid_ic=0.0,
            train_rank_ic=0.0,
            valid_rank_ic=0.0,
            model_path="",
            params={},
        )

    def _mode_get_active_strategies(self) -> dict:
        """多策略模式: 获取所有活跃策略"""
        active = strategy_manager.get_active_strategies()
        logger.info(f"📊 获取活跃策略: {len(active)} 个")

        summaries = strategy_manager.get_summary()
        active_summaries = [s for s in summaries if s["status"] == "active"]

        return {
            "mode": "get_active",
            "active_count": len(active),
            "strategies": active_summaries,
            "plain_summary": (
                f"📊 活跃策略列表\n{'─' * 40}\n"
                f"当前共有 {len(active)} 个活跃策略:\n" +
                "\n".join(
                    f"  • {s['strategy_id']}: {s['name']} "
                    f"({s['model_type']}, 评级: {s['backtest_grade']}, "
                    f"年化: {s['annual_return']:.1%})"
                    for s in active_summaries
                ) if active_summaries else "  (无活跃策略)"
            ),
        }

    def _mode_generate_all_signals(self) -> dict:
        """
        多策略模式: 为所有活跃策略生成交易信号

        对每个活跃策略:
        1. 使用该策略的因子和股票池
        2. 训练模型 (或加载已有模型)
        3. 生成交易信号
        4. 记录回测绩效 (用于后续权重分配)
        """
        from src.engine.portfolio_manager import portfolio_manager

        active_strategies = strategy_manager.get_active_strategies()
        logger.info(f"📊 多策略信号生成 | 策略数: {len(active_strategies)}")

        if not active_strategies:
            return {
                "mode": "generate_all",
                "signals_by_strategy": {},
                "total_signals": 0,
                "plain_summary": "⚠️ 无活跃策略，无法生成信号",
            }

        signals_by_strategy: dict[str, list[TradeSignal]] = {}
        total_signals = 0

        for strat in active_strategies:
            try:
                logger.info(f"  处理策略: {strat.name} ({strat.strategy_id})")

                # 训练模型
                model_result = self._train_model(
                    strat.model_type, strat.factors, strat.stock_pool
                )
                if model_result is None:
                    model_result = self._fallback_model(strat.model_type)

                # 生成信号
                signals = self._generate_signals(strat, model_result, strat.stock_pool)
                signals_by_strategy[strat.strategy_id] = signals
                total_signals += len(signals)

                # 更新组合管理器中的策略绩效 (用于风险平价/绩效加权)
                bt_result = strategy_manager.get_backtest_result(strat.strategy_id)
                if bt_result:
                    portfolio_manager.update_strategy_performance(
                        strat.strategy_id, bt_result
                    )

                logger.info(f"    信号数: {len(signals)}")

            except Exception as e:
                logger.error(f"  策略 {strat.strategy_id} 信号生成失败: {e}")
                signals_by_strategy[strat.strategy_id] = []

        return {
            "mode": "generate_all",
            "signals_by_strategy": signals_by_strategy,
            "total_signals": total_signals,
            "strategy_count": len(active_strategies),
            "plain_summary": (
                f"📊 多策略信号生成完成\n{'─' * 40}\n"
                f"策略数: {len(active_strategies)}\n"
                f"总信号数: {total_signals}\n" +
                "\n".join(
                    f"  • {sid}: {len(sigs)} 条信号"
                    for sid, sigs in signals_by_strategy.items()
                )
            ),
        }

    # ═══════════════════════════════════════════════════════
    # Phase 8: 加密货币信号生成模式
    # ═══════════════════════════════════════════════════════

    def _mode_crypto_signals(
        self,
        candidate_stocks: list = None,
        stock_pool: list = None,
    ) -> dict:
        """
        加密货币信号生成模式

        优先使用动量信号 (不依赖模型训练, 适配 24/7 交易)。
        如果候选列表来自 Research Agent 的 CryptoCandidate, 会提取 symbol。

        Args:
            candidate_stocks: 来自 Research Agent 的候选列表 (CryptoCandidate 或 str)
            stock_pool: 直接指定交易对列表 (优先级更高)

        Returns:
            dict: {
                "crypto_signals": list[TradeSignal],
                "signals": list[TradeSignal],  # 兼容字段
                "plain_summary": str,
            }
        """
        from src.engine.crypto_signal_generator import crypto_signal_generator

        # 1. 确定交易对列表
        if stock_pool:
            symbols = list(stock_pool)
        elif candidate_stocks:
            symbols = []
            for c in candidate_stocks:
                sym = c.symbol if hasattr(c, "symbol") else str(c)
                symbols.append(sym)
        else:
            symbols = config.get_crypto_pairs()

        logger.info(f"🔒 加密货币信号生成开始 | 交易对: {symbols}")

        # 2. 生成动量信号 (24h 回看)
        lookback_hours = 24
        top_k = min(config.TOP_K_SIGNALS, len(symbols))

        try:
            signals = crypto_signal_generator.generate_momentum_signals(
                symbols=symbols,
                lookback_hours=lookback_hours,
                top_k=top_k,
            )
        except Exception as e:
            logger.error(f"加密货币动量信号生成失败: {e}")
            signals = []

        logger.info(f"  加密货币信号: {len(signals)} 条")

        # 3. 生成报告
        if signals:
            lines = [f"🔒 加密货币信号生成报告", f"{'─' * 40}"]
            lines.append(f"  扫描交易对: {len(symbols)} 个")
            lines.append(f"  回看周期: {lookback_hours}h")
            lines.append(f"  生成信号: {len(signals)} 条")
            lines.append("")
            lines.append("  📋 信号明细:")
            for sig in signals:
                icon = "🟢" if sig.direction.value == "buy" else "🔴"
                lines.append(
                    f"  {icon} {sig.symbol}: {sig.direction.value} "
                    f"{sig.target_weight:.0%} | {sig.reason[:50]}"
                )
            lines.append("")
            lines.append("⚠️ 加密货币 24/7 交易, 请关注风控告警。")
        else:
            lines = [
                f"🔒 加密货币信号生成报告",
                f"{'─' * 40}",
                f"  扫描交易对: {len(symbols)} 个",
                f"  生成信号: 0 条",
                "",
                "⚠️ 未生成有效信号, 可能原因:",
                "  • 数据源不可用 (Alpaca 连接异常)",
                "  • K线数据不足 (新上线币种)",
                "  • 市场波动过小 (未达信号阈值)",
            ]

        result = {
            "crypto_signals": signals,
            "signals": signals,  # 兼容字段, 供 Execution Agent 读取
            "plain_summary": "\n".join(lines),
        }

        logger.info("🔒 加密货币信号生成完成")
        return result

    def _mode_aggregate_signals(self, portfolio_id: str = "") -> dict:
        """
        多策略模式: 聚合多策略信号

        1. 获取或创建默认组合
        2. 收集各策略信号 (从 kwargs 或重新生成)
        3. 使用组合管理器聚合信号
        4. 返回聚合后的 AggregatedSignal 列表
        """
        from src.engine.portfolio_manager import portfolio_manager

        logger.info(f"📊 多策略信号聚合 | 组合: {portfolio_id or '默认'}")

        # 获取组合
        portfolio = None
        if portfolio_id:
            portfolio = portfolio_manager.get_portfolio(portfolio_id)

        if portfolio is None:
            # 没有指定组合或组合不存在 -> 创建默认等权组合
            active_strategies = strategy_manager.get_active_strategies()
            if not active_strategies:
                return {
                    "mode": "aggregate",
                    "aggregated_signals": [],
                    "plain_summary": "⚠️ 无活跃策略，无法聚合信号",
                }
            strategy_ids = [s.strategy_id for s in active_strategies]
            portfolio = portfolio_manager.create_portfolio(
                name="默认组合",
                strategy_ids=strategy_ids,
            )
            logger.info(f"  创建默认组合: {portfolio.portfolio_id}")

        # 主动生成各策略信号 (工作流步骤间不共享结果，这里重新生成)
        gen_result = self._mode_generate_all_signals()
        signals_by_strategy = gen_result.get("signals_by_strategy", {})

        if not signals_by_strategy:
            return {
                "mode": "aggregate",
                "aggregated_signals": [],
                "plain_summary": "⚠️ 无可用信号，无法聚合",
            }

        # 执行组合再平衡: 重新分配权重 + 聚合信号
        aggregated_signals = portfolio_manager.rebalance(
            portfolio, signals_by_strategy
        )

        logger.info(f"  聚合信号数: {len(aggregated_signals)}")

        # 构建摘要
        signal_lines = []
        for sig in aggregated_signals[:10]:
            direction_label = "买入" if sig.direction.value == "buy" else "卖出"
            strategies_str = ", ".join(sig.source_strategies)
            signal_lines.append(
                f"  • {sig.symbol}: {direction_label} "
                f"(权重 {sig.target_weight:.0%}, 置信度 {sig.confidence:.0%}, "
                f"来源: {strategies_str})"
            )

        return {
            "mode": "aggregate",
            "portfolio_id": portfolio.portfolio_id,
            "aggregated_signals": aggregated_signals,
            "signal_count": len(aggregated_signals),
            "strategy_weights": portfolio.strategy_allocations,
            "plain_summary": (
                f"📊 多策略信号聚合完成\n{'─' * 40}\n"
                f"组合: {portfolio.name} ({portfolio.portfolio_id})\n"
                f"分配方式: {portfolio.allocation_method.value}\n"
                f"聚合信号: {len(aggregated_signals)} 条\n\n"
                f"信号明细:\n" +
                "\n".join(signal_lines) if signal_lines else "  (无聚合信号)"
            ),
        }

    def _build_plain_summary(self, strategy, factors, model_result, signals) -> str:
        """构建小白友好的策略总结"""
        lines = []
        lines.append(f"🧠 策略生成报告")
        lines.append(f"{'─' * 40}")
        lines.append(f"策略名称: {strategy.name}")
        lines.append(f"模型类型: {strategy.model_type}")
        lines.append(f"股票池: {', '.join(strategy.stock_pool)}")
        lines.append("")

        # 因子解读
        if factors:
            lines.append(f"📊 发现 {len(factors)} 个有效因子:")
            for f in factors[:5]:
                if hasattr(f, "name"):
                    ic_str = f"{f.ic:.3f}" if hasattr(f, "ic") else "N/A"
                    lines.append(f"  • {f.name} (IC={ic_str})")
            lines.append("  (IC 越高说明这个因子预测能力越强)")
        else:
            lines.append("⚠️ 未找到强有效因子，使用默认技术指标")

        lines.append("")

        # 模型解读
        if model_result:
            ic = model_result.valid_ic
            if ic > 0.05:
                ic_label = "预测能力较强"
            elif ic > 0.02:
                ic_label = "有一定预测能力"
            else:
                ic_label = "预测能力一般"
            lines.append(f"🤖 模型验证 IC: {ic:.4f} ({ic_label})")

        lines.append("")

        # 信号解读
        if signals:
            lines.append(f"📡 生成 {len(signals)} 条交易信号:")
            for s in signals[:5]:
                direction_label = "买入" if s.direction.value == "buy" else "卖出"
                weight_pct = f"{s.target_weight:.0%}"
                lines.append(f"  • {s.symbol}: {direction_label} (目标仓位 {weight_pct})")
                if s.reason:
                    lines.append(f"    理由: {s.reason}")
        else:
            lines.append("⚠️ 暂无交易信号 (可能模型未训练成功)")

        lines.append("")
        lines.append("💡 下一步: 建议对策略进行回测验证，确认稳健性后再执行。")
        lines.append("⚠️ 风险提示: 量化策略存在过拟合风险，历史表现不代表未来。")

        return "\n".join(lines)

    def plain_explain(self, result: Any = None) -> str:
        """通俗解读"""
        if result is None:
            result = self._last_result
        if isinstance(result, dict) and "plain_summary" in result:
            return result["plain_summary"]
        return "策略生成完成，请查看报告。"
