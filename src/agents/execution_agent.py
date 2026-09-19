"""
Execution Agent - 交易执行员
=====================================================
职责:
1. 接收交易信号 (来自 Strategy Agent)
2. 逐条进行风控检查 (7条规则)
3. 执行通过风控的订单 (Paper Trading)
4. 记录执行结果与失败原因
5. 生成执行报告 (含小白解读)
6. 支持紧急清仓模式 (liquidate)

工具链:
- order_manager.execute_signal() -> 信号执行
- risk_guard.check_order() -> 风控检查
- position_manager.get_position_summary() -> 持仓查询
- position_manager.liquidate_all() -> 清仓所有持仓

v0.2.1 修复:
- P3: 新增 mode="liquidate" 紧急清仓模式，支持 emergency_stop 工作流
"""


from typing import Any
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.execution.order_manager import order_manager
from src.execution.position_manager import position_manager
from src.execution.alpaca_client import alpaca_client
from src.models import TradeSignal, Direction
from src.utils.config import config


class ExecutionAgent(BaseAgent):
    """交易执行员 Agent"""

    def __init__(self):
        super().__init__(name="execution", role="交易执行员")

    def _execute(
        self,
        signals: list[TradeSignal] = None,
        strategy_result: dict = None,
        aggregated_signals: list = None,
        auto_confirm: bool = True,
        mode: str = "normal",
        **kwargs,
    ) -> dict:
        """
        执行交易信号

        Args:
            signals: 交易信号列表 (优先)
            strategy_result: Strategy Agent 的输出 (从中提取 signals)
            aggregated_signals: 聚合信号列表 (AggregatedSignal, 多策略模式)
            auto_confirm: 是否自动确认 (False 时仅模拟不执行)
            mode: 执行模式
                  "normal"    - 正常信号执行 (默认)
                  "liquidate" - 紧急清仓，卖出所有持仓
                  "portfolio" - 多策略聚合信号执行
                  "crypto"    - 加密货币信号执行 (Phase 8 新增)

        Returns:
            dict: {
                "total_signals": int,
                "executed": int,
                "rejected": int,
                "skipped": int,
                "details": list[dict],
                "plain_summary": str,
            }
        """
        # P3 修复: 紧急清仓模式
        if mode == "liquidate":
            return self._execute_liquidate()

        # Phase 6: 多策略聚合信号执行
        if mode == "portfolio" or aggregated_signals:
            return self._execute_aggregated(aggregated_signals or [])

        # Phase 8: 加密货币信号执行
        if mode == "crypto":
            return self._execute_crypto(signals, strategy_result)

        # 1. 获取交易信号
        if signals is None and strategy_result:
            signals = strategy_result.get("signals", [])
        if signals is None:
            signals = []

        logger.info(f"💰 交易执行开始 | 信号数: {len(signals)}")
        logger.info(f"  交易模式: {'📝 模拟盘' if config.is_paper_trading() else '🔴 实盘'}")

        # 2. 检查 Alpaca 是否可用
        if not alpaca_client.available:
            logger.warning("  Alpaca 客户端不可用，仅模拟执行流程")
            return self._simulate_execution(signals)

        # 3. 逐条执行信号
        results = {
            "total_signals": len(signals),
            "executed": 0,
            "rejected": 0,
            "skipped": 0,
            "details": [],
        }

        for i, signal in enumerate(signals):
            logger.info(f"  [{i+1}/{len(signals)}] {signal.symbol} "
                        f"{signal.direction.value} (目标仓位: {signal.target_weight:.0%})")

            try:
                order = order_manager.execute_signal(signal)

                if order:
                    results["executed"] += 1
                    results["details"].append({
                        "symbol": signal.symbol,
                        "direction": signal.direction.value,
                        "target_weight": signal.target_weight,
                        "order_id": order.order_id,
                        "status": order.status.value,
                        "reason": signal.reason,
                    })
                    logger.info(f"    ✅ 已执行 (订单ID: {order.order_id})")
                else:
                    # 风控拒绝或下单失败
                    results["rejected"] += 1
                    results["details"].append({
                        "symbol": signal.symbol,
                        "direction": signal.direction.value,
                        "target_weight": signal.target_weight,
                        "order_id": None,
                        "status": "rejected",
                        "reason": signal.reason or "风控拒绝或下单失败",
                    })
                    logger.warning(f"    ❌ 被拒绝 (风控检查未通过或下单失败)")

            except Exception as e:
                results["rejected"] += 1
                results["details"].append({
                    "symbol": signal.symbol,
                    "direction": signal.direction.value,
                    "status": "error",
                    "reason": str(e),
                })
                logger.error(f"    ❌ 执行异常: {e}")

        # 4. 生成执行报告 (AI 增强: 优先使用 AI 执行审查)
        results["plain_summary"] = self._build_plain_summary(results)
        from src.llm.ai_analyzer import ai_analyzer
        ai_review = ai_analyzer.analyze_execution(
            total_signals=results["total_signals"],
            executed=results["executed"],
            rejected=results["rejected"],
            skipped=results.get("skipped", 0),
            details=results.get("details", []),
            trading_mode="paper" if config.is_paper_trading() else "live",
            simulated=results.get("simulated", False),
        )
        if ai_review:
            results["plain_summary"] += f"\n\n🤖 AI 执行审查:\n{'─' * 40}\n{ai_review}"

        logger.info(f"💰 交易执行完成 | 成功: {results['executed']} | "
                     f"拒绝: {results['rejected']} | 跳过: {results['skipped']}")

        return results

    def _execute_liquidate(self) -> dict:
        """
        P3 新增: 紧急清仓模式
        查询所有持仓，逐个卖出清仓
        """
        logger.warning("🚨 紧急清仓模式启动!")

        results = {
            "mode": "liquidate",
            "total_signals": 0,
            "executed": 0,
            "rejected": 0,
            "skipped": 0,
            "details": [],
        }

        # 检查 Alpaca 是否可用
        if not alpaca_client.available:
            logger.warning("  Alpaca 客户端不可用，无法执行清仓")
            results["plain_summary"] = (
                "🚨 紧急清仓 - 无法执行\n\n"
                "❌ Alpaca 客户端未配置或不可用，无法执行清仓操作。\n"
                "请检查 .env 中的 ALPACA_API_KEY 配置。"
            )
            return results

        # 获取所有持仓
        try:
            positions = position_manager.get_positions()
        except Exception as e:
            logger.error(f"获取持仓失败: {e}")
            results["plain_summary"] = f"🚨 紧急清仓 - 获取持仓失败: {e}"
            return results

        if not positions:
            logger.info("  当前无持仓，无需清仓")
            results["plain_summary"] = (
                "🚨 紧急清仓完成\n\n"
                "ℹ️ 当前无持仓，无需清仓。"
            )
            return results

        results["total_signals"] = len(positions)
        logger.info(f"  发现 {len(positions)} 个持仓，开始清仓...")

        # 逐个清仓
        for pos in positions:
            symbol = pos.get("symbol", "") if isinstance(pos, dict) else getattr(pos, "symbol", "")
            qty = pos.get("qty", 0) if isinstance(pos, dict) else getattr(pos, "qty", 0)

            if not symbol or qty <= 0:
                results["skipped"] += 1
                continue

            try:
                # 使用 position_manager.liquidate_all 中的单只清仓逻辑
                success = alpaca_client.liquidate(symbol)
                if success:
                    results["executed"] += 1
                    results["details"].append({
                        "symbol": symbol,
                        "direction": "sell",
                        "qty": qty,
                        "status": "liquidated",
                        "reason": "紧急清仓",
                    })
                    logger.info(f"    ✅ {symbol} 已清仓 (数量: {qty})")
                else:
                    results["rejected"] += 1
                    results["details"].append({
                        "symbol": symbol,
                        "direction": "sell",
                        "qty": qty,
                        "status": "failed",
                        "reason": "清仓下单失败",
                    })
                    logger.warning(f"    ❌ {symbol} 清仓失败")

            except Exception as e:
                results["rejected"] += 1
                results["details"].append({
                    "symbol": symbol,
                    "direction": "sell",
                    "status": "error",
                    "reason": str(e),
                })
                logger.error(f"    ❌ {symbol} 清仓异常: {e}")

        # 生成清仓报告
        results["plain_summary"] = self._build_liquidate_summary(results)

        logger.info(f"🚨 紧急清仓完成 | 成功: {results['executed']} | "
                     f"失败: {results['rejected']} | 跳过: {results['skipped']}")

        return results

    # ═══════════════════════════════════════════════════════
    # Phase 8: 加密货币信号执行
    # ═══════════════════════════════════════════════════════

    def _execute_crypto(
        self,
        signals: list[TradeSignal] = None,
        strategy_result: dict = None,
    ) -> dict:
        """
        执行加密货币交易信号

        使用 alpaca_client 的加密货币专用下单方法,
        走 crypto_rules 风控检查,
        订单 type 使用 GTC (Good Till Cancelled) 适配 24/7 交易。

        Args:
            signals: 加密货币交易信号列表 (优先)
            strategy_result: Strategy Agent 的输出 (从中提取 crypto_signals)

        Returns:
            dict: 执行结果 (与 normal 模式结构一致)
        """
        from src.models import AssetType

        # 1. 获取交易信号
        if signals is None and strategy_result:
            signals = strategy_result.get("crypto_signals", [])
            if not signals:
                signals = strategy_result.get("signals", [])
        if signals is None:
            signals = []

        logger.info(f"🔒 加密货币交易执行开始 | 信号数: {len(signals)}")

        results = {
            "mode": "crypto",
            "total_signals": len(signals),
            "executed": 0,
            "rejected": 0,
            "skipped": 0,
            "details": [],
        }

        # 检查加密货币交易是否启用
        if not config.has_crypto_enabled():
            logger.warning("  加密货币模块未启用，跳过执行")
            results["plain_summary"] = (
                "🔒 加密货币交易执行 - 未启用\n\n"
                "❌ 加密货币模块未启用 (CRYPTO_ENABLED=false)。\n"
                "请在 .env 中设置 CRYPTO_ENABLED=true 并重启系统。"
            )
            return results

        # 检查 Alpaca 是否可用
        if not alpaca_client.available:
            logger.warning("  Alpaca 客户端不可用，仅模拟执行流程")
            return self._simulate_crypto_execution(signals)

        # 2. 获取账户和持仓信息 (用于风控检查)
        from src.models import Account as AccountModel, TradingMode
        try:
            # alpaca_client.get_account() 返回 Account 对象, get_positions() 返回 list[Position]
            account = alpaca_client.get_account()
            if account is None:
                account = AccountModel(equity=100000, cash=100000, buying_power=100000,
                                       trading_mode=TradingMode.PAPER)

            # 获取现有持仓 (已经是 Position 对象列表)
            positions = alpaca_client.get_positions()
        except Exception as e:
            logger.error(f"获取账户/持仓信息失败: {e}")
            # 降级: 使用空账户和持仓
            account = AccountModel(equity=100000, cash=100000, buying_power=100000,
                                   trading_mode=TradingMode.PAPER)
            positions = []

        # 3. 逐条执行信号 (加密货币专属风控)
        from src.execution.risk_guard import risk_guard

        for i, signal in enumerate(signals):
            # 确保信号标记为 CRYPTO 类型
            if not hasattr(signal, "asset_type") or signal.asset_type != AssetType.CRYPTO:
                signal.asset_type = AssetType.CRYPTO

            logger.info(
                f"  [{i+1}/{len(signals)}] {signal.symbol} "
                f"{signal.direction.value} (目标仓位: {signal.target_weight:.0%})"
            )

            try:
                # 加密货币专属风控检查
                risk_result = risk_guard.check_crypto_order(signal, account, positions)
                if not risk_result.passed:
                    results["rejected"] += 1
                    results["details"].append({
                        "symbol": signal.symbol,
                        "direction": signal.direction.value,
                        "target_weight": signal.target_weight,
                        "order_id": None,
                        "status": "rejected",
                        "reason": risk_result.reason,
                    })
                    logger.warning(f"    🛡️ 加密货币风控拒绝: {risk_result.reason}")
                    continue

                # 通过风控, 执行加密货币下单
                # 计算下单数量 - 使用 data_collector 获取最新价格
                from src.data.collector import data_collector
                price = data_collector.get_crypto_latest_price(signal.symbol)
                if not price or price <= 0:
                    results["rejected"] += 1
                    results["details"].append({
                        "symbol": signal.symbol,
                        "direction": signal.direction.value,
                        "status": "rejected",
                        "reason": "无法获取最新价格",
                    })
                    logger.warning(f"    ❌ 无法获取 {signal.symbol} 最新价格")
                    continue

                if signal.direction == Direction.BUY:
                    # 买入: 按目标仓位计算金额
                    order_amount = account.equity * signal.target_weight
                    # 前置校验: Alpaca 要求最小订单金额 $10
                    if order_amount < 10.0:
                        results["skipped"] += 1
                        results["details"].append({
                            "symbol": signal.symbol,
                            "direction": "buy",
                            "target_weight": signal.target_weight,
                            "status": "skipped",
                            "reason": f"订单金额 ${order_amount:.2f} < $10 (Alpaca 最小限额)",
                        })
                        logger.warning(
                            f"    ⏭️ {signal.symbol} 买入金额 ${order_amount:.2f} < $10, 跳过"
                        )
                        continue
                    qty = order_amount / price
                    side = "buy"
                else:
                    # 卖出: 查找现有持仓数量
                    existing = next(
                        (p for p in positions if p.symbol == signal.symbol), None
                    )
                    if existing and existing.qty > 0:
                        qty = existing.qty
                        # 前置校验: 卖出金额也需 >= $10
                        sell_value = qty * price
                        if sell_value < 10.0:
                            results["skipped"] += 1
                            results["details"].append({
                                "symbol": signal.symbol,
                                "direction": "sell",
                                "status": "skipped",
                                "reason": f"卖出金额 ${sell_value:.2f} < $10 (Alpaca 最小限额)",
                            })
                            logger.warning(
                                f"    ⏭️ {signal.symbol} 卖出金额 ${sell_value:.2f} < $10, 跳过"
                            )
                            continue
                    else:
                        results["skipped"] += 1
                        results["details"].append({
                            "symbol": signal.symbol,
                            "direction": "sell",
                            "status": "skipped",
                            "reason": "无持仓可卖",
                        })
                        logger.info(f"    ⏭️ {signal.symbol} 无持仓, 跳过卖出")
                        continue
                    side = "sell"

                # 调用加密货币下单 (传入 price 做安全兜底校验)
                order = alpaca_client.submit_crypto_order(
                    symbol=signal.symbol,
                    side=side,
                    qty=qty,
                    price=price,
                )

                if order:
                    results["executed"] += 1
                    # 记录加密货币交易
                    risk_guard.record_crypto_trade()
                    results["details"].append({
                        "symbol": signal.symbol,
                        "direction": signal.direction.value,
                        "target_weight": signal.target_weight,
                        "qty": qty,
                        "price": price,
                        "order_id": getattr(order, "id", str(order)),
                        "status": "filled",
                        "reason": signal.reason,
                    })
                    logger.info(f"    ✅ 已执行 (数量: {qty:.6f}, 价格: )")
                else:
                    results["rejected"] += 1
                    results["details"].append({
                        "symbol": signal.symbol,
                        "direction": signal.direction.value,
                        "status": "rejected",
                        "reason": "下单失败",
                    })
                    logger.warning(f"    ❌ 下单失败")

            except Exception as e:
                results["rejected"] += 1
                results["details"].append({
                    "symbol": signal.symbol,
                    "direction": signal.direction.value,
                    "status": "error",
                    "reason": str(e),
                })
                logger.error(f"    ❌ 执行异常: {e}")

        # 4. 生成执行报告
        results["plain_summary"] = self._build_crypto_summary(results)

        logger.info(
            f"🔒 加密货币交易执行完成 | 成功: {results['executed']} | "
            f"拒绝: {results['rejected']} | 跳过: {results['skipped']}"
        )
        return results

    def _simulate_crypto_execution(self, signals: list[TradeSignal]) -> dict:
        """Alpaca 不可用时的加密货币模拟执行"""
        results = {
            "mode": "crypto",
            "total_signals": len(signals),
            "executed": 0,
            "rejected": 0,
            "skipped": len(signals),
            "details": [],
            "simulated": True,
        }

        for signal in signals:
            results["details"].append({
                "symbol": signal.symbol,
                "direction": signal.direction.value,
                "target_weight": signal.target_weight,
                "order_id": None,
                "status": "simulated",
                "reason": signal.reason,
            })

        results["plain_summary"] = self._build_crypto_summary(results)
        return results

    def _build_crypto_summary(self, results: dict) -> str:
        """构建加密货币交易执行报告"""
        lines = []
        lines.append(f"🔒 加密货币交易执行报告")
        lines.append(f"{'═' * 40}")

        simulated = results.get("simulated", False)
        total = results["total_signals"]
        executed = results["executed"]
        rejected = results["rejected"]
        skipped = results.get("skipped", 0)

        if simulated:
            lines.append(f"  ⚠️ Alpaca 未配置，以下为模拟执行")
            lines.append("")

        lines.append(f"  信号总数: {total}")
        lines.append(f"  ✅ 成功执行: {executed}")
        lines.append(f"  🛡️ 风控拒绝: {rejected}")
        lines.append(f"  ⏭️ 跳过: {skipped}")
        lines.append("")

        details = results.get("details", [])
        if details:
            lines.append("  📋 执行明细:")
            for d in details:
                sym = d.get("symbol", "?")
                direction = d.get("direction", "?")
                weight = d.get("target_weight", 0)
                status = d.get("status", "unknown")

                if status == "filled":
                    status_icon = "✅"
                    status_label = "已成交"
                elif status == "simulated":
                    status_icon = "📝"
                    status_label = "模拟"
                elif status == "rejected":
                    status_icon = "🛡️"
                    status_label = "拒绝"
                elif status == "skipped":
                    status_icon = "⏭️"
                    status_label = "跳过"
                else:
                    status_icon = "⏳"
                    status_label = status

                lines.append(f"  {status_icon} {sym}: {direction} {weight:.0%} -> {status_label}")
                if d.get("reason"):
                    lines.append(f"     理由: {d['reason']}")

        lines.append("")
        mode_label = "📝 模拟盘" if config.is_paper_trading() else "🔴 实盘"
        lines.append(f"交易模式: {mode_label}")
        lines.append("⚠️ 加密货币 24/7 交易, 请关注风控告警和止损通知。")

        return "\n".join(lines)

    def _execute_aggregated(self, aggregated_signals: list) -> dict:
        """
        Phase 6: 执行多策略聚合信号

        将 AggregatedSignal 转换为 TradeSignal 后执行，
        每个聚合信号会附带来源策略信息用于风控审计。

        Args:
            aggregated_signals: AggregatedSignal 列表

        Returns:
            dict: 执行结果 (与 normal 模式结构一致)
        """

        logger.info(f"💰 多策略聚合信号执行 | 信号数: {len(aggregated_signals)}")

        if not aggregated_signals:
            return {
                "mode": "portfolio",
                "total_signals": 0,
                "executed": 0,
                "rejected": 0,
                "skipped": 0,
                "details": [],
                "plain_summary": "⚠️ 无聚合信号可执行",
            }

        # 将 AggregatedSignal 转为 TradeSignal
        trade_signals = []
        for agg in aggregated_signals:
            signal = TradeSignal(
                signal_id=agg.signal_id,
                strategy_id=f"portfolio:{'+'.join(agg.source_strategies)}",
                symbol=agg.symbol,
                direction=agg.direction,
                target_weight=agg.target_weight,
                reason=f"聚合信号 (置信度 {agg.confidence:.0%}, 来源: {', '.join(agg.source_strategies)})",
                confidence=agg.confidence,
            )
            trade_signals.append(signal)

        # 复用正常执行逻辑
        results = {
            "mode": "portfolio",
            "total_signals": len(trade_signals),
            "executed": 0,
            "rejected": 0,
            "skipped": 0,
            "details": [],
        }

        # 检查 Alpaca 是否可用
        if not alpaca_client.available:
            logger.warning("  Alpaca 客户端不可用，仅模拟执行流程")
            sim_result = self._simulate_execution(trade_signals)
            sim_result["mode"] = "portfolio"
            return sim_result

        for i, signal in enumerate(trade_signals):
            agg = aggregated_signals[i]  # 对应的原始聚合信号
            logger.info(
                f"  [{i+1}/{len(trade_signals)}] {signal.symbol} "
                f"{signal.direction.value} (目标仓位: {signal.target_weight:.0%})"
            )

            try:
                order = order_manager.execute_signal(signal)
                if order:
                    results["executed"] += 1
                    results["details"].append({
                        "symbol": signal.symbol,
                        "direction": signal.direction.value,
                        "target_weight": signal.target_weight,
                        "order_id": order.order_id,
                        "status": order.status.value,
                        "source_strategies": agg.source_strategies,
                        "reason": signal.reason,
                    })
                    logger.info(f"    ✅ 已执行 (订单ID: {order.order_id})")
                else:
                    results["rejected"] += 1
                    results["details"].append({
                        "symbol": signal.symbol,
                        "direction": signal.direction.value,
                        "target_weight": signal.target_weight,
                        "order_id": None,
                        "status": "rejected",
                        "reason": signal.reason or "风控拒绝或下单失败",
                    })
                    logger.warning(f"    ❌ 被拒绝")
            except Exception as e:
                results["rejected"] += 1
                results["details"].append({
                    "symbol": signal.symbol,
                    "direction": signal.direction.value,
                    "status": "error",
                    "reason": str(e),
                })
                logger.error(f"    ❌ 执行异常: {e}")

        results["plain_summary"] = self._build_portfolio_summary(results)
        logger.info(
            f"💰 多策略执行完成 | 成功: {results['executed']} | "
            f"拒绝: {results['rejected']}"
        )
        return results

    def _build_portfolio_summary(self, results: dict) -> str:
        """构建多策略执行报告"""
        lines = []
        lines.append(f"💰 多策略组合执行报告")
        lines.append(f"{'═' * 40}")

        total = results["total_signals"]
        executed = results["executed"]
        rejected = results["rejected"]
        skipped = results.get("skipped", 0)

        lines.append(f"  聚合信号数: {total}")
        lines.append(f"  ✅ 成功执行: {executed}")
        lines.append(f"  🛡️ 风控拒绝: {rejected}")
        lines.append(f"  ⏭️ 跳过: {skipped}")
        lines.append("")

        details = results.get("details", [])
        if details:
            lines.append("  📋 执行明细:")
            for d in details:
                sym = d.get("symbol", "?")
                direction = d.get("direction", "?")
                weight = d.get("target_weight", 0)
                status = d.get("status", "unknown")
                sources = d.get("source_strategies", [])

                if status in ("filled",):
                    status_icon = "✅"
                    status_label = "已成交"
                elif status == "simulated":
                    status_icon = "📝"
                    status_label = "模拟"
                elif status == "rejected":
                    status_icon = "🛡️"
                    status_label = "风控拒绝"
                else:
                    status_icon = "⏳"
                    status_label = status

                source_str = f" (来源: {', '.join(sources)})" if sources else ""
                lines.append(
                    f"  {status_icon} {sym}: {direction} {weight:.0%} -> {status_label}{source_str}"
                )
                if d.get("reason"):
                    lines.append(f"     理由: {d['reason']}")

        lines.append("")
        mode_label = "📝 模拟盘" if config.is_paper_trading() else "🔴 实盘"
        lines.append(f"交易模式: {mode_label}")
        lines.append("⚠️ 多策略聚合信号由多个策略加权投票产生，请关注组合级风控。")

        return "\n".join(lines)

    def _simulate_execution(self, signals: list[TradeSignal]) -> dict:
        """Alpaca 不可用时的模拟执行"""
        results = {
            "total_signals": len(signals),
            "executed": 0,
            "rejected": 0,
            "skipped": len(signals),
            "details": [],
            "simulated": True,
        }

        for signal in signals:
            results["details"].append({
                "symbol": signal.symbol,
                "direction": signal.direction.value,
                "target_weight": signal.target_weight,
                "order_id": None,
                "status": "simulated",
                "reason": signal.reason,
            })

        results["plain_summary"] = self._build_plain_summary(results)
        return results

    def _build_plain_summary(self, results: dict) -> str:
        """构建小白友好的执行报告"""
        lines = []
        lines.append(f"💰 交易执行报告")
        lines.append(f"{'─' * 40}")

        simulated = results.get("simulated", False)

        # 统计
        total = results["total_signals"]
        executed = results["executed"]
        rejected = results["rejected"]
        skipped = results["skipped"]

        if simulated:
            lines.append(f"  ⚠️ Alpaca 未配置，以下为模拟执行")
            lines.append("")

        lines.append(f"  信号总数: {total}")
        lines.append(f"  ✅ 成功执行: {executed}")
        lines.append(f"  🛡️ 风控拒绝: {rejected}")
        lines.append(f"  ⏭️ 跳过: {skipped}")
        lines.append("")

        # 详情
        details = results.get("details", [])
        if details:
            lines.append("  📋 执行明细:")
            for d in details:
                sym = d.get("symbol", "?")
                direction = d.get("direction", "?")
                weight = d.get("target_weight", 0)
                status = d.get("status", "unknown")

                if status in ("filled",):
                    status_icon = "✅"
                    status_label = "已成交"
                elif status == "simulated":
                    status_icon = "📝"
                    status_label = "模拟"
                elif status == "rejected":
                    status_icon = "🛡️"
                    status_label = "风控拒绝"
                else:
                    status_icon = "⏳"
                    status_label = status

                lines.append(f"  {status_icon} {sym}: {direction} {weight} -> {status_label}")
                if d.get("reason"):
                    lines.append(f"     理由: {d['reason']}")

        lines.append("")

        # 持仓概览 (如果 Alpaca 可用)
        if not simulated and alpaca_client.available:
            try:
                summary = position_manager.get_position_summary()
                if "error" not in summary:
                    lines.append("📈 当前持仓概览:")
                    lines.append(f"  总资产: ${summary['total_equity']:,.2f}")
                    lines.append(f"  现金: ${summary['cash']:,.2f}")
                    lines.append(f"  持仓数: {summary['position_count']}")
                    pnl = summary.get("total_unrealized_pnl", 0)
                    pnl_icon = "📈" if pnl >= 0 else "📉"
                    lines.append(f"  浮动盈亏: {pnl_icon} ${pnl:,.2f}")
            except Exception as e:
                logger.debug(f"获取持仓概览失败: {e}")

        lines.append("")
        mode_label = "📝 模拟盘" if config.is_paper_trading() else "🔴 实盘"
        lines.append(f"交易模式: {mode_label}")
        lines.append("⚠️ 风险提示: 所有交易决策由 AI 策略生成，请关注风控告警。")

        return "\n".join(lines)

    def _build_liquidate_summary(self, results: dict) -> str:
        """构建紧急清仓报告"""
        lines = []
        lines.append(f"🚨 紧急清仓报告")
        lines.append(f"{'═' * 40}")

        total = results["total_signals"]
        executed = results["executed"]
        rejected = results["rejected"]
        skipped = results["skipped"]

        lines.append(f"  持仓数量: {total}")
        lines.append(f"  ✅ 清仓成功: {executed}")
        lines.append(f"  ❌ 清仓失败: {rejected}")
        lines.append(f"  ⏭️ 跳过: {skipped}")
        lines.append("")

        # 详情
        details = results.get("details", [])
        if details:
            lines.append("  📋 清仓明细:")
            for d in details:
                sym = d.get("symbol", "?")
                qty = d.get("qty", 0)
                status = d.get("status", "unknown")

                if status == "liquidated":
                    status_icon = "✅"
                    status_label = "已清仓"
                elif status == "failed":
                    status_icon = "❌"
                    status_label = "失败"
                else:
                    status_icon = "⏳"
                    status_label = status

                lines.append(f"  {status_icon} {sym} (数量: {qty}) -> {status_label}")
                if d.get("reason"):
                    lines.append(f"     理由: {d['reason']}")

        lines.append("")

        if executed == total and rejected == 0:
            lines.append("✅ 所有持仓已全部清仓完毕。")
        elif executed > 0:
            lines.append(f"⚠️ 部分清仓完成 ({executed}/{total})，请检查失败原因。")
        else:
            lines.append("❌ 清仓全部失败，请手动处理持仓！")

        lines.append("")
        lines.append("⚠️ 风险提示: 紧急清仓是最后的保护措施，请确认市场情况后再做后续决策。")

        return "\n".join(lines)

    def plain_explain(self, result: Any = None) -> str:
        """通俗解读"""
        if result is None:
            result = self._last_result
        if isinstance(result, dict) and "plain_summary" in result:
            return result["plain_summary"]
        return "交易执行完成，请查看报告。"
