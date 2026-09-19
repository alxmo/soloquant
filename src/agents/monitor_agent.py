"""
Monitor Agent - 风险监控员
=====================================================
职责:
1. 实时监控持仓盈亏
2. 止损止盈检查
3. 风控告警 (7条规则)
4. 组合表现分析
5. 生成日报/周报

工具链:
- portfolio_monitor.check_once() -> 组合检查
- portfolio_monitor.get_portfolio_status() -> 状态查询
- risk_guard.check_portfolio() -> 组合风控
- position_manager.check_stop_loss() -> 止损检查
"""

from typing import Any
from datetime import datetime

from loguru import logger

from src.models import AssetType
from src.agents.base_agent import BaseAgent
from src.execution.alpaca_client import alpaca_client
from src.execution.position_manager import position_manager
from src.execution.risk_guard import risk_guard


class MonitorAgent(BaseAgent):
    """风险监控员 Agent"""

    def __init__(self):
        super().__init__(name="monitor", role="风险监控员")

    def _execute(
        self,
        mode: str = "snapshot",
        check_stop_loss: bool = True,
        **kwargs,
    ) -> dict:
        """
        执行监控任务

        Args:
            mode: "snapshot" (快照) | "report" (日报)
            check_stop_loss: 是否检查止损

        Returns:
            dict: {
                "account": dict,
                "positions": list[dict],
                "risk_status": dict,
                "alerts": list[str],
                "stop_loss_triggered": list[dict],
                "plain_summary": str,
            }
        """
        logger.info(f"📊 监控检查开始 | 模式: {mode}")

        # Phase 8: 加密货币监控模式路由
        if mode == "crypto_snapshot":
            return self._execute_crypto_snapshot()
        if mode == "crypto_monitor":
            return self._execute_crypto_monitor(check_stop_loss=check_stop_loss)

        # 1. 检查 Alpaca 是否可用
        if not alpaca_client.available:
            logger.warning("  Alpaca 客户端不可用，仅返回本地缓存状态")
            return self._offline_status()

        # 2. 获取账户和持仓
        account = alpaca_client.get_account()
        positions = alpaca_client.get_positions()

        if account is None:
            logger.error("  无法获取账户信息")
            return self._offline_status()

        # 3. 风控检查
        risk_status = risk_guard.check_portfolio(account, positions)
        alerts = risk_status.alerts if risk_status.is_alert else []

        # 4. 止损检查
        stop_loss_triggered = []
        if check_stop_loss:
            try:
                stop_signals = position_manager.check_stop_loss()
                for sig in stop_signals:
                    stop_loss_triggered.append({
                        "symbol": sig.symbol,
                        "direction": sig.direction.value,
                        "target_weight": sig.target_weight,
                        "reason": sig.reason,
                    })
                    alerts.append(f"止损触发: {sig.symbol} - {sig.reason}")
            except Exception as e:
                logger.debug(f"止损检查失败: {e}")

        # 5. 组装结果
        result = {
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "account": {
                "equity": account.equity,
                "cash": account.cash,
                "buying_power": account.buying_power,
                "trading_mode": account.trading_mode.value,
                "day_trade_count": account.day_trade_count,
            },
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_cost": p.avg_cost,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct,
                    "weight": p.weight,
                }
                for p in positions
            ],
            "risk_status": {
                "is_alert": risk_status.is_alert,
                "current_drawdown": risk_status.current_drawdown,
                "current_daily_loss": risk_status.current_daily_loss,
                "max_single_position": risk_status.max_single_position,
                "max_total_drawdown": risk_status.max_total_drawdown,
                "daily_loss_limit": risk_status.daily_loss_limit,
            },
            "alerts": alerts,
            "stop_loss_triggered": stop_loss_triggered,
        }

        # 6. 生成报告 (AI 增强: 优先使用 AI 监控分析)
        if mode == "report":
            result["plain_summary"] = self._build_daily_report(result)
        else:
            result["plain_summary"] = self._build_snapshot(result)

        # AI 分析增强
        from src.llm.ai_analyzer import ai_analyzer
        ai_analysis = ai_analyzer.analyze_monitor(
            equity=account.equity,
            cash=account.cash,
            positions=result.get("positions", []),
            risk_status=result.get("risk_status", {}),
            alerts=alerts,
            stop_loss_triggered=stop_loss_triggered,
            mode=mode,
        )
        if ai_analysis:
            result["plain_summary"] += f"\n\n🤖 AI 持仓分析:\n{'─' * 40}\n{ai_analysis}"

        logger.info(f"📊 监控完成 | 净值: ${account.equity:,.2f} | "
                     f"持仓: {len(positions)} | 告警: {len(alerts)}")

        return result

    # ═══════════════════════════════════════════════════════
    # Phase 8: 加密货币监控
    # ═══════════════════════════════════════════════════════

    def _execute_crypto_snapshot(self) -> dict:
        """
        加密货币组合风控快照

        使用 crypto_rules 进行风控检查,
        仅返回加密货币相关持仓状态。
        """
        logger.info("🔒 加密货币组合风控快照开始")

        if not alpaca_client.available:
            return self._offline_status()

        try:
            account = alpaca_client.get_account()
            if not account:
                return self._offline_status()

            # 获取所有持仓 (返回 list[Position]), 过滤出加密货币
            all_positions = alpaca_client.get_positions()
            crypto_positions = [
                p for p in all_positions
                if "/" in p.symbol or p.asset_type == AssetType.CRYPTO
            ]

            # 加密货币专属风控检查
            risk_status = risk_guard.check_crypto_portfolio(account, crypto_positions)
            alerts = risk_status.alerts if risk_status.is_alert else []

            result = {
                "mode": "crypto_snapshot",
                "timestamp": datetime.now().isoformat(),
                "account": {
                    "equity": account.equity,
                    "cash": account.cash,
                    "buying_power": account.buying_power,
                    "trading_mode": account.trading_mode.value,
                },
                "positions": [
                    {
                        "symbol": p.symbol,
                        "qty": p.qty,
                        "avg_cost": p.avg_cost,
                        "market_value": p.market_value,
                        "unrealized_pnl": p.unrealized_pnl,
                        "unrealized_pnl_pct": p.unrealized_pnl_pct,
                        "weight": p.weight,
                        "asset_type": "crypto",
                    }
                    for p in crypto_positions
                ],
                "risk_status": {
                    "is_alert": risk_status.is_alert,
                    "current_drawdown": risk_status.current_drawdown,
                    "current_daily_loss": risk_status.current_daily_loss,
                    "max_single_position": risk_status.max_single_position,
                },
                "alerts": alerts,
                "stop_loss_triggered": [],
            }

            result["plain_summary"] = self._build_crypto_snapshot(result)

            # AI 加密货币分析增强
            from src.llm.ai_analyzer import ai_analyzer
            ai_crypto = ai_analyzer.analyze_crypto_monitor(
                equity=account.equity if account else 0,
                cash=account.cash if account else 0,
                positions=result.get("positions", []),
                risk_status=result.get("risk_status", {}),
                alerts=alerts,
                stop_loss_triggered=[],
            )
            if ai_crypto:
                result["plain_summary"] += f"\n\n🤖 AI 加密货币分析:\n{'─' * 40}\n{ai_crypto}"

            logger.info(
                f"🔒 加密货币快照完成 | 净值:  | "
                f"加密货币持仓: {len(crypto_positions)} | 告警: {len(alerts)}"
            )
            return result

        except Exception as e:
            logger.error(f"加密货币快照失败: {e}")
            return self._offline_status()

    def _execute_crypto_monitor(self, check_stop_loss: bool = True) -> dict:
        """
        加密货币持仓监控 (含止损检查)

        使用 crypto_rules 的止损阈值 (10%, 比股票 5% 更宽松)。
        """
        logger.info("🔒 加密货币持仓监控开始")

        if not alpaca_client.available:
            return self._offline_status()

        try:
            account = alpaca_client.get_account()
            if not account:
                return self._offline_status()

            # 获取所有持仓 (返回 list[Position]), 过滤出加密货币
            all_positions = alpaca_client.get_positions()
            crypto_positions = [
                p for p in all_positions
                if "/" in p.symbol or p.asset_type == AssetType.CRYPTO
            ]

            # 加密货币风控检查
            risk_status = risk_guard.check_crypto_portfolio(account, crypto_positions)
            alerts = list(risk_status.alerts) if risk_status.is_alert else []

            # 加密货币止损检查 (使用 crypto_rules 的 10% 阈值)
            stop_loss_triggered = []
            if check_stop_loss:
                for pos in crypto_positions:
                    if risk_guard.should_crypto_stop_loss(pos):
                        stop_loss_triggered.append({
                            "symbol": pos.symbol,
                            "qty": pos.qty,
                            "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                            "reason": f"加密货币止损: 亏损 {pos.unrealized_pnl_pct:.1%} < -{risk_guard.crypto_rules['single_stop_loss']:.0%}",
                        })
                        alerts.append(
                            f"🔒 加密货币止损触发: {pos.symbol} "
                            f"(亏损 {pos.unrealized_pnl_pct:.1%})"
                        )

            result = {
                "mode": "crypto_monitor",
                "timestamp": datetime.now().isoformat(),
                "account": {
                    "equity": account.equity,
                    "cash": account.cash,
                    "trading_mode": account.trading_mode.value,
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
                    for p in crypto_positions
                ],
                "risk_status": {
                    "is_alert": risk_status.is_alert,
                    "current_drawdown": risk_status.current_drawdown,
                    "current_daily_loss": risk_status.current_daily_loss,
                },
                "alerts": alerts,
                "stop_loss_triggered": stop_loss_triggered,
            }

            result["plain_summary"] = self._build_crypto_monitor_report(result)

            # AI 加密货币监控分析增强
            from src.llm.ai_analyzer import ai_analyzer
            ai_crypto_monitor = ai_analyzer.analyze_crypto_monitor(
                equity=account.equity if account else 0,
                cash=account.cash if account else 0,
                positions=result.get("positions", []),
                risk_status=result.get("risk_status", {}),
                alerts=alerts,
                stop_loss_triggered=stop_loss_triggered,
            )
            if ai_crypto_monitor:
                result["plain_summary"] += f"\n\n🤖 AI 加密货币监控分析:\n{'─' * 40}\n{ai_crypto_monitor}"

            logger.info(
                f"🔒 加密货币监控完成 | 持仓: {len(crypto_positions)} | "
                f"止损: {len(stop_loss_triggered)} | 告警: {len(alerts)}"
            )
            return result

        except Exception as e:
            logger.error(f"加密货币监控失败: {e}")
            return self._offline_status()

    def _build_crypto_snapshot(self, result: dict) -> str:
        """构建加密货币快照报告"""
        lines = []
        lines.append(f"🔒 加密货币组合快照")
        lines.append(f"{'─' * 40}")

        acct = result.get("account")
        if acct:
            lines.append(f"💰 账户:")
            lines.append(f"  总资产: ")
            lines.append(f"  现金: ")
            mode_label = "📝 模拟盘" if acct["trading_mode"] == "paper" else "🔴 实盘"
            lines.append(f"  交易模式: {mode_label}")
            lines.append("")

        positions = result.get("positions", [])
        if positions:
            lines.append(f"📈 加密货币持仓 ({len(positions)} 个):")
            lines.append(f"  {'代码':<12} {'数量':>10} {'市值':>12} {'盈亏':>10} {'仓位':>8}")
            lines.append(f"  {'─' * 56}")
            for p in positions:
                pnl_icon = "🟢" if p["unrealized_pnl"] >= 0 else "🔴"
                lines.append(
                    f"  {p['symbol']:<12} {p['qty']:>10.6f} "
                    f" "
                    f"{pnl_icon} "
                    f"{p['weight']:>7.1%}"
                )
        else:
            lines.append("📭 当前无加密货币持仓")

        lines.append("")

        risk = result.get("risk_status", {})
        if risk:
            lines.append(f"🛡️ 加密货币风控:")
            dd = risk.get("current_drawdown", 0)
            lines.append(f"  当前回撤: {dd:.1%}")
            dl = risk.get("current_daily_loss", 0)
            lines.append(f"  日亏损: {dl:.1%}")
            if risk.get("is_alert"):
                lines.append(f"  ⚠️ 风控告警!")

        alerts = result.get("alerts", [])
        if alerts:
            lines.append("")
            lines.append("⚠️ 告警:")
            for a in alerts:
                lines.append(f"  • {a}")

        return "\n".join(lines)

    def _build_crypto_monitor_report(self, result: dict) -> str:
        """构建加密货币监控报告"""
        lines = []
        lines.append(f"🔒 加密货币持仓监控报告")
        lines.append(f"{'═' * 40}")

        positions = result.get("positions", [])
        if positions:
            lines.append(f"📈 持仓状态 ({len(positions)} 个):")
            for p in positions:
                pnl_icon = "🟢" if p["unrealized_pnl"] >= 0 else "🔴"
                lines.append(
                    f"  {pnl_icon} {p['symbol']}: "
                    f"{p['qty']:.6f} 个 | "
                    " | "
                    f"盈亏 {p['unrealized_pnl_pct']:+.1%}"
                )
        else:
            lines.append("📭 当前无加密货币持仓")

        lines.append("")

        # 止损
        stop_loss = result.get("stop_loss_triggered", [])
        if stop_loss:
            lines.append(f"🚨 止损触发 ({len(stop_loss)} 个):")
            for s in stop_loss:
                lines.append(f"  • {s['symbol']}: {s['reason']}")
        else:
            lines.append("✅ 无止损触发")

        # 告警
        alerts = result.get("alerts", [])
        if alerts:
            lines.append("")
            lines.append(f"⚠️ 告警 ({len(alerts)} 条):")
            for a in alerts:
                lines.append(f"  • {a}")

        lines.append("")
        lines.append("💡 加密货币 24/7 交易, 建议设置定时止损检查。")

        return "\n".join(lines)

    def _offline_status(self) -> dict:
        """Alpaca 不可用时的离线状态"""
        return {
            "mode": "offline",
            "timestamp": datetime.now().isoformat(),
            "account": None,
            "positions": [],
            "risk_status": {"is_alert": False},
            "alerts": ["Alpaca 客户端未配置，无法获取实时持仓数据"],
            "stop_loss_triggered": [],
            "plain_summary": (
                "📊 监控状态: 离线\n"
                "{'─' * 40}\n"
                "⚠️ Alpaca 客户端未配置，无法获取实时持仓数据。\n"
                "请配置 ALPACA_API_KEY 和 ALPACA_SECRET_KEY 后使用。\n"
                "当前仅可使用本地缓存的策略和回测功能。"
            ),
        }

    def _build_snapshot(self, result: dict) -> str:
        """构建快照报告"""
        lines = []
        lines.append(f"📊 组合监控快照")
        lines.append(f"{'─' * 40}")

        acct = result.get("account")
        if acct:
            lines.append(f"💰 账户:")
            lines.append(f"  总资产: ${acct['equity']:,.2f}")
            lines.append(f"  现金: ${acct['cash']:,.2f}")
            lines.append(f"  购买力: ${acct['buying_power']:,.2f}")
            mode_label = "📝 模拟盘" if acct["trading_mode"] == "paper" else "🔴 实盘"
            lines.append(f"  交易模式: {mode_label}")
            lines.append("")

        positions = result.get("positions", [])
        if positions:
            lines.append(f"📈 持仓明细 ({len(positions)} 只):")
            lines.append(f"  {'代码':<8} {'数量':>8} {'市值':>12} {'盈亏':>10} {'仓位':>8}")
            lines.append(f"  {'─' * 50}")
            for p in positions:
                pnl_icon = "🟢" if p["unrealized_pnl"] >= 0 else "🔴"
                lines.append(
                    f"  {p['symbol']:<8} {p['qty']:>8.2f} "
                    f"${p['market_value']:>10,.0f} "
                    f"{pnl_icon}${p['unrealized_pnl']:>+8,.0f} "
                    f"{p['weight']:>7.1%}"
                )
        else:
            lines.append("📭 当前无持仓")

        lines.append("")

        # 风控状态
        risk = result.get("risk_status", {})
        if risk:
            lines.append(f"🛡️ 风控状态:")
            dd = risk.get("current_drawdown", 0)
            dd_label = "正常" if dd < 0.05 else "注意" if dd < 0.10 else "警告"
            lines.append(f"  当前回撤: {dd:.1%} ({dd_label})")

            daily_loss = risk.get("current_daily_loss", 0)
            if daily_loss < 0:
                loss_label = "正常" if abs(daily_loss) < 0.02 else "注意" if abs(daily_loss) < 0.05 else "警告"
                lines.append(f"  今日亏损: {daily_loss:.1%} ({loss_label})")
            else:
                lines.append(f"  今日盈亏: {daily_loss:+.1%}")

            if risk.get("is_alert"):
                lines.append(f"  ⚠️ 风控告警: 有 {len(result.get('alerts', []))} 条告警")

        # 止损触发
        stop_losses = result.get("stop_loss_triggered", [])
        if stop_losses:
            lines.append("")
            lines.append(f"🚨 止损触发 ({len(stop_losses)} 只):")
            for sl in stop_losses:
                lines.append(f"  • {sl['symbol']}: {sl['reason']}")

        # 告警
        alerts = result.get("alerts", [])
        if alerts:
            lines.append("")
            lines.append(f"⚠️ 告警 ({len(alerts)} 条):")
            for a in alerts:
                lines.append(f"  • {a}")

        return "\n".join(lines)

    def _build_daily_report(self, result: dict) -> str:
        """构建日报"""
        lines = []
        lines.append(f"📊 交易日报 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"{'═' * 50}")

        # 快照部分
        lines.append(self._build_snapshot(result))

        lines.append("")
        lines.append(f"{'═' * 50}")
        lines.append("💡 监控建议:")

        # 根据状态给出建议
        risk = result.get("risk_status", {})
        dd = risk.get("current_drawdown", 0)
        alerts = result.get("alerts", [])

        if not alerts and dd < 0.05:
            lines.append("  ✅ 组合运行正常，无异常告警。")
            lines.append("  建议继续持有，关注市场变化。")
        elif dd > 0.10:
            lines.append("  ⚠️ 回撤较大，建议检查策略有效性。")
            lines.append("  考虑是否需要调整仓位或暂停策略。")
        elif alerts:
            lines.append(f"  ⚠️ 有 {len(alerts)} 条告警，请及时处理。")
        else:
            lines.append("  组合基本正常，保持关注即可。")

        lines.append("")
        lines.append("⚠️ 风险提示: 监控数据仅供参考，请结合自身情况做出决策。")

        return "\n".join(lines)

    def plain_explain(self, result: Any = None) -> str:
        """通俗解读"""
        if result is None:
            result = self._last_result
        if isinstance(result, dict) and "plain_summary" in result:
            return result["plain_summary"]
        return "监控完成，请查看报告。"
