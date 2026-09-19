"""
风控引擎 - 小白安全网
=====================================================
T3.3: 风控引擎 (7 条规则)
R1: 单票最大仓位 (默认 30%)
R2: 最大持仓数 (默认 10 只)
R3: 日最大亏损 (默认 5%)
R4: 总最大回撤 (默认 15%)
R5: 单票止损 (默认 -5%)
R6: 日交易次数上限 (默认 20)
R7: Paper Trading 最少天数 (默认 30)

稳定性加固:
- 风控状态 (日交易次数/峰值权益/Paper起始日) 持久化到 SQLite
- 系统重启后自动恢复风控状态
"""

from datetime import datetime
from typing import Optional

from loguru import logger

from src.models import (
    Account, AssetType, Direction, Position, RiskCheckItem,
    RiskCheckResult, RiskStatus, TradeSignal,
)
from src.utils.config import config
from src.utils.notifier import notifier
from src.utils.risk_state_db import risk_state_db


class RiskGuard:
    """风控引擎"""

    def __init__(self):
        # ── 股票风控规则 ──
        self.rules = {
            "max_single_position": config.MAX_POSITION_PCT,
            "max_total_positions": config.MAX_TOTAL_POSITIONS,
            "max_daily_loss": config.MAX_DAILY_LOSS,
            "max_drawdown": config.MAX_DRAWDOWN,
            "single_stop_loss": config.SINGLE_STOP_LOSS,
            "daily_trade_limit": config.DAILY_TRADE_LIMIT,
            "min_paper_days": config.MIN_PAPER_DAYS,
        }
        # 实盘模式使用更严格的风控参数 (Phase 6 新增)
        if not config.is_paper_trading():
            self.rules["max_single_position"] = config.LIVE_MAX_POSITION_PCT
            self.rules["daily_trade_limit"] = config.LIVE_DAILY_TRADE_LIMIT
            self.rules["max_drawdown"] = config.LIVE_MAX_DRAWDOWN
            logger.info(
                f"🔴 实盘风控参数已启用: "
                f"单票上限 {self.rules['max_single_position']:.0%}, "
                f"日交易上限 {self.rules['daily_trade_limit']}, "
                f"最大回撤 {self.rules['max_drawdown']:.0%}"
            )

        # ── 加密货币专属风控规则 (Phase 8 新增) ──
        self.crypto_rules = {
            "max_single_position": config.CRYPTO_MAX_POSITION_PCT,
            "max_total_positions": config.CRYPTO_MAX_TOTAL_POSITIONS,
            "max_daily_loss": config.CRYPTO_MAX_DAILY_LOSS,
            "max_drawdown": config.CRYPTO_MAX_DRAWDOWN,
            "single_stop_loss": config.CRYPTO_SINGLE_STOP_LOSS,
            "daily_trade_limit": config.CRYPTO_DAILY_TRADE_LIMIT,
            "min_paper_days": config.CRYPTO_MIN_PAPER_DAYS,
        }
        if config.has_crypto_enabled():
            logger.info(
                f"🛡️ 加密货币风控规则已加载: "
                f"单币上限 {self.crypto_rules['max_single_position']:.0%}, "
                f"日交易上限 {self.crypto_rules['daily_trade_limit']}, "
                f"最大回撤 {self.crypto_rules['max_drawdown']:.0%}, "
                f"止损 {self.crypto_rules['single_stop_loss']:.0%}"
            )

        # 交易记录 (用于计算日交易次数) - 从 SQLite 恢复
        self._daily_trades: dict[str, int] = risk_state_db.get_all_daily_trades()
        # 加密货币交易记录 (独立计数, key 前缀 "crypto_")
        self._crypto_daily_trades: dict[str, int] = risk_state_db.get("crypto_daily_trades", {})
        # 高水位 (用于计算回撤) - 从 SQLite 恢复
        self._peak_equity: float = risk_state_db.get_peak_equity()
        # 加密货币峰值权益 (独立追踪)
        self._crypto_peak_equity: float = risk_state_db.get("crypto_peak_equity", 0.0)
        # Paper Trading 起始日期 - 从 SQLite 恢复
        self._paper_start_date: Optional[datetime] = risk_state_db.get_paper_start_date()
        # 加密货币 Paper Trading 起始日期
        self._crypto_paper_start: Optional[datetime] = risk_state_db.get("crypto_paper_start_date")
        if self._crypto_paper_start:
            try:
                self._crypto_paper_start = datetime.fromisoformat(self._crypto_paper_start)
            except (ValueError, TypeError):
                self._crypto_paper_start = None

        # 清理过期交易记录
        risk_state_db.cleanup_old_trades(keep_days=30)

        logger.info(
            f"🛡️ 风控引擎已加载状态 | "
            f"日交易记录: {len(self._daily_trades)} 天, "
            f"加密货币交易记录: {len(self._crypto_daily_trades)} 天, "
            f"峰值权益: , "
            f"Paper起始: {self._paper_start_date.strftime('%Y-%m-%d') if self._paper_start_date else '未设置'}"
        )

    def check_order(
        self,
        signal: TradeSignal,
        account: Account,
        positions: list[Position],
    ) -> RiskCheckResult:
        """下单前风控检查 (7 项全检) - 自动根据 asset_type 选择规则集"""
        # Phase 8: 加密货币信号使用专属风控规则
        if signal.asset_type == AssetType.CRYPTO:
            return self.check_crypto_order(signal, account, positions)

        checks = []

        # R1: 单票最大仓位
        checks.append(self._check_single_position(signal, account, positions))

        # R2: 最大持仓数 (仅买入时检查)
        if signal.direction == Direction.BUY:
            checks.append(self._check_position_limit(signal, positions))

        # R3: 日最大亏损
        checks.append(self._check_daily_loss(account))

        # R4: 总最大回撤
        checks.append(self._check_drawdown(account))

        # R5: 止损单优先 (不拦截止损卖出)
        if signal.direction != Direction.SELL or "stop_loss" not in signal.reason.lower():
            checks.append(self._check_trade_frequency())

        # R6: 日交易次数
        # (已在上面通过 _check_trade_frequency 处理)

        # R7: Paper Trading 周期 (仅实盘时检查)
        if not config.is_paper_trading():
            checks.append(self._check_paper_period())

        failed = [c for c in checks if not c.passed]
        result = RiskCheckResult(
            passed=len(failed) == 0,
            failed_checks=failed,
            reason="; ".join(c.reason for c in failed),
        )

        if not result.passed:
            logger.warning(f"风控拦截 [{signal.symbol}]: {result.reason}")
            notifier.warning(
                "风控拦截",
                f"订单 {signal.direction.value.upper()} {signal.symbol} 被拦截: {result.reason}",
            )

        return result

    # ═══════════════════════════════════════════════════════
    # 加密货币风控 (Phase 8 新增)
    # ═══════════════════════════════════════════════════════

    def check_crypto_order(
        self,
        signal: TradeSignal,
        account: Account,
        positions: list[Position],
    ) -> RiskCheckResult:
        """
        加密货币订单风控检查

        使用 crypto_rules 规则集，比股票更保守:
        - 单币仓位上限 20% (vs 股票 30%)
        - 日最大亏损 8% (vs 股票 5%)
        - 最大回撤 25% (vs 股票 15%)
        - 单币止损 10% (vs 股票 5%)
        - 日交易次数 50 (vs 股票 20, 24/7 交易)
        """
        if not config.has_crypto_enabled():
            return RiskCheckResult(
                passed=False,
                reason="加密货币模块未启用",
            )

        checks = []
        rules = self.crypto_rules

        # CR1: 单币最大仓位
        checks.append(self._check_crypto_single_position(signal, account, positions, rules))

        # CR2: 最大持仓币种数 (仅买入时检查)
        if signal.direction == Direction.BUY:
            checks.append(self._check_crypto_position_limit(signal, positions, rules))

        # CR3: 日最大亏损
        checks.append(self._check_crypto_daily_loss(account, rules))

        # CR4: 总最大回撤
        checks.append(self._check_crypto_drawdown(account, rules))

        # CR5: 日交易次数 (止损卖出不拦截)
        if signal.direction != Direction.SELL or "stop_loss" not in signal.reason.lower():
            checks.append(self._check_crypto_trade_frequency(rules))

        # CR6: Paper Trading 周期 (仅实盘时检查)
        if not config.is_paper_trading():
            checks.append(self._check_crypto_paper_period(rules))

        failed = [c for c in checks if not c.passed]
        result = RiskCheckResult(
            passed=len(failed) == 0,
            failed_checks=failed,
            reason="; ".join(c.reason for c in failed),
        )

        if not result.passed:
            logger.warning(f"🔒 加密货币风控拦截 [{signal.symbol}]: {result.reason}")
            notifier.warning(
                "加密货币风控拦截",
                f"订单 {signal.direction.value.upper()} {signal.symbol} 被拦截: {result.reason}",
            )

        return result

    def check_crypto_portfolio(
        self, account: Account, positions: list[Position],
    ) -> RiskStatus:
        """加密货币组合级风控检查 (定时运行)"""
        alerts = []
        rules = self.crypto_rules

        # 过滤出加密货币持仓
        crypto_positions = [p for p in positions if "/" in p.symbol or
                           getattr(p, "asset_type", AssetType.STOCK) == AssetType.CRYPTO]

        # 计算当前回撤 (使用加密货币峰值权益)
        if account.equity > self._crypto_peak_equity:
            self._crypto_peak_equity = account.equity
            risk_state_db.set("crypto_peak_equity", self._crypto_peak_equity)
        drawdown = 0.0
        if self._crypto_peak_equity > 0:
            drawdown = (self._crypto_peak_equity - account.equity) / self._crypto_peak_equity

        # 计算日亏损
        daily_loss = 0.0
        if account.last_equity > 0:
            daily_loss = (account.last_equity - account.equity) / account.last_equity

        # 检查最大回撤
        if drawdown > rules["max_drawdown"]:
            alerts.append(
                f"加密货币总回撤 {drawdown:.1%} 超过上限 {rules['max_drawdown']:.1%}"
            )

        # 检查日亏损
        if daily_loss > rules["max_daily_loss"]:
            alerts.append(
                f"加密货币日亏损 {daily_loss:.1%} 超过上限 {rules['max_daily_loss']:.1%}"
            )

        # 检查单币仓位
        max_pos = max((p.weight for p in crypto_positions), default=0.0)
        if max_pos > rules["max_single_position"]:
            alerts.append(
                f"单币最大仓位 {max_pos:.1%} 超过上限 {rules['max_single_position']:.1%}"
            )

        # 检查持仓数
        if len(crypto_positions) > rules["max_total_positions"]:
            alerts.append(
                f"加密货币持仓数 {len(crypto_positions)} 超过上限 {rules['max_total_positions']}"
            )

        is_alert = len(alerts) > 0
        if is_alert:
            for a in alerts:
                notifier.warning("加密货币组合风控告警", a)

        return RiskStatus(
            max_single_position=max_pos,
            max_total_drawdown=drawdown,
            daily_loss_limit=daily_loss,
            current_drawdown=drawdown,
            current_daily_loss=daily_loss,
            is_alert=is_alert,
            alerts=alerts,
        )

    def should_crypto_stop_loss(self, position: Position) -> bool:
        """加密货币止损检查 - 亏损超过阈值 (比股票更宽松)"""
        threshold = self.crypto_rules["single_stop_loss"]
        if position.unrealized_pnl_pct < -threshold:
            logger.warning(
                f"🔒 加密货币止损触发 [{position.symbol}]: "
                f"亏损 {position.unrealized_pnl_pct:.1%} < -{threshold:.1%}"
            )
            return True
        return False

    def record_crypto_trade(self):
        """记录一次加密货币交易 (独立计数) + 持久化"""
        today = datetime.now().strftime("%Y-%m-%d")
        self._crypto_daily_trades[today] = self._crypto_daily_trades.get(today, 0) + 1
        risk_state_db.set("crypto_daily_trades", self._crypto_daily_trades)

    def set_crypto_paper_start(self, date: datetime):
        """设置加密货币 Paper Trading 起始日期"""
        self._crypto_paper_start = date
        risk_state_db.set("crypto_paper_start_date", date.isoformat())
        logger.info(f"加密货币 Paper Trading 起始日期: {date.strftime('%Y-%m-%d')}")

    # ─── 加密货币单项检查 ───

    def _check_crypto_single_position(
        self, signal: TradeSignal,
        account: Account, positions: list[Position], rules: dict,
    ) -> RiskCheckItem:
        """CR1: 单币最大仓位"""
        max_pct = rules["max_single_position"]
        current_pos = next(
            (p for p in positions if p.symbol == signal.symbol), None
        )
        current_weight = current_pos.weight if current_pos else 0.0

        if signal.direction == Direction.BUY:
            new_weight = current_weight + signal.target_weight
            if new_weight > max_pct:
                return RiskCheckItem(
                    rule_name="CR1_单币最大仓位",
                    passed=False,
                    reason=f"{signal.symbol} 买入后仓位 {new_weight:.1%} > 上限 {max_pct:.1%}",
                )
        return RiskCheckItem(rule_name="CR1_单币最大仓位", passed=True)

    def _check_crypto_position_limit(
        self, signal: TradeSignal, positions: list[Position], rules: dict,
    ) -> RiskCheckItem:
        """CR2: 最大持仓币种数"""
        max_count = rules["max_total_positions"]
        # 只统计加密货币持仓
        crypto_positions = [p for p in positions if "/" in p.symbol or
                           getattr(p, "asset_type", AssetType.STOCK) == AssetType.CRYPTO]
        current_count = len(crypto_positions)
        existing = any(p.symbol == signal.symbol for p in crypto_positions)
        if not existing and current_count >= max_count:
            return RiskCheckItem(
                rule_name="CR2_最大持仓币种数",
                passed=False,
                reason=f"加密货币持仓数 {current_count} >= 上限 {max_count}",
            )
        return RiskCheckItem(rule_name="CR2_最大持仓币种数", passed=True)

    def _check_crypto_daily_loss(self, account: Account, rules: dict) -> RiskCheckItem:
        """CR3: 日最大亏损"""
        max_loss = rules["max_daily_loss"]
        if account.last_equity <= 0:
            return RiskCheckItem(rule_name="CR3_日最大亏损", passed=True)

        daily_loss = (account.last_equity - account.equity) / account.last_equity
        if daily_loss > max_loss:
            return RiskCheckItem(
                rule_name="CR3_日最大亏损",
                passed=False,
                reason=f"日亏损 {daily_loss:.1%} > 上限 {max_loss:.1%}, 当日停止交易",
            )
        return RiskCheckItem(rule_name="CR3_日最大亏损", passed=True)

    def _check_crypto_drawdown(self, account: Account, rules: dict) -> RiskCheckItem:
        """CR4: 总最大回撤"""
        max_dd = rules["max_drawdown"]
        if account.equity > self._crypto_peak_equity:
            self._crypto_peak_equity = account.equity
            risk_state_db.set("crypto_peak_equity", self._crypto_peak_equity)
        if self._crypto_peak_equity <= 0:
            return RiskCheckItem(rule_name="CR4_总最大回撤", passed=True)

        drawdown = (self._crypto_peak_equity - account.equity) / self._crypto_peak_equity
        if drawdown > max_dd:
            return RiskCheckItem(
                rule_name="CR4_总最大回撤",
                passed=False,
                reason=f"加密货币总回撤 {drawdown:.1%} > 上限 {max_dd:.1%}, 暂停策略",
            )
        return RiskCheckItem(rule_name="CR4_总最大回撤", passed=True)

    def _check_crypto_trade_frequency(self, rules: dict) -> RiskCheckItem:
        """CR5: 日交易次数"""
        max_trades = rules["daily_trade_limit"]
        today = datetime.now().strftime("%Y-%m-%d")
        today_count = self._crypto_daily_trades.get(today, 0)
        if today_count >= max_trades:
            return RiskCheckItem(
                rule_name="CR5_日交易次数",
                passed=False,
                reason=f"今日加密货币已交易 {today_count} 次 >= 上限 {max_trades}",
            )
        return RiskCheckItem(rule_name="CR5_日交易次数", passed=True)

    def _check_crypto_paper_period(self, rules: dict) -> RiskCheckItem:
        """CR6: Paper Trading 最少天数"""
        min_days = rules["min_paper_days"]
        if self._crypto_paper_start is None:
            return RiskCheckItem(
                rule_name="CR6_Paper周期",
                passed=False,
                reason=f"未设置加密货币 Paper Trading 起始日期, 不允许实盘",
            )
        days_elapsed = (datetime.now() - self._crypto_paper_start).days
        if days_elapsed < min_days:
            return RiskCheckItem(
                rule_name="CR6_Paper周期",
                passed=False,
                reason=f"加密货币 Paper Trading 仅 {days_elapsed} 天 < 最低 {min_days} 天",
            )
        return RiskCheckItem(rule_name="CR6_Paper周期", passed=True)

    def check_portfolio(
        self, account: Account, positions: list[Position],
    ) -> RiskStatus:
        """组合级风控检查 (定时运行)"""
        alerts = []

        # 计算当前回撤
        if account.equity > self._peak_equity:
            self._peak_equity = account.equity
            # 峰值权益更新时持久化到 SQLite
            risk_state_db.set_peak_equity(self._peak_equity)
        drawdown = 0.0
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - account.equity) / self._peak_equity

        # 计算日亏损
        daily_loss = 0.0
        if account.last_equity > 0:
            daily_loss = (account.last_equity - account.equity) / account.last_equity

        # 检查最大回撤
        if drawdown > self.rules["max_drawdown"]:
            alerts.append(
                f"总回撤 {drawdown:.1%} 超过上限 {self.rules['max_drawdown']:.1%}"
            )

        # 检查日亏损
        if daily_loss > self.rules["max_daily_loss"]:
            alerts.append(
                f"日亏损 {daily_loss:.1%} 超过上限 {self.rules['max_daily_loss']:.1%}"
            )

        # 检查单票仓位
        max_pos = max((p.weight for p in positions), default=0.0)
        if max_pos > self.rules["max_single_position"]:
            alerts.append(
                f"单票最大仓位 {max_pos:.1%} 超过上限 {self.rules['max_single_position']:.1%}"
            )

        # 检查持仓数
        if len(positions) > self.rules["max_total_positions"]:
            alerts.append(
                f"持仓数 {len(positions)} 超过上限 {self.rules['max_total_positions']}"
            )

        is_alert = len(alerts) > 0
        if is_alert:
            for a in alerts:
                notifier.warning("组合风控告警", a)

        return RiskStatus(
            max_single_position=max_pos,
            max_total_drawdown=drawdown,
            daily_loss_limit=daily_loss,
            current_drawdown=drawdown,
            current_daily_loss=daily_loss,
            is_alert=is_alert,
            alerts=alerts,
        )

    def should_stop_loss(self, position: Position) -> bool:
        """止损检查 - 亏损超过阈值"""
        if position.unrealized_pnl_pct < -self.rules["single_stop_loss"]:
            logger.warning(
                f"止损触发 [{position.symbol}]: "
                f"亏损 {position.unrealized_pnl_pct:.1%} < -{self.rules['single_stop_loss']:.1%}"
            )
            return True
        return False

    def record_trade(self):
        """记录一次交易 (用于日交易次数统计) + 持久化"""
        today = datetime.now().strftime("%Y-%m-%d")
        self._daily_trades[today] = self._daily_trades.get(today, 0) + 1
        # 持久化到 SQLite
        risk_state_db.set_daily_trade_count(today, self._daily_trades[today])

    def set_paper_start(self, date: datetime):
        """设置 Paper Trading 起始日期 + 持久化"""
        self._paper_start_date = date
        risk_state_db.set_paper_start_date(date)
        logger.info(f"Paper Trading 起始日期: {date.strftime('%Y-%m-%d')}")

    def persist_state(self) -> None:
        """手动持久化所有风控状态到 SQLite"""
        risk_state_db.set_peak_equity(self._peak_equity)
        if self._paper_start_date:
            risk_state_db.set_paper_start_date(self._paper_start_date)
        for date_str, count in self._daily_trades.items():
            risk_state_db.set_daily_trade_count(date_str, count)
        logger.info("风控状态已持久化到 SQLite")

    def load_state(self) -> None:
        """从 SQLite 重新加载风控状态"""
        self._daily_trades = risk_state_db.get_all_daily_trades()
        self._peak_equity = risk_state_db.get_peak_equity()
        self._paper_start_date = risk_state_db.get_paper_start_date()
        logger.info(
            f"🛡️ 风控状态已从 SQLite 恢复 | "
            f"日交易记录: {len(self._daily_trades)} 天, "
            f"峰值权益: "
        )

    # ─── 内部: 单项检查 ───

    def _check_single_position(
        self, signal: TradeSignal,
        account: Account, positions: list[Position],
    ) -> RiskCheckItem:
        """R1: 单票最大仓位"""
        max_pct = self.rules["max_single_position"]
        current_pos = next(
            (p for p in positions if p.symbol == signal.symbol), None
        )
        current_weight = current_pos.weight if current_pos else 0.0

        # 买入后预估仓位
        if signal.direction == Direction.BUY:
            # 实盘渐进式仓位缩放 (Phase 6 新增)
            scale_factor = 1.0
            if not config.is_paper_trading():
                try:
                    from src.execution.live_migration import live_migration_manager
                    scale_factor = live_migration_manager.get_position_scale_factor()
                except Exception:
                    pass

            new_weight = (current_weight + signal.target_weight) * scale_factor
            if new_weight > max_pct:
                return RiskCheckItem(
                    rule_name="R1_单票最大仓位",
                    passed=False,
                    reason=f"{signal.symbol} 买入后仓位 {new_weight:.1%} > 上限 {max_pct:.1%}"
                           + (f" (实盘缩放 {scale_factor:.0%})" if scale_factor < 1.0 else ""),
                )
        return RiskCheckItem(rule_name="R1_单票最大仓位", passed=True)

    def _check_position_limit(
        self, signal: TradeSignal, positions: list[Position],
    ) -> RiskCheckItem:
        """R2: 最大持仓数"""
        max_count = self.rules["max_total_positions"]
        current_count = len(positions)
        # 如果是新仓 (不在已有持仓中), 检查是否超限
        existing = any(p.symbol == signal.symbol for p in positions)
        if not existing and current_count >= max_count:
            return RiskCheckItem(
                rule_name="R2_最大持仓数",
                passed=False,
                reason=f"持仓数 {current_count} >= 上限 {max_count}",
            )
        return RiskCheckItem(rule_name="R2_最大持仓数", passed=True)

    def _check_daily_loss(self, account: Account) -> RiskCheckItem:
        """R3: 日最大亏损"""
        max_loss = self.rules["max_daily_loss"]
        if account.last_equity <= 0:
            return RiskCheckItem(rule_name="R3_日最大亏损", passed=True)

        daily_loss = (account.last_equity - account.equity) / account.last_equity
        if daily_loss > max_loss:
            return RiskCheckItem(
                rule_name="R3_日最大亏损",
                passed=False,
                reason=f"日亏损 {daily_loss:.1%} > 上限 {max_loss:.1%}, 当日停止交易",
            )
        return RiskCheckItem(rule_name="R3_日最大亏损", passed=True)

    def _check_drawdown(self, account: Account) -> RiskCheckItem:
        """R4: 总最大回撤"""
        max_dd = self.rules["max_drawdown"]
        if account.equity > self._peak_equity:
            self._peak_equity = account.equity
            # 峰值权益更新时持久化到 SQLite
            risk_state_db.set_peak_equity(self._peak_equity)
        if self._peak_equity <= 0:
            return RiskCheckItem(rule_name="R4_总最大回撤", passed=True)

        drawdown = (self._peak_equity - account.equity) / self._peak_equity
        if drawdown > max_dd:
            return RiskCheckItem(
                rule_name="R4_总最大回撤",
                passed=False,
                reason=f"总回撤 {drawdown:.1%} > 上限 {max_dd:.1%}, 暂停策略",
            )
        return RiskCheckItem(rule_name="R4_总最大回撤", passed=True)

    def _check_trade_frequency(self) -> RiskCheckItem:
        """R6: 日交易次数"""
        max_trades = self.rules["daily_trade_limit"]
        today = datetime.now().strftime("%Y-%m-%d")
        today_count = self._daily_trades.get(today, 0)
        if today_count >= max_trades:
            return RiskCheckItem(
                rule_name="R6_日交易次数",
                passed=False,
                reason=f"今日已交易 {today_count} 次 >= 上限 {max_trades}",
            )
        return RiskCheckItem(rule_name="R6_日交易次数", passed=True)

    def _check_paper_period(self) -> RiskCheckItem:
        """R7: Paper Trading 最少天数"""
        min_days = self.rules["min_paper_days"]
        if self._paper_start_date is None:
            return RiskCheckItem(
                rule_name="R7_Paper周期",
                passed=False,
                reason=f"未设置 Paper Trading 起始日期, 不允许实盘",
            )
        days_elapsed = (datetime.now() - self._paper_start_date).days
        if days_elapsed < min_days:
            return RiskCheckItem(
                rule_name="R7_Paper周期",
                passed=False,
                reason=f"Paper Trading 仅 {days_elapsed} 天 < 最低 {min_days} 天",
            )
        return RiskCheckItem(rule_name="R7_Paper周期", passed=True)


# 全局实例
risk_guard = RiskGuard()
