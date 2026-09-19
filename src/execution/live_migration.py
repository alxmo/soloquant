"""
实盘迁移安全管理器
=====================================================
Phase 6 新增 (T6.1): Paper Trading -> Live Trading 安全迁移

功能:
- 6 项预检: 确保模拟盘运行充分后才允许切换实盘
- 渐进式仓位: 切换实盘后分 N 周逐步放大仓位
- 审计日志: 记录迁移过程，可追溯
- 状态查询: 迁移进度和仓位系数

安全设计:
1. Paper Trading 天数 >= 30 天 (R7 规则)
2. 至少 1 个 A/B 级策略通过回测
3. 当前无风控告警
4. 总回撤 < 10%
5. 日交易次数合理
6. 用户二次确认

使用示例:
    from src.execution.live_migration import live_migration_manager

    # 检查迁移就绪状态
    status = live_migration_manager.check_readiness()
    if status.all_passed:
        # 执行迁移
        live_migration_manager.migrate(confirm=True)
"""

from datetime import datetime
import json
from typing import Optional

from loguru import logger

from src.utils.config import config
from src.utils.notifier import notifier


class MigrationCheckItem:
    """单项预检结果"""

    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


class MigrationStatus:
    """迁移就绪状态"""

    def __init__(self):
        self.checks: list[MigrationCheckItem] = []
        self.all_passed: bool = False
        self.current_mode: str = "paper"
        self.summary: str = ""

    def to_dict(self) -> dict:
        return {
            "checks": [c.to_dict() for c in self.checks],
            "all_passed": self.all_passed,
            "current_mode": self.current_mode,
            "summary": self.summary,
        }


class LiveMigrationManager:
    """
    实盘迁移安全管理器

    管理 Paper Trading -> Live Trading 的安全迁移流程
    """

    def __init__(self):
        self._audit_log_path = config.DATA_DIR / "migration_audit.json"
        self._migration_state: Optional[dict] = None
        self._load_state()

    # ─── 预检 ───

    def check_readiness(
        self,
        paper_start_date: Optional[datetime] = None,
        strategies: Optional[list] = None,
        risk_alerts: Optional[list[str]] = None,
        current_drawdown: float = 0.0,
        daily_trade_count: int = 0,
    ) -> MigrationStatus:
        """
        执行 6 项预检，判断是否可以迁移到实盘

        Args:
            paper_start_date: Paper Trading 起始日期
            strategies: 当前策略列表 (StrategyConfig 对象)
            risk_alerts: 当前风控告警列表
            current_drawdown: 当前回撤 (0~1)
            daily_trade_count: 今日交易次数

        Returns:
            MigrationStatus
        """
        status = MigrationStatus()
        status.current_mode = config.TRADING_MODE

        # 如果已经是实盘模式
        if not config.is_paper_trading():
            status.summary = "当前已是实盘模式"
            status.all_passed = False
            return status

        # 检查 1: Paper Trading 天数 >= 30 天
        days = 0
        if paper_start_date:
            days = (datetime.now() - paper_start_date).days
        passed_1 = days >= config.MIN_PAPER_DAYS
        status.checks.append(MigrationCheckItem(
            name="R7_Paper周期",
            passed=passed_1,
            detail=f"已模拟 {days} 天 (要求 >= {config.MIN_PAPER_DAYS} 天)",
        ))

        # 检查 2: 至少 1 个 A/B 级策略
        grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
        min_grade_order = grade_order.get(config.LIVE_MIN_GRADE, 1)
        qualified = []
        if strategies:
            for s in strategies:
                bt = getattr(s, "_backtest_result", None)
                if bt and hasattr(bt, "grade"):
                    grade_val = bt.grade.value if hasattr(bt.grade, "value") else str(bt.grade)
                    if grade_order.get(grade_val, 3) <= min_grade_order:
                        qualified.append(f"{s.name}({grade_val})")
        passed_2 = len(qualified) > 0
        detail_2 = f"有 {len(qualified)} 个达标策略: {', '.join(qualified)}" if qualified else \
                   f"无 {config.LIVE_MIN_GRADE} 级及以上策略"
        status.checks.append(MigrationCheckItem(
            name="策略质量",
            passed=passed_2,
            detail=detail_2,
        ))

        # 检查 3: 当前无风控告警
        alert_count = len(risk_alerts) if risk_alerts else 0
        passed_3 = alert_count == 0
        status.checks.append(MigrationCheckItem(
            name="风控告警",
            passed=passed_3,
            detail=f"当前 {alert_count} 条告警" + (
                f": {'; '.join(risk_alerts[:3])}" if risk_alerts else ""
            ),
        ))

        # 检查 4: 总回撤 < LIVE_MAX_DRAWDOWN
        passed_4 = current_drawdown < config.LIVE_MAX_DRAWDOWN
        status.checks.append(MigrationCheckItem(
            name="回撤控制",
            passed=passed_4,
            detail=f"当前回撤 {current_drawdown:.1%} (要求 < {config.LIVE_MAX_DRAWDOWN:.1%})",
        ))

        # 检查 5: 日交易次数合理 (不超过模拟盘上限的 50%)
        reasonable_limit = config.DAILY_TRADE_LIMIT // 2
        passed_5 = daily_trade_count <= reasonable_limit
        status.checks.append(MigrationCheckItem(
            name="交易频率",
            passed=passed_5,
            detail=f"今日交易 {daily_trade_count} 次 (合理上限 {reasonable_limit} 次)",
        ))

        # 检查 6: Alpaca API 已配置
        passed_6 = config.has_alpaca_keys()
        status.checks.append(MigrationCheckItem(
            name="API配置",
            passed=passed_6,
            detail="Alpaca API Key 已配置" if passed_6 else "Alpaca API Key 未配置",
        ))

        # 汇总
        status.all_passed = all(c.passed for c in status.checks)

        if status.all_passed:
            status.summary = "✅ 所有预检通过，可以安全迁移到实盘"
        else:
            failed = [c.name for c in status.checks if not c.passed]
            status.summary = f"❌ {len(failed)} 项未通过: {', '.join(failed)}"

        return status

    # ─── 迁移执行 ───

    def migrate(self, confirm: bool = False) -> dict:
        """
        执行 Paper -> Live 迁移

        Args:
            confirm: 用户二次确认 (必须为 True 才执行)

        Returns:
            dict: 迁移结果
        """
        if not confirm:
            logger.warning("迁移需要用户确认 (confirm=True)")
            return {"success": False, "reason": "需要用户确认"}

        if not config.is_paper_trading():
            logger.info("当前已是实盘模式，无需迁移")
            return {"success": False, "reason": "已是实盘模式"}

        logger.warning("🔴 开始执行 Paper -> Live 迁移!")

        # 记录迁移审计日志
        migration_record = {
            "timestamp": datetime.now().isoformat(),
            "action": "paper_to_live",
            "previous_mode": config.TRADING_MODE,
            "new_mode": "live",
            "progressive_weeks": config.LIVE_PROGRESSIVE_WEEKS,
        }

        # 切换模式
        config.TRADING_MODE = "live"

        # 初始化渐进式仓位状态
        self._migration_state = {
            "migrated": True,
            "migration_date": datetime.now().isoformat(),
            "progressive_weeks": config.LIVE_PROGRESSIVE_WEEKS,
            "current_week": 0,
            "scale_factors": self._calc_scale_factors(config.LIVE_PROGRESSIVE_WEEKS),
        }
        self._save_state()

        # 通知
        notifier.critical(
            "实盘迁移",
            f"已从模拟盘切换到实盘! "
            f"渐进式仓位: 第1周 {self._migration_state['scale_factors'][0]:.0%} -> "
            f"第{config.LIVE_PROGRESSIVE_WEEKS}周 100%",
        )

        migration_record["success"] = True
        self._append_audit_log(migration_record)

        logger.info("✅ 迁移完成，实盘模式已激活 (渐进式仓位)")
        return {"success": True, "state": self._migration_state}

    def rollback(self) -> dict:
        """
        回滚到 Paper Trading 模式

        Returns:
            dict: 回滚结果
        """
        logger.warning("📋 回滚到模拟盘模式")

        previous_mode = config.TRADING_MODE
        config.TRADING_MODE = "paper"

        rollback_record = {
            "timestamp": datetime.now().isoformat(),
            "action": "live_to_paper_rollback",
            "previous_mode": previous_mode,
            "new_mode": "paper",
        }

        if self._migration_state:
            self._migration_state["migrated"] = False
            self._migration_state["rollback_date"] = datetime.now().isoformat()
            self._save_state()

        notifier.warning(
            "实盘回滚",
            "已从实盘回滚到模拟盘模式",
        )

        rollback_record["success"] = True
        self._append_audit_log(rollback_record)

        return {"success": True, "message": "已回滚到模拟盘"}

    # ─── 渐进式仓位 ───

    def get_position_scale_factor(self) -> float:
        """
        获取当前仓位缩放系数

        迁移后按周递增:
        - 第 1 周: 50%
        - 第 2 周: 75%
        - 第 3 周: 100%

        未迁移或模拟盘模式返回 1.0

        Returns:
            float: 0~1 的缩放系数
        """
        if not self._migration_state or not self._migration_state.get("migrated"):
            return 1.0

        if config.is_paper_trading():
            return 1.0

        migration_date = datetime.fromisoformat(
            self._migration_state["migration_date"]
        )
        weeks_elapsed = (datetime.now() - migration_date).days // 7

        scale_factors = self._migration_state.get("scale_factors", [1.0])

        if weeks_elapsed >= len(scale_factors):
            # 超过渐进期，全仓位
            return 1.0

        self._migration_state["current_week"] = weeks_elapsed
        return scale_factors[weeks_elapsed]

    @staticmethod
    def _calc_scale_factors(weeks: int) -> list[float]:
        """
        计算渐进式仓位系数

        策略: 线性递增从 50% 到 100%
        - 1 周: [1.0] (直接全仓位)
        - 2 周: [0.5, 1.0]
        - 3 周: [0.5, 0.75, 1.0]

        Args:
            weeks: 渐进周数

        Returns:
            list[float]: 每周的仓位系数
        """
        if weeks <= 1:
            return [1.0]

        factors = []
        for i in range(weeks):
            # 线性从 0.5 到 1.0
            factor = 0.5 + 0.5 * (i / (weeks - 1))
            factors.append(round(factor, 4))
        return factors

    # ─── 状态查询 ───

    def get_migration_status(self) -> dict:
        """获取迁移状态摘要"""
        if not self._migration_state:
            return {
                "migrated": False,
                "mode": config.TRADING_MODE,
                "scale_factor": 1.0,
                "message": "未迁移 (模拟盘模式)",
            }

        scale = self.get_position_scale_factor()
        migration_date = self._migration_state.get("migration_date", "")
        weeks_elapsed = 0
        if migration_date:
            try:
                weeks_elapsed = (datetime.now() - datetime.fromisoformat(migration_date)).days // 7
            except Exception:
                pass

        progressive_weeks = self._migration_state.get("progressive_weeks", 0)

        if weeks_elapsed >= progressive_weeks:
            phase = "全仓位阶段"
        else:
            phase = f"渐进期第 {weeks_elapsed + 1}/{progressive_weeks} 周"

        return {
            "migrated": self._migration_state.get("migrated", False),
            "mode": config.TRADING_MODE,
            "migration_date": migration_date,
            "weeks_elapsed": weeks_elapsed,
            "progressive_weeks": progressive_weeks,
            "current_scale_factor": scale,
            "phase": phase,
        }

    # ─── 内部: 状态持久化 ───

    def _load_state(self):
        """从文件加载迁移状态"""
        if not self._audit_log_path.exists():
            self._migration_state = None
            return

        try:
            with open(self._audit_log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 取最后一条迁移记录作为当前状态
            records = data.get("records", [])
            for record in reversed(records):
                if record.get("action") == "paper_to_live" and record.get("success"):
                    self._migration_state = {
                        "migrated": True,
                        "migration_date": record["timestamp"],
                        "progressive_weeks": record.get("progressive_weeks", 3),
                        "current_week": 0,
                        "scale_factors": self._calc_scale_factors(
                            record.get("progressive_weeks", 3)
                        ),
                    }
                    break
        except Exception as e:
            logger.debug(f"加载迁移状态失败: {e}")
            self._migration_state = None

    def _save_state(self):
        """保存迁移状态到文件"""
        try:
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)

            # 读取现有审计日志
            data = {"records": []}
            if self._audit_log_path.exists():
                with open(self._audit_log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            # 追加状态记录
            if self._migration_state:
                data["current_state"] = self._migration_state

            with open(self._audit_log_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"保存迁移状态失败: {e}")

    def _append_audit_log(self, record: dict):
        """追加审计日志"""
        try:
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)

            data = {"records": []}
            if self._audit_log_path.exists():
                with open(self._audit_log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            data.setdefault("records", []).append(record)

            with open(self._audit_log_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"写入审计日志失败: {e}")


# 全局实例
live_migration_manager = LiveMigrationManager()
