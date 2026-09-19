"""
工作流编排引擎 - 串联 5 个 Agent 完成完整交易链路
=====================================================
支持三种工作流:
  A. daily_trading_loop  - 每日自动运转主循环
  B. strategy_iteration  - 策略迭代优化
  C. emergency_stop      - 紧急止损

工作流模式:
  - sequential: 顺序执行 (默认)
  - 自带条件回退: 评级不达标时自动回退

v0.2.1 修复:
- P2: _pass_result 改为按 agent_name 匹配，支持非标准步骤名
- P3: emergency_stop 的 liquidate 步骤通过 extra_kwargs 传入 mode="liquidate"
- P7: strategy_iteration 增加条件检查，前置步骤无结果则跳过
- P9: 工作流状态判断修正为 completed_with_errors
"""

from datetime import datetime
from typing import Any, Callable, Optional

from loguru import logger

from src.agents.registry import agent_registry


class WorkflowStep:
    """工作流步骤"""

    def __init__(
        self,
        name: str,
        agent_name: str,
        description: str = "",
        condition: Optional[Callable] = None,
        on_fail: str = "continue",  # "continue" | "stop" | "retry"
        extra_kwargs: Optional[dict] = None,  # 该步骤专属的额外参数
    ):
        self.name = name
        self.agent_name = agent_name
        self.description = description
        self.condition = condition  # 执行条件函数
        self.on_fail = on_fail      # 失败时的行为
        self.extra_kwargs = extra_kwargs or {}  # P3: 步骤专属参数
        self.result: Any = None
        self.status: str = "pending"  # pending | running | done | failed | skipped

    def __repr__(self):
        return f"<WorkflowStep: {self.name} [{self.status}]>"


class Workflow:
    """工作流"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: list[WorkflowStep] = []
        self.results: dict[str, Any] = {}
        self._callbacks: dict[str, Callable] = {}

    def add_step(self, step: WorkflowStep) -> "Workflow":
        """添加步骤"""
        self.steps.append(step)
        return self

    def on(self, event: str, callback: Callable) -> "Workflow":
        """
        注册回调

        事件:
        - start: 工作流开始时触发
        - step_start: 步骤开始时触发
        - step_done: 步骤完成时触发
        - step_failed: 步骤失败时触发
        - end: 工作流结束时触发
        - workflow_done: 工作流完成时触发 (兼容旧名称)
        """
        self._callbacks[event] = callback
        return self

    def _emit(self, event: str, **kwargs) -> None:
        """触发回调"""
        cb = self._callbacks.get(event)
        if cb:
            try:
                cb(**kwargs)
            except Exception as e:
                logger.debug(f"回调 {event} 异常: {e}")

    def run(self, **initial_kwargs) -> dict:
        """
        运行工作流

        Args:
            initial_kwargs: 传递给第一个步骤的参数

        Returns:
            dict: 所有步骤的结果
        """
        logger.info(f"🔄 工作流 [{self.name}] 开始执行 | 步骤数: {len(self.steps)}")
        start_time = datetime.now()

        # 触发 start 事件
        self._emit("start", workflow=self, steps=self.steps)

        kwargs = initial_kwargs.copy()
        has_failure = False

        for i, step in enumerate(self.steps):
            logger.info(f"  [{i+1}/{len(self.steps)}] 步骤: {step.name} ({step.agent_name})")

            # 检查执行条件
            if step.condition and not step.condition(self.results):
                step.status = "skipped"
                logger.info(f"    ⏭️ 条件不满足，跳过")
                self.results[step.name] = None
                continue

            # 获取 Agent
            agent = agent_registry.get(step.agent_name)
            if agent is None:
                step.status = "failed"
                logger.error(f"    ❌ Agent 未找到: {step.agent_name}")
                has_failure = True
                if step.on_fail == "stop":
                    break
                continue

            # 执行步骤
            step.status = "running"
            self._emit("step_start", step=step, index=i)

            try:
                # P3 修复: 合并步骤专属参数 (extra_kwargs 优先级高于传递参数)
                step_kwargs = {**kwargs, **step.extra_kwargs}
                result = agent.run(**step_kwargs)
                step.result = result
                step.status = "done"
                self.results[step.name] = result

                self._emit("step_done", step=step, index=i, result=result)

                # 将结果传递给下一步
                kwargs = self._pass_result(step, result, kwargs)

            except Exception as e:
                step.status = "failed"
                step.result = None
                self.results[step.name] = None
                has_failure = True
                logger.error(f"    ❌ 步骤失败: {e}")

                self._emit("step_failed", step=step, index=i, error=str(e))

                if step.on_fail == "stop":
                    logger.warning(f"    步骤失败，工作流终止")
                    break

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"🔄 工作流 [{self.name}] 完成 ({elapsed:.1f}s)")

        # P9 修复: 工作流状态判断更清晰
        if has_failure:
            wf_status = "completed_with_errors"
        else:
            wf_status = "success"

        self._emit("end", workflow=self, results=self.results, elapsed=elapsed)
        self._emit("workflow_done", results=self.results, elapsed=elapsed)

        return {
            "workflow": self.name,
            "status": wf_status,
            "elapsed_seconds": elapsed,
            "steps": {s.name: s.status for s in self.steps},
            "results": self.results,
        }

    def _pass_result(self, step: WorkflowStep, result: Any, kwargs: dict) -> dict:
        """
        将当前步骤结果传递给下一步

        P2 修复: 改为按 agent_name 匹配而非 step.name，
        使得 strategy_iteration 和 emergency_stop 等非标准步骤名
        也能正确传递结果。

        Args:
            step: 当前步骤
            result: 当前步骤的执行结果
            kwargs: 当前步骤的参数

        Returns:
            dict: 传递给下一步的参数
        """
        new_kwargs = {}

        # P2 修复: 按 agent_name 匹配，而非 step.name
        agent_name = step.agent_name

        if agent_name == "research":
            # Research Agent 结果: ResearchReport 对象，有 candidate_stocks 属性
            if result and hasattr(result, "candidate_stocks"):
                new_kwargs["candidate_stocks"] = result.candidate_stocks
        elif agent_name == "strategy":
            # Strategy Agent 结果: dict
            # 单策略模式: 包含 strategy, signals 等
            # 多策略模式: 包含 signals_by_strategy, aggregated_signals 等
            new_kwargs["strategy_result"] = result
            # 传递多策略模式的中间结果
            if isinstance(result, dict):
                if "signals_by_strategy" in result:
                    new_kwargs["signals_by_strategy"] = result["signals_by_strategy"]
                if "aggregated_signals" in result:
                    new_kwargs["aggregated_signals"] = result["aggregated_signals"]
        elif agent_name == "backtest":
            # Backtest Agent 结果: BacktestResult 对象
            new_kwargs["backtest_result"] = result
        elif agent_name == "execution":
            # Execution Agent 结果: dict
            new_kwargs["execution_result"] = result
        elif agent_name == "monitor":
            # Monitor Agent 结果: dict
            new_kwargs["monitor_result"] = result

        # 保留原始 kwargs 中不在 new_kwargs 中的键
        for k, v in kwargs.items():
            if k not in new_kwargs:
                new_kwargs[k] = v

        return new_kwargs

    def summary(self) -> str:
        """生成工作流执行摘要"""
        lines = []
        lines.append(f"🔄 工作流: {self.name}")
        lines.append(f"{'─' * 40}")

        status_icons = {
            "done": "✅",
            "failed": "❌",
            "skipped": "⏭️",
            "pending": "⏳",
            "running": "🔄",
        }

        for step in self.steps:
            icon = status_icons.get(step.status, "❓")
            lines.append(f"  {icon} {step.name}: {step.status}")

        return "\n".join(lines)


class WorkflowEngine:
    """工作流引擎 - 管理工作流定义与执行"""

    def __init__(self):
        self._workflows: dict[str, Workflow] = {}
        self._initialized = False

    def initialize(self) -> None:
        """初始化预定义工作流"""
        if self._initialized:
            return

        # 确保 Agent 已初始化
        agent_registry.initialize()

        # 注册工作流 A: 每日自动运转主循环
        self._register_daily_trading_loop()

        # 注册工作流 B: 策略迭代优化
        self._register_strategy_iteration()

        # 注册工作流 C: 紧急止损
        self._register_emergency_stop()

        # 注册工作流 D: 多策略组合再平衡 (Phase 6 T6.2)
        self._register_multi_strategy_rebalance()

        # 注册工作流 E: 加密货币交易主循环 (Phase 8)
        self._register_crypto_trading_loop()

        self._initialized = True
        logger.info(f"✅ 工作流引擎初始化完成 | 已注册 {len(self._workflows)} 个工作流")

    def _register_daily_trading_loop(self) -> None:
        """工作流 A: 每日自动运转主循环"""
        wf = Workflow(
            name="daily_trading_loop",
            description="每日自动运转: 调研 -> 策略 -> 回测 -> 执行 -> 监控",
        )

        wf.add_step(WorkflowStep(
            name="research",
            agent_name="research",
            description="市场调研 + 股票筛选",
            on_fail="stop",  # 调研失败则不继续
        ))

        wf.add_step(WorkflowStep(
            name="strategy",
            agent_name="strategy",
            description="因子搜索 + 模型训练 + 信号生成",
            on_fail="stop",  # 策略失败则不继续
        ))

        wf.add_step(WorkflowStep(
            name="backtest",
            agent_name="backtest",
            description="快速回测验证策略",
            on_fail="continue",
        ))

        # 执行条件: 回测评级 >= C 才执行
        # 兼容 BacktestResult 对象 (有 .grade 属性) 和普通 dict (有 "grade" 键)
        def _can_execute(results):
            bt = results.get("backtest")
            if bt is None:
                return True  # 无回测结果时默认执行

            # 提取评级值，兼容对象和 dict
            grade = None
            if isinstance(bt, dict):
                grade = bt.get("grade")
            elif hasattr(bt, "grade"):
                grade = bt.grade

            if grade is None:
                return True  # 无评级信息时默认执行

            # 统一转为字符串值进行比较
            grade_str = grade.value if hasattr(grade, "value") else str(grade)

            # A/B/C 可执行; D 不可执行; S (超优) 也可执行
            return grade_str in ("A", "B", "C", "S")

        wf.add_step(WorkflowStep(
            name="execution",
            agent_name="execution",
            description="风控检查 + 执行交易信号",
            condition=_can_execute,
            on_fail="continue",
        ))

        wf.add_step(WorkflowStep(
            name="monitor",
            agent_name="monitor",
            description="盘中监控 + 止损检查",
            on_fail="continue",
        ))

        self._workflows["daily_trading_loop"] = wf

    def _register_strategy_iteration(self) -> None:
        """工作流 B: 策略迭代优化"""
        wf = Workflow(
            name="strategy_iteration",
            description="策略迭代: 回顾 -> 分析 -> 优化 -> 回测 -> 淘汰",
        )

        wf.add_step(WorkflowStep(
            name="monitor_review",
            agent_name="monitor",
            description="回顾上周策略表现",
            on_fail="continue",
        ))

        wf.add_step(WorkflowStep(
            name="strategy_optimize",
            agent_name="strategy",
            description="重新优化因子和参数",
            on_fail="continue",
        ))

        # P7 修复: backtest_validate 增加条件检查
        # 如果 strategy_optimize 失败或无结果，则跳过回测
        def _can_backtest(results):
            # 检查 strategy_optimize 步骤是否有结果
            strategy_result = results.get("strategy_optimize")
            if strategy_result is None:
                logger.info("    ⏭️ 策略优化步骤无结果，跳过回测验证")
                return False
            return True

        wf.add_step(WorkflowStep(
            name="backtest_validate",
            agent_name="backtest",
            description="完整回测验证 (1年历史数据)",
            condition=_can_backtest,
            on_fail="continue",
        ))

        self._workflows["strategy_iteration"] = wf

    def _register_emergency_stop(self) -> None:
        """工作流 C: 紧急止损"""
        wf = Workflow(
            name="emergency_stop",
            description="紧急止损: 风控检查 -> 清仓 -> 通知",
        )

        wf.add_step(WorkflowStep(
            name="risk_check",
            agent_name="monitor",
            description="紧急风控检查",
            on_fail="continue",
            extra_kwargs={"mode": "snapshot"},  # 确保 monitor 用快照模式
        ))

        # P3 修复: liquidate 步骤通过 extra_kwargs 传入 mode="liquidate"
        wf.add_step(WorkflowStep(
            name="liquidate",
            agent_name="execution",
            description="清仓所有持仓",
            on_fail="stop",  # 清仓失败必须停止
            extra_kwargs={"mode": "liquidate"},  # 执行清仓模式
        ))

        wf.add_step(WorkflowStep(
            name="notify",
            agent_name="monitor",
            description="发送止损通知",
            on_fail="continue",
            extra_kwargs={"mode": "report"},  # 生成止损报告
        ))

        self._workflows["emergency_stop"] = wf

    def _register_multi_strategy_rebalance(self) -> None:
        """工作流 D: 多策略组合再平衡 (Phase 6 T6.2)

        步骤:
        1. get_active_strategies - 获取所有活跃策略
        2. generate_signals      - 各策略生成交易信号
        3. aggregate_signals     - 组合管理器聚合信号
        4. risk_check            - 组合级风控检查 (暂用 execution agent 快照)
        5. execute_trades        - 执行聚合后的交易信号
        """
        wf = Workflow(
            name="multi_strategy_rebalance",
            description="多策略组合再平衡: 获取活跃策略 -> 各策略信号生成 -> 组合聚合 -> 风控 -> 执行",
        )

        wf.add_step(WorkflowStep(
            name="get_active_strategies",
            agent_name="strategy",
            description="获取所有活跃策略",
            on_fail="stop",
            extra_kwargs={"mode": "get_active"},
        ))

        wf.add_step(WorkflowStep(
            name="generate_signals",
            agent_name="strategy",
            description="各策略生成交易信号",
            on_fail="continue",
            extra_kwargs={"mode": "generate_all"},
        ))

        wf.add_step(WorkflowStep(
            name="aggregate_signals",
            agent_name="strategy",
            description="组合管理器聚合多策略信号",
            on_fail="continue",
            extra_kwargs={"mode": "aggregate"},
        ))

        # 风控检查: 使用 monitor agent 做组合级快照
        wf.add_step(WorkflowStep(
            name="risk_check",
            agent_name="monitor",
            description="组合级风控检查",
            on_fail="continue",
            extra_kwargs={"mode": "snapshot"},
        ))

        # 执行聚合信号: 仅在风控通过时执行
        def _can_execute_portfolio(results):
            """风控通过才执行"""
            risk = results.get("risk_check")
            if risk is None:
                return True  # 风控步骤被跳过或失败时默认执行
            if isinstance(risk, dict):
                risk_status = risk.get("risk_status", {})
                if risk_status.get("is_alert", False):
                    logger.warning("⚠️ 组合风控告警，跳过执行")
                    return False
            return True

        wf.add_step(WorkflowStep(
            name="execute_trades",
            agent_name="execution",
            description="执行聚合后的交易信号",
            condition=_can_execute_portfolio,
            on_fail="continue",
            extra_kwargs={"mode": "portfolio"},
        ))

        self._workflows["multi_strategy_rebalance"] = wf

    # ═══════════════════════════════════════════════════════
    # Phase 8: 加密货币交易工作流
    # ═══════════════════════════════════════════════════════

    def _register_crypto_trading_loop(self) -> None:
        """
        工作流 E: 加密货币交易主循环 (Phase 8 新增)

        24/7 交易, 与股票工作流并行但独立运行。

        步骤:
        1. crypto_research   - 加密货币市场调研 (Finnhub crypto news + 市场总览)
        2. crypto_strategy   - 加密货币信号生成 (动量/模型信号, 使用 crypto_signal_generator)
        3. crypto_risk_check - 加密货币专属风控检查 (crypto_rules)
        4. crypto_execution  - 执行加密货币交易信号
        5. crypto_monitor    - 加密货币持仓监控与止损

        与股票工作流的区别:
        - 数据源: crypto_collector (Alpaca Crypto API)
        - 风控: crypto_rules (更保守的参数)
        - 调度: 24/7, 信号扫描间隔 4 小时
        - 资产标记: 所有信号 asset_type=CRYPTO
        """
        wf = Workflow(
            name="crypto_trading_loop",
            description="加密货币交易循环: 调研 -> 信号生成 -> 风控 -> 执行 -> 监控 (24/7)",
        )

        # 步骤1: 加密货币市场调研
        wf.add_step(WorkflowStep(
            name="crypto_research",
            agent_name="research",
            description="加密货币市场调研 + 候选币种筛选",
            on_fail="stop",
            extra_kwargs={"asset_type": "crypto"},
        ))

        # 步骤2: 加密货币信号生成
        wf.add_step(WorkflowStep(
            name="crypto_strategy",
            agent_name="strategy",
            description="加密货币信号生成 (动量/模型信号)",
            on_fail="stop",
            extra_kwargs={"mode": "crypto_signals"},
        ))

        # 步骤3: 风控检查 (使用 monitor agent 快照模式, 但走 crypto_rules)
        wf.add_step(WorkflowStep(
            name="crypto_risk_check",
            agent_name="monitor",
            description="加密货币风控检查 (crypto_rules)",
            on_fail="continue",
            extra_kwargs={"mode": "crypto_snapshot"},
        ))

        # 步骤4: 执行交易信号 (仅风控通过时执行)
        def _can_execute_crypto(results):
            """加密货币风控通过才执行"""
            risk = results.get("crypto_risk_check")
            if risk is None:
                return True
            if isinstance(risk, dict):
                risk_status = risk.get("risk_status", {})
                if risk_status.get("is_alert", False):
                    logger.warning("⚠️ 加密货币风控告警，跳过执行")
                    return False
            return True

        wf.add_step(WorkflowStep(
            name="crypto_execution",
            agent_name="execution",
            description="执行加密货币交易信号",
            condition=_can_execute_crypto,
            on_fail="continue",
            extra_kwargs={"mode": "crypto"},
        ))

        # 步骤5: 加密货币持仓监控
        wf.add_step(WorkflowStep(
            name="crypto_monitor",
            agent_name="monitor",
            description="加密货币持仓监控 + 止损检查",
            on_fail="continue",
            extra_kwargs={"mode": "crypto_monitor"},
        ))

        self._workflows["crypto_trading_loop"] = wf
        logger.info("🔄 加密货币交易工作流已注册: crypto_trading_loop")

    def get(self, name: str) -> Optional[Workflow]:
        """获取工作流"""
        if not self._initialized:
            self.initialize()
        return self._workflows.get(name)

    def list_workflows(self) -> list[str]:
        """列出所有工作流名称"""
        if not self._initialized:
            self.initialize()
        return list(self._workflows.keys())

    def run(self, name: str, **kwargs) -> dict:
        """
        运行指定工作流

        Args:
            name: 工作流名称
            **kwargs: 传递给工作流的参数

        Returns:
            dict: 工作流执行结果
        """
        wf = self.get(name)
        if wf is None:
            logger.error(f"工作流不存在: {name}")
            return {"error": f"工作流不存在: {name}"}

        return wf.run(**kwargs)

    def run_daily_loop(self, user_interest: str = "") -> dict:
        """快捷方法: 运行每日主循环"""
        return self.run("daily_trading_loop", user_interest=user_interest)

    def run_strategy_iteration(self, stock_pool: list[str] = None) -> dict:
        """快捷方法: 运行策略迭代"""
        return self.run("strategy_iteration", stock_pool=stock_pool)

    def run_emergency_stop(self) -> dict:
        """快捷方法: 紧急止损"""
        return self.run("emergency_stop")

    def run_multi_strategy_rebalance(self, portfolio_id: str = "") -> dict:
        """快捷方法: 多策略组合再平衡"""
        return self.run("multi_strategy_rebalance", portfolio_id=portfolio_id)

    def run_crypto_loop(self, user_interest: str = "") -> dict:
        """快捷方法: 运行加密货币交易循环 (Phase 8)"""
        return self.run("crypto_trading_loop", user_interest=user_interest)


# 全局实例
workflow_engine = WorkflowEngine()
