"""
目标驱动 AI 自主交易引擎
=====================================================
用户只需说 "我有10万美元，想赚20%"，
AI 全自主完成: 解析目标 -> 推导策略参数 -> 选股 -> 训练 -> 回测 -> 迭代 -> 执行 -> 报告

核心流程:
  1. parse_goal()    - 解析自然语言，提取资金/目标收益/时间/风险偏好
  2. derive_params() - 根据目标推导风险等级、板块、模型、因子、仓位
  3. run()           - 全自主执行: 调研->策略->回测(迭代)->执行->报告
"""

import re
from typing import Optional

from loguru import logger

from src.utils.config import config


# ─── 风险等级配置 ───
RISK_PROFILES = {
    "conservative": {
        "label": "🟢 保守型",
        "desc": "稳扎稳打，优先保本",
        "sectors": ["消费", "金融", "医疗"],
        "model_type": "lightgbm",
        "factor_focus": ["value", "quality"],       # 价值因子 + 质量因子
        "max_position": 0.15,                        # 单票最大仓位 15%
        "stop_loss": 0.03,                           # 单票止损 3%
        "max_drawdown": 0.08,                        # 组合最大回撤 8%
        "max_candidates": 5,
    },
    "balanced": {
        "label": "🟡 稳健型",
        "desc": "攻守平衡，适合大多数投资者",
        "sectors": ["科技", "消费", "金融"],
        "model_type": "lightgbm",
        "factor_focus": ["momentum", "value", "technical"],
        "max_position": 0.25,
        "stop_loss": 0.05,
        "max_drawdown": 0.12,
        "max_candidates": 5,
    },
    "aggressive": {
        "label": "🟠 进取型",
        "desc": "追求高收益，能承受较大波动",
        "sectors": ["科技", "AI芯片", "新能源"],
        "model_type": "lightgbm",
        "factor_focus": ["momentum", "sentiment", "technical"],
        "max_position": 0.30,
        "stop_loss": 0.07,
        "max_drawdown": 0.18,
        "max_candidates": 7,
    },
    "extreme": {
        "label": "🔴 激进型",
        "desc": "高风险高回报，波动极大",
        "sectors": ["AI芯片", "新能源", "加密货币概念"],
        "model_type": "lstm",
        "factor_focus": ["momentum", "sentiment", "technical"],
        "max_position": 0.35,
        "stop_loss": 0.10,
        "max_drawdown": 0.25,
        "max_candidates": 8,
    },
}


def _classify_risk(target_return: float, user_hint: str = "") -> str:
    """
    根据目标收益率和用户提示词推导风险等级

    Args:
        target_return: 年化目标收益率 (百分比, 如 20 表示 20%)
        user_hint: 用户额外的风险偏好描述

    Returns:
        风险等级 key: conservative / balanced / aggressive / extreme
    """
    hint_lower = user_hint.lower()

    # 用户显式偏好优先
    if any(w in hint_lower for w in ["保守", "稳妥", "稳健", "风险小", "别太", "低风险", "safe", "conservative"]):
        if target_return > 25:
            logger.info("用户希望保守但目标较高，取 balanced 折中")
            return "balanced"
        return "conservative"
    if any(w in hint_lower for w in ["激进", "高风险", "全部投入", "赌", "aggressive", "extreme"]):
        return "extreme"

    # 按目标收益率自动推导
    if target_return <= 10:
        return "conservative"
    elif target_return <= 25:
        return "balanced"
    elif target_return <= 50:
        return "aggressive"
    else:
        return "extreme"


class GoalPlan:
    """目标投资计划"""

    def __init__(
        self,
        capital: float,
        target_return: float,
        timeframe_months: int = 12,
        risk_level: str = "balanced",
        user_text: str = "",
    ):
        self.capital = capital
        self.target_return = target_return          # 年化目标收益率 (%)
        self.timeframe_months = timeframe_months
        self.risk_level = risk_level
        self.user_text = user_text

        # 从风险配置推导
        profile = RISK_PROFILES[risk_level]
        self.risk_label = profile["label"]
        self.risk_desc = profile["desc"]
        self.sectors = profile["sectors"]
        self.model_type = profile["model_type"]
        self.factor_focus = profile["factor_focus"]
        self.max_position = profile["max_position"]
        self.stop_loss = profile["stop_loss"]
        self.max_drawdown = profile["max_drawdown"]
        self.max_candidates = profile["max_candidates"]

        # 计算目标金额
        self.target_amount = capital * (1 + target_return / 100)
        self.profit_target = self.target_amount - capital

    def __repr__(self):
        return (
            f"GoalPlan(capital=${self.capital:,.0f}, "
            f"target={self.target_return}%/年, "
            f"risk={self.risk_level}, "
            f"model={self.model_type})"
        )


class GoalEngine:
    """
    目标驱动 AI 自主交易引擎

    用法:
        engine = GoalEngine()
        report = engine.run("我有10万美元，想赚20%")
    """

    # ─── 目标解析 ───

    def parse_goal(self, text: str) -> Optional[GoalPlan]:
        """
        从自然语言解析投资目标

        支持的句式:
          - "我有10万美元，想赚20%"
          - "5万本金，目标翻倍"
          - "投10万，年化15%就行，风险别太大"
          - "100k, want 20% return"
          - "10万块，想赚5万"

        Returns:
            GoalPlan 或 None (解析失败)
        """
        logger.info(f"🎯 开始解析投资目标: {text}")

        capital = self._extract_capital(text)
        target_return = self._extract_target_return(text, capital)
        timeframe = self._extract_timeframe(text)
        risk_hint = self._extract_risk_hint(text)

        if capital is None or capital <= 0:
            logger.warning("无法解析投资金额")
            return None

        if target_return is None or target_return <= 0:
            logger.warning("无法解析目标收益率，默认 15%")
            target_return = 15.0

        risk_level = _classify_risk(target_return, risk_hint)

        plan = GoalPlan(
            capital=capital,
            target_return=target_return,
            timeframe_months=timeframe,
            risk_level=risk_level,
            user_text=text,
        )

        logger.info(f"  资金: ${plan.capital:,.0f}")
        logger.info(f"  目标: 年化 {plan.target_return}% (目标金额 ${plan.target_amount:,.0f})")
        logger.info(f"  周期: {plan.timeframe_months} 个月")
        logger.info(f"  风险: {plan.risk_label} ({plan.risk_level})")
        logger.info(f"  板块: {plan.sectors}")
        logger.info(f"  模型: {plan.model_type} | 因子: {plan.factor_focus}")

        return plan

    def _extract_capital(self, text: str) -> Optional[float]:
        """提取投资金额"""
        # 中文: 10万 / 10万美元 / 5万块 / 10w
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:万|w|W)\s*(?:美元|美金|刀|块|元|美金)?', text)
        if m:
            return float(m.group(1)) * 10000

        # 英文: 100k / 100000 / $100,000
        m = re.search(r'\$?\s*(\d{1,3}(?:,\d{3})+|\d+)\s*(k|K)?', text)
        if m:
            raw = m.group(1).replace(",", "")
            val = float(raw)
            if m.group(2):  # 有 k/K 后缀
                val *= 1000
            # 只取合理的金额范围 (1000 ~ 100,000,000)
            if 1000 <= val <= 100_000_000:
                return val

        return None

    def _extract_target_return(self, text: str, capital: float) -> Optional[float]:
        """提取目标收益率"""
        # "翻倍" = 100%
        if any(w in text for w in ["翻倍", "翻一倍", "double", "2x"]):
            return 100.0
        # "翻两倍" = 200%
        if "翻两倍" in text or "翻三倍" in text:
            m = re.search(r'翻(\w)倍', text)
            if m:
                multiplier_map = {"一": 1, "两": 2, "三": 3, "四": 4}
                mult = multiplier_map.get(m.group(1), 2)
                return mult * 100.0

        # "年化20%" / "收益率20%" / "回报15%"
        m = re.search(r'(?:年化|收益率|回报|回报率|收益|return)\s*(\d+(?:\.\d+)?)\s*%', text)
        if m:
            return float(m.group(1))

        # "想赚20%" / "赚15%"
        m = re.search(r'(?:想赚|赚|盈利|目标|期望|want|target)\s*(\d+(?:\.\d+)?)\s*%', text)
        if m:
            return float(m.group(1))

        # "想赚5万" (绝对金额 -> 转为收益率)
        m = re.search(r'(?:想赚|赚|盈利|目标|期望)\s*(\d+(?:\.\d+)?)\s*(?:万|w|W)\s*(?:美元|块|元)?', text)
        if m and capital:
            profit = float(m.group(1)) * 10000
            return round(profit / capital * 100, 1)

        # "想赚50000" (绝对金额)
        m = re.search(r'(?:想赚|赚|盈利|目标|期望)\s*\$?(\d{4,})', text)
        if m and capital:
            profit = float(m.group(1))
            return round(profit / capital * 100, 1)

        # 纯百分比 "20%"
        m = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if m:
            val = float(m.group(1))
            if 1 <= val <= 500:  # 合理范围 1%~500%
                return val

        return None

    def _extract_timeframe(self, text: str) -> int:
        """提取时间周期 (月), 默认 12"""
        # "3个月" / "半年" / "1年" / "2年"
        if "半年" in text:
            return 6
        m = re.search(r'(\d+)\s*(?:年|year)', text)
        if m:
            return int(m.group(1)) * 12
        m = re.search(r'(\d+)\s*(?:个月|月|month)', text)
        if m:
            return int(m.group(1))
        return 12  # 默认1年

    def _extract_risk_hint(self, text: str) -> str:
        """提取用户风险偏好提示词"""
        return text  # 整句传给分类器做关键词匹配

    # ─── 全自主执行 ───

    def run(self, text: str, dry_run: bool = False) -> str:
        """
        全自主执行目标驱动交易

        流程:
          1. 解析目标
          2. 市场调研 (AI 根据风险等级选板块)
          3. 策略生成 (AI 根据风险等级选模型和因子)
          4. 回测验证 (不达标自动迭代, 最多3轮)
          5. 仓位计算 (根据资金和风险等级分配)
          6. 执行交易 (模拟盘)
          7. 生成大白话投资计划书

        Args:
            text: 用户自然语言输入
            dry_run: True = 只生成计划不下单

        Returns:
            大白话投资计划书 (str)
        """
        from src.agents.registry import agent_registry
        from src.workflow.engine import workflow_engine

        # 延迟初始化
        agent_registry.initialize()
        workflow_engine.initialize()

        # ═══ 1. 解析目标 ═══
        plan = self.parse_goal(text)
        if plan is None:
            return (
                "❌ 无法理解你的投资目标，请试试这样说:\n\n"
                "  • \"我有10万美元，想赚20%\"\n"
                "  • \"5万本金，目标翻倍\"\n"
                "  • \"投10万，年化15%就行，风险别太大\"\n"
                "  • \"10万块，想赚5万\""
            )

        report_lines = []
        report_lines.append(self._format_plan_header(plan))

        # ═══ 2. 市场调研 ═══
        report_lines.append("\n🔍 AI 正在扫描市场...")
        sector_interest = "、".join(plan.sectors)

        research_agent = agent_registry.get_research_agent()
        research_result = research_agent.run(user_interest=sector_interest)

        if research_result is None:
            report_lines.append("❌ 市场调研失败，请检查数据源配置")
            return "\n".join(report_lines)

        report_lines.append(self._format_research(research_agent, plan))

        # ═══ 3. 策略生成 ═══
        report_lines.append("\n🧠 AI 正在生成交易策略...")

        # 从调研结果提取股票池
        candidate_stocks = []
        stock_pool = []
        if hasattr(research_result, "candidate_stocks"):
            candidate_stocks = research_result.candidate_stocks or []
            stock_pool = [c.symbol for c in candidate_stocks if hasattr(c, "symbol")]

        if not stock_pool:
            stock_pool = config.WATCHLIST or config.DEFAULT_WATCHLIST

        strategy_agent = agent_registry.get_strategy_agent()
        strategy_result = strategy_agent.run(
            candidate_stocks=candidate_stocks,
            stock_pool=stock_pool,
            model_type=plan.model_type,
            optimize=True,
        )

        if strategy_result is None:
            report_lines.append("❌ 策略生成失败")
            return "\n".join(report_lines)

        # ═══ 4. 回测验证 (带自动迭代) ═══
        report_lines.append("\n📊 AI 正在回测验证策略是否达标...")

        backtest_agent = agent_registry.get_backtest_agent()
        best_result = None
        best_annual = -999
        iteration_count = 0
        max_iterations = 3
        target_met = False

        while iteration_count < max_iterations:
            iteration_count += 1
            logger.info(f"  📊 回测第 {iteration_count}/{max_iterations} 轮...")

            bt_result = backtest_agent.run(
                strategy_result=strategy_result,
                initial_capital=plan.capital,
            )

            if bt_result is None:
                logger.warning(f"  第 {iteration_count} 轮回测失败")
                break

            # 提取年化收益率
            annual_return = 0.0
            if hasattr(bt_result, "annual_return"):
                annual_return = bt_result.annual_return * 100  # 转为百分比
            elif isinstance(bt_result, dict):
                annual_return = bt_result.get("annual_return", 0) * 100

            logger.info(f"  第 {iteration_count} 轮: 年化 {annual_return:.1f}% (目标 {plan.target_return}%)")

            if annual_return > best_annual:
                best_annual = annual_return
                best_result = bt_result

            # 判断是否达标
            if annual_return >= plan.target_return:
                target_met = True
                report_lines.append(f"  ✅ 第 {iteration_count} 轮回测达标! 年化 {annual_return:.1f}% >= 目标 {plan.target_return}%")
                break
            else:
                gap = plan.target_return - annual_return
                report_lines.append(f"  ⚠️ 第 {iteration_count} 轮: 年化 {annual_return:.1f}%, 差目标 {gap:.1f}%")

                if iteration_count < max_iterations:
                    # 自动迭代: 重新生成策略
                    report_lines.append(f"  🔄 AI 自动调整策略参数, 重新训练...")
                    strategy_result = strategy_agent.run(
                        candidate_stocks=candidate_stocks,
                        stock_pool=stock_pool,
                        model_type=plan.model_type,
                        optimize=True,
                    )

        # 回测总结
        if best_result:
            report_lines.append(self._format_backtest(best_result, plan, target_met, iteration_count))

        # ═══ 5. 仓位计算 ═══
        allocation = self._calculate_allocation(plan, strategy_result)
        report_lines.append(self._format_allocation(allocation, plan))

        # ═══ 6. 执行交易 ═══
        if dry_run:
            report_lines.append("\n📝 演练模式: 不实际下单")
        else:
            report_lines.append("\n💰 AI 正在模拟盘执行建仓...")
            execution_agent = agent_registry.get_execution_agent()
            exec_result = execution_agent.run(strategy_result=strategy_result)

            if exec_result is not None:
                report_lines.append(self._format_execution(exec_result))
            else:
                report_lines.append("  ⚠️ 交易执行未完成 (可能 Alpaca 未配置)")

        # ═══ 7. 风控提示 ═══
        report_lines.append(self._format_risk_warnings(plan))

        # ═══ 尾部 ═══
        report_lines.append(self._format_footer(plan, target_met))

        return "\n".join(report_lines)

    # ─── 仓位计算 ───

    def _calculate_allocation(self, plan: GoalPlan, strategy_result: dict) -> list[dict]:
        """
        根据资金和策略信号计算仓位分配

        策略: 等权分配 (受 max_position 约束)
        每只股票分配金额 = min(capital / N, capital * max_position)
        """
        signals = []
        if strategy_result and isinstance(strategy_result, dict):
            signals = strategy_result.get("signals", [])

        # 过滤只保留 BUY 信号
        buy_signals = [s for s in signals if hasattr(s, "direction") and s.direction.value == "buy"]
        if not buy_signals:
            # 没有信号时使用股票池等权
            strategy_cfg = strategy_result.get("strategy") if strategy_result else None
            if strategy_cfg and hasattr(strategy_cfg, "stock_pool"):
                buy_signals = strategy_cfg.stock_pool[:5]
            else:
                buy_signals = config.WATCHLIST or config.DEFAULT_WATCHLIST[:5]

        n = len(buy_signals)
        if n == 0:
            return []

        # 等权分配 + 单票上限
        equal_weight = 1.0 / n
        capped_weight = min(equal_weight, plan.max_position)

        # 归一化 (因为 cap 可能导致总和 < 1)
        total_weight = capped_weight * n
        if total_weight < 1.0:
            # 有剩余资金，按比例补回去
            scale = 1.0 / total_weight if total_weight > 0 else 1.0
            capped_weight *= scale
            capped_weight = min(capped_weight, plan.max_position)

        allocation = []
        remaining = plan.capital
        for i, sig in enumerate(buy_signals):
            symbol = sig.symbol if hasattr(sig, "symbol") else str(sig)
            amount = plan.capital * capped_weight
            if i == n - 1:
                amount = remaining  # 最后一只拿剩余资金
            remaining -= amount
            allocation.append({
                "symbol": symbol,
                "amount": round(amount, 2),
                "weight": capped_weight,
                "reason": sig.reason if hasattr(sig, "reason") else "AI 策略信号",
            })

        return allocation

    # ─── 通俗报告格式化 ───

    @staticmethod
    def _format_plan_header(plan: GoalPlan) -> str:
        lines = [
            "",
            "╔═══════════════════════════════════════════════════════════╗",
            "║  🤖 AI 自主投资计划书                                      ║",
            "╠═══════════════════════════════════════════════════════════╣",
            f"║  💰 你的目标:  ${plan.capital:,.0f} -> ${plan.target_amount:,.0f}",
            f"║     年化收益:  {plan.target_return}%  (想赚 ${plan.profit_target:,.0f})",
            f"║  📅 投资周期:  {plan.timeframe_months} 个月",
            f"║  ⚡ 风险等级:  {plan.risk_label} ({plan.risk_desc})",
            f"║  🎯 AI 决策:",
            f"║     • 关注板块: {'、'.join(plan.sectors)}",
            f"║     • 策略模型: {plan.model_type.upper()} + {'/'.join(plan.factor_focus)}因子",
            f"║     • 单票仓位: ≤{plan.max_position:.0%}  | 止损: -{plan.stop_loss:.0%}",
            "╚═══════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_research(agent, plan: GoalPlan) -> str:
        try:
            explanation = agent.plain_explain()
            # 截取前 300 字避免过长
            if len(explanation) > 300:
                explanation = explanation[:300] + "..."
            return f"  📰 市场调研完成:\n{explanation}"
        except Exception:
            return "  📰 市场调研完成"

    @staticmethod
    def _format_backtest(bt_result, plan: GoalPlan, target_met: bool, iterations: int) -> str:
        """格式化回测结果"""
        # 兼容对象和 dict
        if hasattr(bt_result, "annual_return"):
            annual = bt_result.annual_return * 100
            sharpe = bt_result.sharpe_ratio
            max_dd = bt_result.max_drawdown * 100
            win_rate = bt_result.win_rate * 100
            grade = bt_result.grade.value if hasattr(bt_result.grade, "value") else str(bt_result.grade)
            final_val = bt_result.final_value
        elif isinstance(bt_result, dict):
            annual = bt_result.get("annual_return", 0) * 100
            sharpe = bt_result.get("sharpe_ratio", 0)
            max_dd = bt_result.get("max_drawdown", 0) * 100
            win_rate = bt_result.get("win_rate", 0) * 100
            grade = bt_result.get("grade", "?")
            final_val = bt_result.get("final_value", 0)
        else:
            return "  📊 回测完成"

        # 通俗解读
        if target_met:
            status = "✅ 达标!"
        else:
            status = f"⚠️ 未完全达标 (经过 {iterations} 轮优化)"

        lines = [
            f"\n  📊 回测验证 ({status}):",
            f"     • 年化收益: {annual:.1f}%  (你的目标: {plan.target_return}%)",
            f"     • 夏普比率: {sharpe:.2f}  (越高越好, >1 不错, >1.5 优秀)",
            f"     • 最大回撤: -{max_dd:.1f}%  (最惨的时候亏这么多)",
            f"     • 胜率:     {win_rate:.0f}%  (赚钱的交易占多少比例)",
            f"     • 策略评级: {grade}",
            f"     • 模拟终值: ${final_val:,.0f}  (用历史数据模拟跑出来的)",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_allocation(allocation: list[dict], plan: GoalPlan) -> str:
        """格式化仓位分配"""
        if not allocation:
            return "\n  💰 资金分配: 暂无信号"

        lines = ["\n  💰 AI 资金分配:"]
        lines.append(f"     {'股票':6s} | {'金额':>12s} | {'占比':>6s} | 推荐理由")
        lines.append(f"     {'─'*55}")

        for i, alloc in enumerate(allocation):
            symbol = alloc["symbol"]
            amount = alloc["amount"]
            weight = alloc["weight"]
            reason = alloc["reason"][:30] if alloc["reason"] else "AI策略信号"
            arrow = "⭐" if i == 0 else "  "
            lines.append(f"  {arrow} {symbol:6s} | ${amount:>10,.0f} | {weight:>5.0%} | {reason}")

        total = sum(a["amount"] for a in allocation)
        lines.append(f"     {'─'*55}")
        lines.append(f"     合计:    ${total:>10,.0f} | {total/plan.capital:>5.0%}")

        return "\n".join(lines)

    @staticmethod
    def _format_execution(exec_result) -> str:
        """格式化执行结果"""
        if isinstance(exec_result, dict):
            executed = exec_result.get("executed", 0)
            total = exec_result.get("total_signals", 0)
            rejected = exec_result.get("rejected", 0)

            lines = [
                f"  ✅ 模拟盘建仓完成!",
                f"     • 执行: {executed}/{total} 笔",
            ]
            if rejected:
                lines.append(f"     • 风控拦截: {rejected} 笔")

            details = exec_result.get("details", [])
            if details:
                lines.append("     • 明细:")
                for d in details[:5]:
                    sym = d.get("symbol", "?")
                    direction = d.get("direction", "?")
                    weight = d.get("target_weight", 0)
                    lines.append(f"       {sym:6s} {direction} (仓位 {weight:.0%})")

            return "\n".join(lines)
        return "  ✅ 模拟盘建仓完成"

    @staticmethod
    def _format_risk_warnings(plan: GoalPlan) -> str:
        """风控提示"""
        lines = [
            "\n  🛡️ AI 设置的风控:",
            f"     • 单票止损: -{plan.stop_loss:.0%}  (亏到这里自动卖)",
            f"     • 组合回撤: -{plan.max_drawdown:.0%}  (整体亏到这里全部清仓)",
            f"     • 单票最大: {plan.max_position:.0%}  (一只股票最多占这么多)",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_footer(plan: GoalPlan, target_met: bool) -> str:
        """尾部提示"""
        lines = [
            "",
            "╔═══════════════════════════════════════════════════════════╗",
        ]
        if target_met:
            lines.append("║  ✅ AI 已根据你的目标完成全自主建仓!                       ║")
        else:
            lines.append("║  ⚠️ 经 3 轮优化未完全达标, 已使用最优策略执行               ║")
            lines.append("║     建议: 适当调整目标收益率或投资周期                      ║")

        lines.extend([
            "║                                                           ║",
            "║  💡 接下来你可以:                                         ║",
            '║     • 输入 "查看持仓" 看实时盈亏                           ║',
            '║     • 输入 "一键全自动" 让 AI 继续调仓                     ║',
            '║     • 输入 "紧急清仓" 随时止损                             ║',
            "║                                                           ║",
            "║  ⚠️ 风险提示: 所有分析仅供参考, 不构成投资建议!            ║",
            "╚═══════════════════════════════════════════════════════════╝",
        ])
        return "\n".join(lines)


# ─── 全局实例 ───
goal_engine = GoalEngine()
