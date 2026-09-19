"""
自然语言交互引擎 - 理解小白用户的意图
=====================================================
职责:
- 解析用户自然语言输入
- 意图识别 (调研/策略/回测/交易/监控/帮助)
- 路由到对应的 Agent 或工作流
- 返回通俗易读的响应

v0.4.0 LLM 智能对话升级:
- LLM 驱动意图识别 (替代关键词匹配)
- LLM 生成自然语言回复 (替代模板拼接)
- 多轮对话上下文记忆
- LLM 不可用时自动降级到规则引擎

v0.2.1 修复:
- P1: 修复意图识别平局 bug
- P4: 修复 _handle_strategy 中 ResearchReport 对象/dict 兼容性问题
"""

import re
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from src.agents.registry import agent_registry
from src.ui.plain_explainer import plain_explainer
from src.utils.config import config
from src.workflow.engine import workflow_engine

# LLM 模块 (v0.4.0)
from src.llm.llm_client import llm_client
from src.llm.llm_config import llm_config
from src.llm.dialogue_manager import dialogue_manager
from src.llm.intent_recognizer import intent_recognizer, LLMIntent, LLMIntentChain
from src.llm.response_generator import response_generator
# v1.4.0: Function-Calling Agent (完全由 LLM 判断意图并调用系统功能)
from src.llm.function_agent import function_agent


class Intent:
    """用户意图"""

    # 意图类型
    RESEARCH = "research"           # 市场调研
    STRATEGY = "strategy"           # 策略生成
    BACKTEST = "backtest"           # 回测
    EXECUTE = "execute"             # 执行交易
    MONITOR = "monitor"             # 监控持仓
    STATUS = "status"               # 系统状态
    HELP = "help"                   # 帮助
    FULL_LOOP = "full_loop"         # 全自动循环
    EMERGENCY = "emergency"         # 紧急止损
    STOCK_SCREEN = "stock_screen"   # 智能选股
    GOAL_DRIVEN = "goal_driven"    # 目标驱动 AI 自主交易
    UNKNOWN = "unknown"             # 未识别


class NLRouter:
    """自然语言意图路由"""

    # 意图关键词映射 (降级模式使用)
    INTENT_KEYWORDS = {
        Intent.GOAL_DRIVEN: [
            "我有", "我投", "投入", "本金", "想赚", "目标", "翻倍",
            "年化", "收益率", "回报", "赚多少", "赚万", "赚k",
            "capital", "invest", "target return", "want to earn",
        ],
        Intent.RESEARCH: [
            "调研", "市场", "新闻", "热点", "机会", "发现", "关注",
            "research", "news", "market", "scan", "explore",
            "看看", "有什么", "值得",
        ],
        Intent.STOCK_SCREEN: [
            "选股", "筛选股票", "推荐股票", "找股票", "挑股票", "选几支", "选几只",
            "潜力股", "低价股", "成长股", "帮选", "帮我选",
            "选支", "选只", "选个", "推荐只", "推荐支", "选3", "选5", "选10",
            "选2", "选4", "选6", "选7", "选8", "选9",
            "pick stock", "screen stock", "find stock", "stock screener",
        ],
        Intent.STRATEGY: [
            "策略", "因子", "模型", "训练", "生成", "优化",
            "strategy", "factor", "model", "train", "optimize",
            "创造", "设计",
        ],
        Intent.BACKTEST: [
            "回测", "测试", "历史", "验证", "表现",
            "backtest", "test", "verify", "historical",
        ],
        Intent.EXECUTE: [
            "执行", "交易", "下单", "买入", "卖出", "交易信号",
            "execute", "trade", "order", "buy", "sell",
            "模拟盘", "paper",
        ],
        Intent.MONITOR: [
            "监控", "持仓", "盈亏", "日报", "周报", "止损",
            "monitor", "position", "portfolio", "pnl", "report",
            "查看", "账户",
        ],
        Intent.STATUS: [
            "状态", "配置", "系统", "设置",
            "status", "config", "system", "settings",
            "当前",
        ],
        Intent.HELP: [
            "帮助", "怎么用", "功能", "可以做什么",
            "help", "usage", "guide",
        ],
        Intent.FULL_LOOP: [
            "全自动", "一键", "全流程", "搞定",
            "auto", "full", "all in one",
        ],
        Intent.EMERGENCY: [
            "紧急", "止损", "清仓", "全部卖出",
            "emergency", "liquidate", "stop loss",
        ],
    }

    # P1 修复: 意图优先级 (得分相同时，优先级高的胜出)
    INTENT_PRIORITY = {
        Intent.EMERGENCY: 100,
        Intent.GOAL_DRIVEN: 95,  # 目标驱动优先级高 (用户有明确投资目标)
        Intent.FULL_LOOP: 90,
        Intent.HELP: 80,
        Intent.STATUS: 70,
        Intent.BACKTEST: 60,
        Intent.EXECUTE: 55,
        Intent.MONITOR: 50,
        Intent.STOCK_SCREEN: 45,
        Intent.STRATEGY: 40,
        Intent.RESEARCH: 30,
        Intent.UNKNOWN: 0,
    }

    def __init__(self):
        self._history: list[dict] = []
        self._llm_checked: bool = False
        self._llm_active: bool = False

    def _check_llm(self) -> bool:
        """检测 LLM 是否可用 (首次调用时检测，之后缓存结果)"""
        if not self._llm_checked:
            llm_config.load()
            self._llm_active = llm_client.is_available()
            self._llm_checked = True
            if self._llm_active:
                # logger.info(f"🧠 LLM 智能对话已启用 (模型: {llm_config.get_default_name()})")
                # 初始化对话上下文
                dialogue_manager.initialize()
            # else:
            #     logger.info("📝 LLM 未配置，对话使用规则引擎模式")
        return self._llm_active

    @property
    def is_llm_active(self) -> bool:
        """当前是否处于 LLM 模式"""
        return self._check_llm()

    def respond(self, text: str, page_context: str = "") -> str:
        """
        处理用户输入，返回响应

        v0.4.0: 优先使用 LLM 意图识别和回复生成，
        LLM 不可用时自动降级到关键词匹配 + 模板回复。

        Args:
            text: 用户自然语言输入
            page_context: 用户当前查看的页面内容摘要 (v1.1.0 页面感知)

        Returns:
            通俗易读的响应文本
        """
        text_lower = text.lower().strip()

        # 记录对话历史
        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "user": text,
        })

        # 检测 LLM 可用性
        use_llm = self._check_llm()

        if use_llm:
            response = self._respond_with_llm(text, page_context)
        else:
            response = self._respond_with_rules(text_lower, text)

        # 记录响应
        self._history[-1]["response"] = response
        self._history[-1]["llm_mode"] = use_llm

        return response

    # ═══════════════════════════════════════════════════════
    # LLM 模式 (v0.4.0)
    # ═══════════════════════════════════════════════════════

    def _respond_with_llm(self, text: str, page_context: str = "") -> str:
        """LLM 模式 (v1.4.0: Function-Calling Agent)

        完全由 LLM 判断意图并选择工具调用，无关键词匹配。
        收集 Agent 流式输出，返回完整文本。

        Args:
            text: 用户输入文本
            page_context: 当前页面内容摘要 (v1.1.0 页面感知)
        """
        full_response = ""
        try:
            for chunk in function_agent.run_stream(text, page_context):
                if chunk["type"] == "content":
                    full_response += chunk["text"]
        except Exception as e:
            logger.error(f"Function Agent 执行异常: {e}", exc_info=True)
            full_response = f"⚠️ 处理请求时出现异常: {e}\n\n请稍后重试。"

        if not full_response:
            full_response = "抱歉，我暂时没有生成有效回复，请换个说法试试。"

        return full_response

    def _route_intent(
        self, intent: str, text: str, params: dict
    ) -> tuple:
        """
        根据意图路由到对应 Agent，返回 (result, role, error)

        Returns:
            (agent_result, agent_role, error_message)
        """
        try:
            if intent == Intent.RESEARCH:
                return self._exec_research(text, params)
            elif intent == Intent.STRATEGY:
                return self._exec_strategy(text, params)
            elif intent == Intent.BACKTEST:
                return self._exec_backtest(text, params)
            elif intent == Intent.EXECUTE:
                return self._exec_execute(text, params)
            elif intent == Intent.MONITOR:
                return self._exec_monitor(text, params)
            elif intent == Intent.STATUS:
                return self._exec_status(text, params)
            elif intent == Intent.FULL_LOOP:
                return self._exec_full_loop(text, params)
            elif intent == Intent.EMERGENCY:
                return self._exec_emergency(text, params)
            elif intent == Intent.STOCK_SCREEN:
                return self._exec_stock_screen(text, params)
            elif intent == Intent.GOAL_DRIVEN:
                return self._exec_goal_driven(text, params)
            elif intent == Intent.HELP:
                return (self._help_text(), "帮助", None)
            else:
                # unknown: 闲聊或通用问题
                return (None, "", None)
        except Exception as e:
            logger.error(f"意图路由异常 [{intent}]: {e}")
            return (None, "", str(e))

    # ─── LLM 模式下的 Agent 执行方法 ───
    # 返回 (result, role, error) 三元组

    def _exec_research(self, text: str, params: dict) -> tuple:
        """执行市场调研"""
        agent = agent_registry.get_research_agent()
        user_interest = params.get("user_interest", "")
        result = agent.run(user_interest=user_interest)
        if result is None:
            return (None, agent.role, "调研失败，请检查数据源配置")
        return (agent, agent.role, None)  # 返回 agent 实例，回复生成器会调用 plain_explain

    def _exec_strategy(self, text: str, params: dict) -> tuple:
        """执行策略生成"""
        ctx = agent_registry.context
        research_result = ctx.get("research_result")

        # LLM 提取的股票代码优先
        symbols = params.get("symbols", [])

        if research_result is None and not symbols:
            # 没有调研结果也没有指定股票，使用配置的关注列表
            stock_pool = config.WATCHLIST if config.WATCHLIST else config.DEFAULT_WATCHLIST
            candidate_stocks = []
        elif symbols:
            stock_pool = symbols
            candidate_stocks = []
        else:
            candidate_stocks = self._extract_candidates(research_result)
            stock_pool = self._extract_stock_pool(research_result)

        agent = agent_registry.get_strategy_agent()
        result = agent.run(
            candidate_stocks=candidate_stocks,
            stock_pool=stock_pool,
        )
        if result is None:
            return (None, agent.role, "策略生成失败")
        return (agent, agent.role, None)

    def _exec_backtest(self, text: str, params: dict) -> tuple:
        """执行回测"""
        ctx = agent_registry.context
        strategy_result = ctx.get("strategy_result")

        if strategy_result is None:
            return (None, "回测验证师", "请先生成交易策略，然后才能回测")

        agent = agent_registry.get_backtest_agent()
        result = agent.run(strategy_result=strategy_result)
        if result is None:
            return (None, agent.role, "回测失败")
        return (agent, agent.role, None)

    def _exec_execute(self, text: str, params: dict) -> tuple:
        """执行交易"""
        ctx = agent_registry.context
        strategy_result = ctx.get("strategy_result")

        if strategy_result is None:
            return (None, "交易执行员", "请先生成交易策略，然后才能执行交易")

        agent = agent_registry.get_execution_agent()
        result = agent.run(strategy_result=strategy_result)
        if result is None:
            return (None, agent.role, "交易执行失败")
        return (agent, agent.role, None)

    def _exec_monitor(self, text: str, params: dict) -> tuple:
        """执行监控"""
        agent = agent_registry.get_monitor_agent()
        mode = params.get("mode", "")
        if not mode:
            mode = "report" if any(kw in text for kw in ["日报", "周报", "报告"]) else "snapshot"
        result = agent.run(mode=mode)
        if result is None:
            return (None, agent.role, "监控失败，请检查 Alpaca 配置")
        return (agent, agent.role, None)

    def _exec_status(self, text: str, params: dict) -> tuple:
        """系统状态"""
        return (self._status_text(), "系统状态", None)

    def _exec_full_loop(self, text: str, params: dict) -> tuple:
        """全自动循环"""
        result = workflow_engine.run_daily_loop()
        # 返回 workflow 结果字符串
        summary = self._format_workflow_result(result, "全自动循环")
        return (summary, "全自动循环", None)

    def _exec_emergency(self, text: str, params: dict) -> tuple:
        """紧急止损"""
        result = workflow_engine.run_emergency_stop()
        summary = self._format_workflow_result(result, "紧急止损")
        return (summary, "紧急止损", None)

    def _exec_stock_screen(self, text: str, params: dict) -> tuple:
        """智能选股"""
        from src.engine.stock_screener import stock_screener

        # 补全默认参数
        screen_params = {
            "count": params.get("count", 5),
            "price_min": params.get("price_min", 0),
            "price_max": params.get("price_max", 0),
            "sector": params.get("sector", ""),
            "keywords": params.get("keywords", ""),
        }

        result = stock_screener.screen(screen_params, query=text)
        if result is None:
            return (None, "智能选股", "选股失败，请检查数据源配置")
        return (result, "智能选股", None)

    def _exec_goal_driven(self, text: str, params: dict) -> tuple:
        """
        执行目标驱动 AI 自主交易
        用户说 "我有10万美元想赚20%" -> AI 全自主完成全流程
        """
        from src.engine.goal_engine import goal_engine

        try:
            report = goal_engine.run(text)
            return (report, "AI自主交易", None)
        except Exception as e:
            logger.error(f"目标驱动交易失败: {e}")
            return (None, "AI自主交易", f"执行失败: {e}")

    @staticmethod
    def _format_workflow_result(result: dict, name: str) -> str:
        """格式化工作流结果为文本摘要"""
        lines = [f"✅ {name}完成!"]
        steps = result.get("steps", {})
        results = result.get("results", {})
        for step_name, step_status in steps.items():
            icon = "✅" if step_status == "success" else "❌"
            lines.append(f"  {icon} {step_name}: {step_status}")
        return "\n".join(lines)

    def _generate_fallback_response(
        self, intent: str, agent_result: Any,
        agent_role: str, error: Optional[str], text: str
    ) -> str:
        """LLM 回复生成失败时的模板降级"""
        if error:
            return f"❌ {error}"

        if agent_result is None and intent == Intent.UNKNOWN:
            return self._handle_unknown(text)

        # 调用 plain_explain
        if hasattr(agent_result, "plain_explain"):
            try:
                return agent_result.plain_explain()
            except Exception:
                pass

        if isinstance(agent_result, str) and agent_result:
            return agent_result

        return f"{agent_role or '系统'}执行完成。"

    # ═══════════════════════════════════════════════════════
    # 规则引擎模式 (降级，保持原有逻辑)
    # ═══════════════════════════════════════════════════════

    def _respond_with_rules(self, text_lower: str, text: str) -> str:
        """规则引擎模式: 关键词匹配 + 模板回复 (原有逻辑)"""

        # 识别意图
        intent = self._detect_intent(text_lower)

        # 记录意图
        self._history[-1]["intent"] = intent

        # 路由到对应处理函数
        handler = {
            Intent.RESEARCH: self._handle_research,
            Intent.STRATEGY: self._handle_strategy,
            Intent.BACKTEST: self._handle_backtest,
            Intent.EXECUTE: self._handle_execute,
            Intent.MONITOR: self._handle_monitor,
            Intent.STATUS: self._handle_status,
            Intent.HELP: self._help_text,
            Intent.FULL_LOOP: self._handle_full_loop,
            Intent.EMERGENCY: self._handle_emergency,
            Intent.STOCK_SCREEN: self._handle_stock_screen,
            Intent.GOAL_DRIVEN: self._handle_goal_driven,
            Intent.UNKNOWN: self._handle_unknown,
        }.get(intent, self._handle_unknown)

        return handler(text)

    def _detect_intent(self, text: str) -> str:
        """
        检测用户意图 (关键词匹配，降级模式使用)

        P1 修复: 当多个意图得分相同时，按预定义优先级排序
        """
        scores = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[intent] = score

        if not scores:
            return Intent.UNKNOWN

        max_score = max(scores.values())
        top_intents = [i for i, s in scores.items() if s == max_score]

        if len(top_intents) == 1:
            return top_intents[0]

        return max(top_intents, key=lambda i: self.INTENT_PRIORITY.get(i, 0))

    def _handle_research(self, text: str) -> str:
        """处理市场调研意图"""
        try:
            agent = agent_registry.get_research_agent()
            result = agent.run()

            if result is None:
                return (
                    "❌ 调研失败，请检查数据源配置。\n\n"
                    "💡 可能的原因:\n"
                    "  • Finnhub API Key 未配置\n"
                    "  • 网络连接问题\n\n"
                    "你可以在 .env 文件中配置 FINNHUB_API_KEY。"
                )

            return agent.plain_explain()

        except Exception as e:
            logger.error(f"调研失败: {e}")
            return f"❌ 调研过程中出现错误: {e}\n\n请稍后重试。"

    def _handle_strategy(self, text: str) -> str:
        """处理策略生成意图"""
        ctx = agent_registry.context
        research_result = ctx.get("research_result")

        if research_result is None:
            default_pool = config.WATCHLIST if config.WATCHLIST else config.DEFAULT_WATCHLIST
            candidate_stocks = []
            stock_pool = default_pool
        else:
            candidate_stocks = self._extract_candidates(research_result)
            stock_pool = self._extract_stock_pool(research_result)

        try:
            agent = agent_registry.get_strategy_agent()
            result = agent.run(
                candidate_stocks=candidate_stocks,
                stock_pool=stock_pool,
            )

            if result is None:
                return "❌ 策略生成失败，请先进行市场调研或指定股票池。"

            return agent.plain_explain()

        except Exception as e:
            logger.error(f"策略生成失败: {e}")
            return f"❌ 策略生成过程中出现错误: {e}"

    def _handle_backtest(self, text: str) -> str:
        """处理回测意图"""
        ctx = agent_registry.context
        strategy_result = ctx.get("strategy_result")

        if strategy_result is None:
            return "⚠️ 请先生成交易策略 (输入 \"帮我生成一个策略\")，然后才能回测。"

        try:
            agent = agent_registry.get_backtest_agent()
            result = agent.run(strategy_result=strategy_result)

            if result is None:
                return "❌ 回测失败，可能是数据不足或策略配置有误。"

            return agent.plain_explain()

        except Exception as e:
            logger.error(f"回测失败: {e}")
            return f"❌ 回测过程中出现错误: {e}"

    def _handle_execute(self, text: str) -> str:
        """处理交易执行意图"""
        ctx = agent_registry.context
        strategy_result = ctx.get("strategy_result")

        if strategy_result is None:
            return "⚠️ 请先生成交易策略，然后才能执行交易。"

        try:
            agent = agent_registry.get_execution_agent()
            result = agent.run(strategy_result=strategy_result)

            if result is None:
                return "❌ 交易执行失败。"

            return agent.plain_explain()

        except Exception as e:
            logger.error(f"交易执行失败: {e}")
            return f"❌ 交易执行过程中出现错误: {e}"

    def _handle_monitor(self, text: str) -> str:
        """处理监控意图"""
        try:
            agent = agent_registry.get_monitor_agent()
            mode = "report" if any(kw in text for kw in ["日报", "周报", "报告"]) else "snapshot"
            result = agent.run(mode=mode)

            if result is None:
                return "❌ 监控失败，请检查 Alpaca 配置。"

            return agent.plain_explain()

        except Exception as e:
            logger.error(f"监控失败: {e}")
            return f"❌ 监控过程中出现错误: {e}"

    def _handle_status(self, text: str) -> str:
        """处理系统状态意图"""
        return self._status_text()

    def _status_text(self) -> str:
        """生成系统状态文本"""
        lines = []
        lines.append("📊 系统状态")
        lines.append(f"{'─' * 40}")
        lines.append(f"  交易模式: {'📝 模拟盘 (Paper)' if config.is_paper_trading() else '🔴 实盘 (Live)'}")
        lines.append(f"  Alpaca:   {'✅ 已配置' if config.has_alpaca_keys() else '❌ 未配置'}")
        lines.append(f"  Finnhub:  {'✅ 已配置' if config.has_finnhub_key() else '❌ 未配置'}")
        lines.append(f"  Qlib数据: {'✅ 已初始化' if config.QLIB_DATA_PATH.exists() else '⚠️ 未初始化'}")
        lines.append(f"  最大仓位: {config.MAX_POSITION_PCT:.0%}")
        lines.append(f"  日最大亏损: {config.MAX_DAILY_LOSS:.0%}")
        lines.append(f"  最大回撤: {config.MAX_DRAWDOWN:.0%}")

        # LLM 状态 (v0.4.0)
        llm_status = llm_config.status()
        if llm_status["enabled"]:
            lines.append(f"  LLM: 🧠 智能模式 (模型: {llm_status['default_model']})")
        elif llm_status["total_models"] > 0:
            lines.append(f"  LLM: 📝 规则模式 (已配置但Key未填)")
        else:
            lines.append(f"  LLM: 📝 规则模式 (未配置)")

        # Agent 状态
        lines.append("")
        lines.append("🤖 Agent 状态:")
        for status in agent_registry.all_status():
            result_icon = "✅" if status["has_result"] else "⬜"
            error_icon = "❌" if status["has_error"] else "✅"
            lines.append(f"  {result_icon}{error_icon} {status['name']} ({status['role']})")

        # 工作流状态
        lines.append("")
        lines.append("🔄 可用工作流:")
        for wf_name in workflow_engine.list_workflows():
            lines.append(f"  • {wf_name}")

        return "\n".join(lines)

    def _handle_full_loop(self, text: str) -> str:
        """处理全自动循环意图"""
        try:
            lines = []
            lines.append("🚀 启动全自动循环...")
            lines.append("  这可能需要几分钟，请耐心等待。")
            lines.append("")

            result = workflow_engine.run_daily_loop()

            lines.append(f"✅ 全自动循环完成!")
            lines.append("")

            for step_name, step_result in result.get("results", {}).items():
                if step_result is None:
                    continue
                agent = agent_registry.get(
                    "research" if step_name == "research" else
                    "strategy" if step_name == "strategy" else
                    "backtest" if step_name == "backtest" else
                    "execution" if step_name == "execution" else
                    "monitor"
                )
                if agent and agent.last_result() is not None:
                    lines.append(f"📋 {agent.role}报告:")
                    lines.append(agent.plain_explain())
                    lines.append("")

            return "\n".join(lines)

        except Exception as e:
            return f"❌ 全自动循环失败: {e}"

    def _handle_stock_screen(self, text: str) -> str:
        """处理智能选股意图 (降级模式)"""
        import re
        from src.engine.stock_screener import stock_screener

        # 降级模式: 用正则提取参数
        params = {"count": 5, "price_min": 0, "price_max": 0, "sector": "", "keywords": ""}

        count_match = re.search(r'(?:选|推荐|找|挑)\s*(\d+)\s*(?:支|只|个|股)', text)
        if count_match:
            params["count"] = int(count_match.group(1))

        # 价格区间优先
        price_range_match = re.search(r'(\d+)\s*[-到~]\s*(\d+)\s*(?:美元|元|\$)', text)
        if price_range_match:
            params["price_min"] = float(price_range_match.group(1))
            params["price_max"] = float(price_range_match.group(2))
        else:
            price_max_match = re.search(r'(\d+)\s*(?:美元|元|\$)\s*(?:以下|以内|下)|低于\s*(\d+)', text)
            if price_max_match:
                params["price_max"] = float(price_max_match.group(1) or price_max_match.group(2))
            else:
                price_min_match = re.search(r'(\d+)\s*(?:美元|元|\$)\s*以上|高于\s*(\d+)', text)
                if price_min_match:
                    params["price_min"] = float(price_min_match.group(1) or price_min_match.group(2))
                else:
                    # 纯 "N美元" 作为价格上限
                    price_bare_match = re.search(r'(\d+)\s*(?:美元|元|\$)', text)
                    if price_bare_match:
                        params["price_max"] = float(price_bare_match.group(1))

        if "潜力" in text:
            params["keywords"] = "潜力"
        elif "成长" in text:
            params["keywords"] = "成长"

        try:
            result = stock_screener.screen(params, query=text)
            return result.plain_summary
        except Exception as e:
            logger.error(f"选股失败: {e}")
            return f"❌ 选股过程中出现错误: {e}"

    def _handle_goal_driven(self, text: str) -> str:
        """处理目标驱动 AI 自主交易意图 (降级模式)"""
        try:
            from src.engine.goal_engine import goal_engine
            return goal_engine.run(text)
        except Exception as e:
            logger.error(f"目标驱动交易失败: {e}")
            return f"❌ AI 自主交易执行失败: {e}\n\n请稍后重试。"

    def _handle_emergency(self, text: str) -> str:
        """处理紧急止损意图"""
        try:
            lines = []
            lines.append("🚨 启动紧急止损流程...")
            lines.append("")

            result = workflow_engine.run_emergency_stop()

            lines.append(f"✅ 紧急止损完成!")
            lines.append("")

            step_agent_map = {
                "risk_check": "monitor",
                "liquidate": "execution",
                "notify": "monitor",
            }
            for step_name, step_result in result.get("results", {}).items():
                if step_result is None:
                    continue
                agent_name = step_agent_map.get(step_name)
                if agent_name:
                    agent = agent_registry.get(agent_name)
                    if agent and agent.last_result() is not None:
                        lines.append(f"📋 {agent.role}报告:")
                        lines.append(agent.plain_explain())
                        lines.append("")

            return "\n".join(lines)

        except Exception as e:
            return f"❌ 紧急止损失败: {e}"

    def _handle_unknown(self, text: str) -> str:
        """处理未识别的输入"""
        return (
            f"🤔 我不太理解你的意思。\n\n"
            f"你可以试试这样说:\n"
            f"  • \"我有10万美元，想赚20%\" -> AI 全自主交易\n"
            f"  • \"帮我选5支10美元的潜力股\" -> 智能选股\n"
            f"  • \"帮我看看今天美股有什么机会\" -> 市场调研\n"
            f"  • \"帮我生成一个交易策略\" -> 策略生成\n"
            f"  • \"回测一下策略\" -> 回测验证\n"
            f"  • \"执行交易\" -> 执行信号\n"
            f"  • \"查看持仓\" -> 监控持仓\n"
            f"  • \"一键全自动\" -> 全流程自动运转\n"
            f"  • \"紧急清仓\" -> 紧急止损\n"
            f"  • \"帮助\" -> 查看所有功能\n"
        )

    def _help_text(self, *args, **kwargs) -> str:
        """帮助文本"""
        return (
            "🤖 AI 量化交易助手 - 使用指南\n"
            f"{'═' * 50}\n\n"
            "📋 我能帮你做这些事:\n\n"
            "  🎯 目标驱动 AI 自主交易 (最省心!)\n"
            "     \"我有10万美元，想赚20%\"\n"
            "     \"5万本金，目标翻倍，风险别太大\"\n"
            "     \"10万块，想赚5万\"\n"
            "     -> AI 全自主: 选股/选策略/训练/回测/下单\n\n"
            "  1️⃣ 智能选股\n"
            "     \"帮我选5支10美元的潜力股\"\n"
            "     \"选3只20-50美元的科技股\"\n\n"
            "  2️⃣ 市场调研\n"
            "     \"帮我看看今天有什么机会\"\n"
            "     \"扫描一下市场新闻\"\n\n"
            "  3️⃣ 策略生成\n"
            "     \"帮我生成一个交易策略\"\n"
            "     \"为 AAPL 和 MSFT 创建策略\"\n\n"
            "  4️⃣ 回测验证\n"
            "     \"回测一下策略\"\n"
            "     \"测试策略的历史表现\"\n\n"
            "  5️⃣ 执行交易\n"
            "     \"执行交易信号\"\n"
            "     \"用模拟盘下单\"\n\n"
            "  6️⃣ 持仓监控\n"
            "     \"查看持仓\"\n"
            "     \"今天的交易日报\"\n\n"
            "  7️⃣ 全自动模式\n"
            "     \"一键全自动\"\n"
            "     \"帮我搞定全流程\"\n\n"
            "  8️⃣ 紧急止损\n"
            "     \"紧急清仓\"\n"
            "     \"全部卖出\"\n\n"
            "  9️⃣ 系统状态\n"
            "     \"系统状态\"\n"
            "     \"当前配置\"\n\n"
            "💡 提示: 用大白话告诉我想做什么就行，我会理解你的意思！\n"
            "⚠️ 风险提示: 所有分析仅供参考，不构成投资建议。"
        )

    # ─── P4 修复: 辅助方法，统一处理 ResearchReport 对象和 dict ───

    @staticmethod
    def _extract_candidates(research_result: Any) -> list:
        """从调研结果中提取候选股票列表，兼容对象和 dict"""
        if hasattr(research_result, "candidate_stocks"):
            return research_result.candidate_stocks or []
        if isinstance(research_result, dict):
            return research_result.get("candidates") or research_result.get("candidate_stocks") or []
        return []

    @staticmethod
    def _extract_stock_pool(research_result: Any) -> list:
        """从调研结果中提取股票池，兼容对象和 dict"""
        if hasattr(research_result, "candidate_stocks"):
            candidates = research_result.candidate_stocks or []
            return [c.symbol for c in candidates if hasattr(c, "symbol")]
        if isinstance(research_result, dict):
            stock_pool = research_result.get("stock_pool")
            if stock_pool:
                return stock_pool
            candidates = research_result.get("candidates") or research_result.get("candidate_stocks") or []
            return [c.get("symbol", "") for c in candidates if isinstance(c, dict) and c.get("symbol")]
        return []

    # ═══════════════════════════════════════════════════════
    # 流式响应 (v0.5.0)
    # ═══════════════════════════════════════════════════════

    def respond_stream(self, text: str, page_context: str = ""):
        """
        流式处理用户输入，逐块 yield 回复

        流程:
        1. 意图识别 (同步)
        2. Agent 执行 (同步)
        3. 流式生成回复 (逐块 yield)

        Args:
            text: 用户输入文本
            page_context: 用户当前查看的页面内容摘要 (v1.1.0 页面感知)

        Yields:
            dict: {"type": "thinking"/"content", "text": "..."}
        """
        text_lower = text.lower().strip()

        # 记录对话历史
        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "user": text,
        })

        use_llm = self._check_llm()

        if use_llm:
            yield from self._respond_stream_with_llm(text, page_context)
        else:
            # 降级模式: 同步执行后一次性返回
            response = self._respond_with_rules(text_lower, text)
            yield {"type": "content", "text": response}

    def _respond_stream_with_llm(self, text: str, page_context: str = ""):
        """
        LLM 模式流式响应 (v1.4.0: Function-Calling Agent)

        完全由 LLM 判断意图并选择工具调用，无关键词匹配。
        Agent 推理循环: LLM 思考 -> 工具调用 -> 结果回传 -> 最终回复
        """
        # v1.4.0: 主路径 = Function Agent (LLM 自主决策工具调用)
        try:
            full_response = ""
            for chunk in function_agent.run_stream(text, page_context):
                yield chunk
                if chunk["type"] == "content":
                    full_response += chunk["text"]

            # 记录响应
            self._history[-1]["response"] = full_response
            self._history[-1]["llm_mode"] = True
            return
        except Exception as e:
            # Agent 循环内部已处理大部分异常，此处兜底
            logger.error(f"Function Agent 执行异常: {e}", exc_info=True)
            yield {
                "type": "content",
                "text": f"⚠️ 处理请求时出现异常: {e}\n\n请稍后重试。",
            }
            self._history[-1]["response"] = f"异常: {e}"
            self._history[-1]["llm_mode"] = True
            return

    def _execute_chain_stream(self, chain: LLMIntentChain, text: str, page_context: str = ""):
        """
        v2.0: 多意图链式执行 (已废弃主路径，保留供降级模式复用)

        v1.4.0: LLM 模式下多步操作由 Function Agent 自主完成，
        此方法仅保留给规则引擎降级模式使用。
        """
        total = len(chain.intents)
        dialogue_manager.add_user_message(text)
        full_response_parts = []

        for idx, llm_intent in enumerate(chain.intents, 1):
            intent = llm_intent.intent
            params = llm_intent.params

            intent_labels = {
                "research": "市场调研", "strategy": "策略生成", "backtest": "回测验证",
                "execute": "执行交易", "monitor": "持仓监控", "status": "系统状态",
                "help": "帮助", "full_loop": "全自动循环", "emergency": "紧急止损",
                "stock_screen": "智能选股", "goal_driven": "AI自主交易", "unknown": "处理请求",
            }
            label = intent_labels.get(intent, intent)

            yield {"type": "thinking", "text": f"⚡ [{idx}/{total}] {label}中，请稍候..."}

            agent_result, agent_role, error = self._route_intent(intent, text, params)

            # 更新记忆
            dialogue_manager.update_memory(intent, params, agent_result)

            # 生成回复 (非流式，避免嵌套流式复杂度)
            try:
                reply = response_generator.generate(
                    user_text=text,
                    intent=intent,
                    agent_result=agent_result,
                    agent_role=agent_role,
                    error=error,
                    page_context=page_context,
                )
            except Exception as e:
                logger.warning(f"链式步骤 {idx} 回复生成失败: {e}")
                reply = self._generate_fallback_response(
                    intent, agent_result, agent_role, error, text
                )

            # 步骤标题
            if total > 1:
                step_header = f"**步骤 {idx}/{total}: {label}**\n\n"
                reply = step_header + reply

            yield {"type": "content", "text": reply}
            full_response_parts.append(reply)

            # 如果出错，中断链
            if error:
                yield {"type": "content", "text": f"\n\n⚠️ 链中断: {error}"}
                break

        full_response = "\n\n---\n\n".join(full_response_parts)
        dialogue_manager.add_assistant_message(full_response)

        # 推送最终跟进建议
        last_intent = chain.intents[-1].intent
        suggestions = self._generate_follow_up_suggestions(last_intent, None, None)
        if suggestions:
            yield {"type": "suggestions", "suggestions": suggestions}

        self._history[-1]["response"] = full_response
        self._history[-1]["llm_mode"] = True

    def _generate_follow_up_suggestions(self, intent: str, agent_result: Any = None,
                                        error: Optional[str] = None) -> list:
        """
        v2.0: 根据当前意图生成跟进建议按钮

        Returns:
            建议列表, 每项为 {"text": "显示文本", "action": "发送的消息"}
        """
        if error:
            return [
                {"text": "🔄 重试", "action": "重试"},
                {"text": "❓ 帮助", "action": "帮助"},
            ]

        suggestions_map = {
            "research": [
                {"text": "📊 生成策略", "action": "帮我生成策略"},
                {"text": "🔍 换个板块", "action": "换个板块调研"},
                {"text": "📈 查看持仓", "action": "查看持仓"},
            ],
            "stock_screen": [
                {"text": "📊 调研这些股票", "action": "调研这些股票"},
                {"text": "🔄 重新选股", "action": "重新选股"},
                {"text": "📈 生成策略", "action": "帮我生成策略"},
            ],
            "strategy": [
                {"text": "📈 回测验证", "action": "回测一下策略"},
                {"text": "🚀 执行交易", "action": "执行交易"},
                {"text": "🔄 优化策略", "action": "优化策略"},
            ],
            "backtest": [
                {"text": "🚀 执行交易", "action": "执行交易"},
                {"text": "🔄 重新生成策略", "action": "重新生成策略"},
                {"text": "📊 查看持仓", "action": "查看持仓"},
            ],
            "execute": [
                {"text": "📊 查看持仓", "action": "查看持仓"},
                {"text": "📰 生成日报", "action": "今天的交易日报"},
                {"text": "🚨 紧急止损", "action": "紧急清仓"},
            ],
            "monitor": [
                {"text": "📰 生成日报", "action": "今天的交易日报"},
                {"text": "🚀 执行交易", "action": "执行交易"},
                {"text": "🔍 市场调研", "action": "帮我看看今天有什么机会"},
            ],
            "status": [
                {"text": "🔍 市场调研", "action": "帮我看看今天有什么机会"},
                {"text": "📊 查看持仓", "action": "查看持仓"},
                {"text": "❓ 帮助", "action": "帮助"},
            ],
            "help": [
                {"text": "🔍 市场调研", "action": "帮我看看今天有什么机会"},
                {"text": "🎯 AI自主交易", "action": "我有10万美元，想赚20%"},
                {"text": "📊 查看持仓", "action": "查看持仓"},
            ],
            "full_loop": [
                {"text": "📊 查看持仓", "action": "查看持仓"},
                {"text": "📰 生成日报", "action": "今天的交易日报"},
            ],
            "goal_driven": [
                {"text": "📊 查看持仓", "action": "查看持仓"},
                {"text": "📰 生成日报", "action": "今天的交易日报"},
            ],
            "unknown": [
                {"text": "❓ 帮助", "action": "帮助"},
                {"text": "🔍 市场调研", "action": "帮我看看今天有什么机会"},
            ],
        }

        return suggestions_map.get(intent, [])

    @property
    def history(self) -> list[dict]:
        """获取对话历史"""
        return self._history

    def reset_dialogue(self) -> None:
        """重置对话上下文 (v0.4.0)"""
        dialogue_manager.reset()
        self._history.clear()
        self._llm_checked = False
        self._llm_active = False
        logger.info("对话上下文已重置")

    def load_session(self, messages: list) -> None:
        """v1.3.0: 加载历史会话消息, 恢复 LLM 对话上下文

        将持久化的会话消息回放到 dialogue_manager, 使切换会话后
        AI 能"记住"该会话之前的对话内容。

        Args:
            messages: [{"role": "user"/"assistant", "content": "..."}]
        """
        try:
            # 重置现有上下文
            dialogue_manager.reset()
            self._history.clear()
            self._llm_checked = False
            self._llm_active = False

            if not isinstance(messages, list):
                return

            # 回放最近的消息 (dialogue_manager 自带窗口裁剪)
            restored = 0
            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                content = m.get("content") or ""
                if not content:
                    continue
                if role == "user":
                    dialogue_manager.add_user_message(content)
                    restored += 1
                elif role == "assistant":
                    dialogue_manager.add_assistant_message(content)
                    restored += 1

            if restored:
                logger.info(f"会话上下文已恢复 ({restored} 条消息)")
        except Exception as e:
            logger.warning(f"恢复会话上下文失败: {e}")


# 全局实例
nl_router = NLRouter()
