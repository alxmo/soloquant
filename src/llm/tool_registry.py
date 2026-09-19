"""
系统功能工具注册表 (Function-Calling Tool Registry)
=====================================================
v1.4.0 新增

功能:
- 将系统所有功能封装为 LLM 可调用的"工具" (Tool)
- 每个工具 = 名称 + 描述 + JSON Schema 参数 + 执行函数
- LLM 通过 JSON 协议自主选择工具并传参，实现自然语言驱动全部系统功能

工具调用协议 (JSON):
  LLM 输出:
    {"tool": "工具名", "arguments": {...}}
  或最终回复:
    {"reply": "给用户的最终回复文本"}

设计原则:
- 执行函数复用现有 Agent / Engine / Collector 代码，不重复实现业务逻辑
- 工具返回通俗文本 (plain_explain)，供 LLM 直接理解并转述
"""

from typing import Any, Callable, Optional

from loguru import logger


# i18n: 工具中文标签 -> 英文标签 (供英文 UI 的 thinking 进度提示)
TOOL_LABEL_EN = {
    "市场调研": "Market Research",
    "智能选股": "Stock Screening",
    "策略生成": "Strategy Generation",
    "回测验证": "Backtest",
    "执行交易": "Trade Execution",
    "持仓监控": "Position Monitoring",
    "查询行情": "Get Quotes",
    "查询个股新闻": "Stock News",
    "查询市场新闻": "Market News",
    "查询策略列表": "List Strategies",
    "运行工作流": "Run Workflow",
    "全自动循环": "Full Auto Loop",
    "紧急止损": "Emergency Stop",
    "AI自主交易": "AI Auto-Trading",
    "查询系统状态": "System Status",
    "查看帮助": "Help",
}


class ToolSpec:
    """工具规格定义"""

    def __init__(self, name: str, description: str, parameters: dict,
                 func: Callable, label: str = ""):
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON Schema
        self.func = func              # 同步执行函数: func(**kwargs) -> str
        self.label = label or name    # 中文标签 (供 thinking 提示)
        self.label_en = TOOL_LABEL_EN.get(label, label)  # 英文标签 (i18n)

    def to_schema(self) -> dict:
        """转为 LLM 提示词中的工具描述"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, parameters: dict,
                 func: Callable, label: str = "") -> None:
        """注册工具"""
        self._tools[name] = ToolSpec(name, description, parameters, func, label)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict]:
        """所有工具的 schema 列表 (注入 LLM 系统提示词)"""
        return [t.to_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> str:
        """
        执行工具，返回结果文本

        Args:
            name: 工具名
            arguments: LLM 传的参数

        Returns:
            工具执行结果文本 (出错时返回错误说明，不抛异常)
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"错误: 未知工具 '{name}'。可用工具: {', '.join(self._tools.keys())}"

        try:
            # 过滤掉 schema 未定义的参数，防止意外传参
            props = tool.parameters.get("properties", {})
            filtered = {k: v for k, v in (arguments or {}).items() if k in props}
            result = tool.func(**filtered)
            if result is None:
                return "工具执行完成 (无返回内容)"
            return str(result)
        except Exception as e:
            logger.error(f"工具 {name} 执行失败: {e}", exc_info=True)
            return f"工具执行失败: {e}"


# ═══════════════════════════════════════════════════════
# 工具实现 (复用现有 Agent / Engine / Collector)
# ═══════════════════════════════════════════════════════

def _tool_market_research(user_interest: str = "") -> str:
    """市场调研工具"""
    from src.agents.registry import agent_registry
    agent = agent_registry.get_research_agent()
    result = agent.run(user_interest=user_interest)
    if result is None:
        return "调研失败，请检查数据源配置 (Finnhub API Key)"
    return agent.plain_explain()


def _tool_stock_screen(count: int = 5, price_min: float = 0, price_max: float = 0,
                       sector: str = "", keywords: str = "", query: str = "") -> str:
    """智能选股工具"""
    from src.engine.stock_screener import stock_screener
    params = {
        "count": int(count) if count else 5,
        "price_min": float(price_min or 0),
        "price_max": float(price_max or 0),
        "sector": sector or "",
        "keywords": keywords or "",
    }
    result = stock_screener.screen(params, query=query or keywords or "")
    if result is None:
        return "选股失败，请检查数据源配置"
    return result.plain_summary


def _tool_generate_strategy(symbols: list = None) -> str:
    """策略生成工具"""
    from src.agents.registry import agent_registry
    from src.utils.config import config

    ctx = agent_registry.context
    research_result = ctx.get("research_result")
    symbols = symbols or []

    if research_result is None and not symbols:
        stock_pool = config.WATCHLIST if config.WATCHLIST else config.DEFAULT_WATCHLIST
        candidate_stocks = []
    elif symbols:
        stock_pool = symbols
        candidate_stocks = []
    else:
        candidate_stocks = _extract_candidates(research_result)
        stock_pool = _extract_stock_pool(research_result)

    agent = agent_registry.get_strategy_agent()
    result = agent.run(candidate_stocks=candidate_stocks, stock_pool=stock_pool)
    if result is None:
        return "策略生成失败"
    return agent.plain_explain()


def _tool_backtest() -> str:
    """回测验证工具"""
    from src.agents.registry import agent_registry
    ctx = agent_registry.context
    strategy_result = ctx.get("strategy_result")
    if strategy_result is None:
        return "请先生成交易策略 (调用 generate_strategy)，然后才能回测"

    agent = agent_registry.get_backtest_agent()
    result = agent.run(strategy_result=strategy_result)
    if result is None:
        return "回测失败，可能是数据不足或策略配置有误"
    return agent.plain_explain()


def _tool_execute_trade() -> str:
    """执行交易工具"""
    from src.agents.registry import agent_registry
    ctx = agent_registry.context
    strategy_result = ctx.get("strategy_result")
    if strategy_result is None:
        return "请先生成交易策略 (调用 generate_strategy)，然后才能执行交易"

    agent = agent_registry.get_execution_agent()
    result = agent.run(strategy_result=strategy_result)
    if result is None:
        return "交易执行失败"
    return agent.plain_explain()


def _tool_monitor_positions(mode: str = "snapshot") -> str:
    """持仓监控工具"""
    from src.agents.registry import agent_registry
    agent = agent_registry.get_monitor_agent()
    mode = "report" if mode == "report" else "snapshot"
    result = agent.run(mode=mode)
    if result is None:
        return "监控失败，请检查 Alpaca 配置"
    return agent.plain_explain()


def _tool_get_stock_quote(symbol: str) -> str:
    """实时行情查询工具"""
    from src.data.collector import data_collector
    quote = data_collector.get_realtime_quote(symbol)
    if not quote:
        return f"未获取到 {symbol} 的实时行情，请检查股票代码是否正确"

    # 兼容不同数据源的字段名
    price = quote.get("c") or quote.get("price") or quote.get("current_price")
    change = quote.get("dp") or quote.get("change_percent") or quote.get("change_pct")
    change_abs = quote.get("d") or quote.get("change")
    high = quote.get("h") or quote.get("day_high")
    low = quote.get("l") or quote.get("day_low")
    open_ = quote.get("o") or quote.get("open")

    lines = [f"{symbol} 实时行情:"]
    if price is not None:
        lines.append(f"  当前价格: ${price}")
    if change is not None:
        icon = "📈" if (change or 0) >= 0 else "📉"
        lines.append(f"  涨跌幅: {icon} {change}%")
    if change_abs is not None:
        lines.append(f"  涨跌额: ${change_abs}")
    if open_ is not None:
        lines.append(f"  今开: ${open_}")
    if high is not None:
        lines.append(f"  最高: ${high}")
    if low is not None:
        lines.append(f"  最低: ${low}")
    return "\n".join(lines)


def _tool_get_stock_news(symbol: str, days: int = 7) -> str:
    """个股新闻查询工具"""
    from src.data.collector import data_collector
    news = data_collector.get_company_news(symbol, days=int(days) if days else 7)
    if not news:
        return f"未获取到 {symbol} 的近期新闻"

    lines = [f"{symbol} 近 {days} 天新闻 (最多显示10条):"]
    for item in news[:10]:
        title = item.get("headline") or item.get("title") or ""
        source = item.get("source") or ""
        summary = (item.get("summary") or "")[:100]
        lines.append(f"  • {title} ({source})")
        if summary:
            lines.append(f"    {summary}")
    return "\n".join(lines)


def _tool_get_market_news(category: str = "general", count: int = 10) -> str:
    """市场新闻查询工具"""
    from src.data.collector import data_collector
    news = data_collector.get_market_news(
        category=category or "general", count=int(count) if count else 10
    )
    if not news:
        return "未获取到市场新闻"

    lines = ["最新市场新闻:"]
    for item in news[:10]:
        title = item.get("headline") or item.get("title") or ""
        source = item.get("source") or ""
        lines.append(f"  • {title} ({source})")
    return "\n".join(lines)


def _tool_list_strategies() -> str:
    """策略列表查询工具"""
    from src.engine.strategy_manager import strategy_manager
    strategies = strategy_manager.list_strategies()
    if not strategies:
        return "当前没有已保存的策略"

    lines = [f"共 {len(strategies)} 个策略:"]
    for s in strategies:
        sid = getattr(s, "strategy_id", "?")
        name = getattr(s, "name", "?")
        status = getattr(s, "status", "")
        lines.append(f"  • {name} (ID: {sid}, 状态: {status})")
    return "\n".join(lines)


def _tool_run_workflow(name: str = "daily_trading_loop", user_interest: str = "") -> str:
    """运行工作流工具"""
    from src.workflow.engine import workflow_engine
    try:
        result = workflow_engine.run(name, user_interest=user_interest)
    except Exception as e:
        return f"工作流 {name} 运行失败: {e}"

    lines = [f"✅ 工作流 {name} 完成!"]
    steps = result.get("steps", {}) if isinstance(result, dict) else {}
    for step_name, step_status in steps.items():
        icon = "✅" if step_status == "success" else "❌"
        lines.append(f"  {icon} {step_name}: {step_status}")
    return "\n".join(lines)


def _tool_full_auto_loop(user_interest: str = "") -> str:
    """全自动循环工具 (调研->策略->回测->执行->监控)"""
    from src.workflow.engine import workflow_engine
    result = workflow_engine.run_daily_loop(user_interest=user_interest)
    lines = ["✅ 全自动循环完成!"]
    steps = result.get("steps", {}) if isinstance(result, dict) else {}
    for step_name, step_status in steps.items():
        icon = "✅" if step_status == "success" else "❌"
        lines.append(f"  {icon} {step_name}: {step_status}")
    return "\n".join(lines)


def _tool_emergency_stop() -> str:
    """紧急止损工具"""
    from src.workflow.engine import workflow_engine
    result = workflow_engine.run_emergency_stop()
    lines = ["✅ 紧急止损完成!"]
    steps = result.get("steps", {}) if isinstance(result, dict) else {}
    for step_name, step_status in steps.items():
        icon = "✅" if step_status == "success" else "❌"
        lines.append(f"  {icon} {step_name}: {step_status}")
    return "\n".join(lines)


def _tool_goal_driven_trading(goal_text: str) -> str:
    """目标驱动 AI 自主交易工具"""
    from src.engine.goal_engine import goal_engine
    return goal_engine.run(goal_text)


def _tool_system_status() -> str:
    """系统状态查询工具"""
    from src.ui.nl_router import nl_router
    return nl_router._status_text()


def _tool_get_help() -> str:
    """帮助工具"""
    from src.ui.nl_router import nl_router
    return nl_router._help_text()


# ─── 辅助: 从调研结果提取候选股票 (兼容对象/dict) ───

def _extract_candidates(research_result: Any) -> list:
    if hasattr(research_result, "candidate_stocks"):
        return research_result.candidate_stocks or []
    if isinstance(research_result, dict):
        return research_result.get("candidates") or research_result.get("candidate_stocks") or []
    return []


def _extract_stock_pool(research_result: Any) -> list:
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
# 注册所有工具
# ═══════════════════════════════════════════════════════

def _build_registry() -> ToolRegistry:
    """构建并注册所有系统工具"""
    reg = ToolRegistry()

    # ─── 交易核心流程 ───
    reg.register(
        name="market_research",
        description="市场调研: 扫描市场新闻、热点、寻找投资机会。用户想了解市场动态、今天有什么机会时使用。",
        parameters={
            "type": "object",
            "properties": {
                "user_interest": {"type": "string", "description": "用户关注的行业/主题，如 '科技'、'新能源'，为空则全市场扫描"},
            },
            "required": [],
        },
        func=_tool_market_research,
        label="市场调研",
    )

    reg.register(
        name="stock_screen",
        description="智能选股: 按数量/价格区间/行业/关键词筛选股票。用户说'选5支10美元的潜力股'、'推荐3只科技股'时使用。",
        parameters={
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "选股数量，默认5"},
                "price_min": {"type": "number", "description": "最低价格，0=不限"},
                "price_max": {"type": "number", "description": "最高价格，0=不限"},
                "sector": {"type": "string", "description": "行业偏好，如 '科技'、'能源'、'医疗'"},
                "keywords": {"type": "string", "description": "选股关键词，如 '潜力'、'成长'、'低价'"},
                "query": {"type": "string", "description": "用户的原始选股需求描述"},
            },
            "required": [],
        },
        func=_tool_stock_screen,
        label="智能选股",
    )

    reg.register(
        name="generate_strategy",
        description="策略生成: 为股票池创建交易策略 (因子分析+模型训练)。用户想'生成策略'、'创建策略'、'为XX股票设计策略'时使用。",
        parameters={
            "type": "object",
            "properties": {
                "symbols": {"type": "array", "items": {"type": "string"},
                            "description": "股票代码列表，如 ['AAPL','MSFT']。为空则使用调研结果或默认关注列表"},
            },
            "required": [],
        },
        func=_tool_generate_strategy,
        label="策略生成",
    )

    reg.register(
        name="backtest_strategy",
        description="回测验证: 用历史数据测试已生成策略的表现。用户说'回测'、'测试策略'、'看看历史表现'时使用。需要先生成策略。",
        parameters={"type": "object", "properties": {}, "required": []},
        func=_tool_backtest,
        label="回测验证",
    )

    reg.register(
        name="execute_trade",
        description="执行交易: 按已生成策略的交易信号下单 (模拟盘)。用户说'执行'、'下单'、'买入'、'交易'时使用。需要先生成策略。",
        parameters={"type": "object", "properties": {}, "required": []},
        func=_tool_execute_trade,
        label="执行交易",
    )

    reg.register(
        name="monitor_positions",
        description="持仓监控: 查看当前持仓、盈亏、账户状态。用户说'查看持仓'、'盈亏'、'账户'、'日报'时使用。",
        parameters={
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["snapshot", "report"],
                         "description": "snapshot=快速快照, report=详细日报"},
            },
            "required": [],
        },
        func=_tool_monitor_positions,
        label="持仓监控",
    )

    # ─── 数据查询 ───
    reg.register(
        name="get_stock_quote",
        description="实时行情: 查询单只股票的当前价格、涨跌幅、最高最低价。用户问'XX多少钱'、'XX现在什么价'、'XX涨了多少'时使用。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码，如 'AAPL'"},
            },
            "required": ["symbol"],
        },
        func=_tool_get_stock_quote,
        label="查询行情",
    )

    reg.register(
        name="get_stock_news",
        description="个股新闻: 查询某只股票的近期新闻。用户说'看看XX的新闻'、'XX最近有什么消息'时使用。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码，如 'AAPL'"},
                "days": {"type": "integer", "description": "查询天数，默认7天"},
            },
            "required": ["symbol"],
        },
        func=_tool_get_stock_news,
        label="查询个股新闻",
    )

    reg.register(
        name="get_market_news",
        description="市场新闻: 查询最新市场新闻。用户说'今天有什么新闻'、'市场动态'时使用。",
        parameters={
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["general", "forex", "crypto", "merger"],
                             "description": "新闻类别，默认 general"},
                "count": {"type": "integer", "description": "新闻条数，默认10"},
            },
            "required": [],
        },
        func=_tool_get_market_news,
        label="查询市场新闻",
    )

    reg.register(
        name="list_strategies",
        description="策略列表: 查看系统中已保存的所有交易策略。用户说'我有哪些策略'、'看看策略'时使用。",
        parameters={"type": "object", "properties": {}, "required": []},
        func=_tool_list_strategies,
        label="查询策略列表",
    )

    # ─── 工作流 ───
    reg.register(
        name="run_workflow",
        description="运行指定工作流。用户明确要求运行某个工作流时使用。",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "工作流名称，如 'daily_trading_loop'、'strategy_iteration'、'emergency_stop'"},
                "user_interest": {"type": "string", "description": "用户关注的行业/主题 (可选)"},
            },
            "required": ["name"],
        },
        func=_tool_run_workflow,
        label="运行工作流",
    )

    reg.register(
        name="full_auto_loop",
        description="全自动循环: 一键完成 调研->策略->回测->执行->监控 全流程。用户说'一键全自动'、'全流程搞定'时使用。耗时较长。",
        parameters={
            "type": "object",
            "properties": {
                "user_interest": {"type": "string", "description": "用户关注的行业/主题 (可选)"},
            },
            "required": [],
        },
        func=_tool_full_auto_loop,
        label="全自动循环",
    )

    reg.register(
        name="emergency_stop",
        description="紧急止损: 立即清仓全部持仓。⚠️ 仅在用户明确要求'紧急清仓'、'全部卖出'、'止损'时使用，调用前务必确认用户意图。",
        parameters={"type": "object", "properties": {}, "required": []},
        func=_tool_emergency_stop,
        label="紧急止损",
    )

    reg.register(
        name="goal_driven_trading",
        description="目标驱动 AI 自主交易: 用户给出本金和收益目标 (如'我有10万美元想赚20%')，AI 全自主完成选股/策略/回测/下单全流程。",
        parameters={
            "type": "object",
            "properties": {
                "goal_text": {"type": "string", "description": "用户的完整目标描述原文"},
            },
            "required": ["goal_text"],
        },
        func=_tool_goal_driven_trading,
        label="AI自主交易",
    )

    # ─── 系统信息 ───
    reg.register(
        name="system_status",
        description="系统状态: 查看交易模式、API配置、风控参数、Agent状态。用户问'系统状态'、'当前配置'时使用。",
        parameters={"type": "object", "properties": {}, "required": []},
        func=_tool_system_status,
        label="查询系统状态",
    )

    reg.register(
        name="get_help",
        description="帮助: 查看系统功能使用指南。用户说'帮助'、'怎么用'、'你能做什么'时使用。",
        parameters={"type": "object", "properties": {}, "required": []},
        func=_tool_get_help,
        label="查看帮助",
    )

    return reg


# 全局单例 (延迟构建，避免 import 时触发重依赖)
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """获取工具注册表单例"""
    global _registry
    if _registry is None:
        _registry = _build_registry()
    return _registry


# 供 LLM 系统提示词使用的工具清单文本
def tools_prompt() -> str:
    """生成工具清单描述文本 (注入 LLM 系统提示词)"""
    reg = get_tool_registry()
    lines = []
    for t in reg._tools.values():
        params_desc = []
        for pname, pschema in t.parameters.get("properties", {}).items():
            ptype = pschema.get("type", "any")
            pdesc = pschema.get("description", "")
            params_desc.append(f"{pname}({ptype}): {pdesc}")
        params_str = "; ".join(params_desc) if params_desc else "无参数"
        lines.append(f"- {t.name}: {t.description}\n  参数: {params_str}")
    return "\n".join(lines)
