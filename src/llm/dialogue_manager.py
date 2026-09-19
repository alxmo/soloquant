"""
对话上下文管理器
=====================================================
v0.4.0 新增

功能:
- 会话内存: 维护 List[dict] 对话历史
- 上下文窗口: 保留最近 N 轮对话，超出自动裁剪
- 系统状态注入: 将当前系统状态、Agent 结果摘要注入上下文
- 对话历史查询: 供 LLM 生成回复时使用

消息格式 (OpenAI 兼容):
  {"role": "system", "content": "..."}
  {"role": "user", "content": "..."}
  {"role": "assistant", "content": "..."}
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from src.agents.registry import agent_registry
from src.utils.config import config


class DialogueManager:
    """对话上下文管理器"""

    # 最大保留的消息条数 (含 system 消息)
    MAX_MESSAGES = 20

    def __init__(self, max_messages: int = None):
        self._messages: List[dict] = []
        self._max_messages = max_messages or self.MAX_MESSAGES
        self._system_prompt: str = ""
        self._initialized = False
        # v2.0 智能升级: 实体记忆 + 任务追踪
        self._last_intent: str = ""          # 上次执行的意图
        self._entities: Dict[str, Any] = {}  # 记住的实体 (symbols, sector, user_interest 等)
        self._pending_suggestion: str = ""   # 下一步建议文本

    def initialize(self, system_prompt: str = "") -> None:
        """
        初始化对话会话

        Args:
            system_prompt: 系统提示词 (定义 AI 角色、规则等)
        """
        self._messages.clear()
        self._system_prompt = system_prompt or self._build_default_system_prompt()
        self._messages.append({
            "role": "system",
            "content": self._system_prompt,
        })
        self._initialized = True
        logger.debug("对话上下文已初始化")

    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        if not self._initialized:
            self.initialize()
        self._messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        """添加助手回复"""
        if not self._initialized:
            self.initialize()
        self._messages.append({"role": "assistant", "content": content})
        self._trim()

    def add_system_message(self, content: str) -> None:
        """添加系统消息 (如注入状态更新)"""
        self._messages.append({"role": "system", "content": content})
        self._trim()

    def get_context(self, include_system: bool = True) -> List[dict]:
        """
        获取当前对话上下文

        Args:
            include_system: 是否包含 system 消息

        Returns:
            消息列表 (深拷贝，避免外部修改)
        """
        if not self._initialized:
            self.initialize()
        # 深拷贝: 防止外部修改影响内部状态
        import copy
        messages = copy.deepcopy(self._messages)
        if not include_system:
            messages = [m for m in messages if m["role"] != "system"]
        return messages

    def get_recent_user_messages(self, n: int = 3) -> List[str]:
        """获取最近 N 条用户消息"""
        user_msgs = [m["content"] for m in self._messages if m["role"] == "user"]
        return user_msgs[-n:] if n > 0 else []

    def get_last_user_message(self) -> Optional[str]:
        """获取最近一条用户消息"""
        msgs = self.get_recent_user_messages(1)
        return msgs[0] if msgs else None

    def clear(self) -> None:
        """清空对话历史"""
        self._messages.clear()
        self._initialized = False
        self._last_intent = ""
        self._entities = {}
        self._pending_suggestion = ""
        logger.debug("对话上下文已清空")

    def reset(self, system_prompt: str = "") -> None:
        """重置对话 (清空并重新初始化)"""
        self.clear()
        self._last_intent = ""
        self._entities = {}
        self._pending_suggestion = ""
        self.initialize(system_prompt)

    def message_count(self) -> int:
        """返回当前消息数"""
        return len(self._messages)

    def _trim(self) -> None:
        """裁剪消息列表，保留 system 消息 + 最近 N 条对话"""
        if len(self._messages) <= self._max_messages:
            return

        # 分离 system 消息和对话消息
        system_msgs = [m for m in self._messages if m["role"] == "system"]
        conv_msgs = [m for m in self._messages if m["role"] != "system"]

        # 保留所有 system 消息 + 最近的对话消息
        keep_conv = self._max_messages - len(system_msgs)
        if keep_conv < 2:
            keep_conv = 2  # 至少保留 1 轮对话

        conv_msgs = conv_msgs[-keep_conv:]
        self._messages = system_msgs + conv_msgs

    def _build_default_system_prompt(self) -> str:
        """构建默认系统提示词"""
        return (
            "你是一个 AI 量化交易助手，帮助用户理解和使用美股量化交易系统。\n"
            "你的回复应该:\n"
            "- 使用通俗易懂的中文，避免专业术语\n"
            "- 语气亲切，像朋友聊天一样\n"
            "- 在涉及交易时始终提醒风险\n"
            "- 不要编造数据，基于实际结果回复\n"
        )

    def build_system_status_block(self) -> str:
        """
        构建当前系统状态信息块，用于注入对话上下文

        包含: 交易模式、API 配置状态、Agent 执行状态、风控参数
        """
        lines = ["[当前系统状态]"]
        lines.append(f"交易模式: {'模拟盘(Paper)' if config.is_paper_trading() else '实盘(Live)'}")
        lines.append(f"Alpaca: {'已配置' if config.has_alpaca_keys() else '未配置'}")
        lines.append(f"Finnhub: {'已配置' if config.has_finnhub_key() else '未配置'}")
        lines.append(f"最大仓位: {config.MAX_POSITION_PCT:.0%} | 日最大亏损: {config.MAX_DAILY_LOSS:.0%} | 最大回撤: {config.MAX_DRAWDOWN:.0%}")

        # Agent 执行状态
        lines.append("")
        lines.append("[Agent 执行状态]")
        for status in agent_registry.all_status():
            result_icon = "✅" if status["has_result"] else "⬜"
            error_icon = "❌" if status["has_error"] else ""
            lines.append(f"  {result_icon}{error_icon} {status['name']} ({status['role']})")

        return "\n".join(lines)

    def inject_system_status(self) -> None:
        """将当前系统状态注入对话上下文"""
        status_block = self.build_system_status_block()
        self.add_system_message(status_block)

    def get_history_summary(self, n: int = 5) -> str:
        """获取对话历史摘要 (最近 N 条，用于日志)"""
        recent = self._messages[-n:]
        lines = []
        for m in recent:
            role = m["role"]
            content = m["content"][:100]
            if len(m["content"]) > 100:
                content += "..."
            lines.append(f"  [{role}] {content}")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════
    # v2.0 智能升级: 实体记忆 + 上下文感知
    # ═══════════════════════════════════════════════════════

    def update_memory(self, intent: str, params: dict, result: Any = None) -> None:
        """
        意图执行后更新实体记忆

        Args:
            intent: 本次执行的意图
            params: LLM 提取的参数
            result: Agent 执行结果 (可选)
        """
        self._last_intent = intent

        # 提取并记住关键实体
        if params:
            for key in ("symbols", "sector", "user_interest", "keywords", "mode"):
                val = params.get(key)
                if val:  # 非空才更新
                    self._entities[key] = val

        # 从 Agent 结果中提取实体
        if result is not None:
            self._extract_entities_from_result(result)

        logger.debug(
            f"记忆更新: intent={intent}, entities={self._entities}"
        )

    def _extract_entities_from_result(self, result: Any) -> None:
        """从 Agent 执行结果中提取有价值的实体"""
        try:
            # 调研结果 -> 提取股票池
            if hasattr(result, "candidate_stocks"):
                stocks = result.candidate_stocks or []
                symbols = [s.symbol for s in stocks if hasattr(s, "symbol")]
                if symbols:
                    self._entities["last_research_symbols"] = symbols

            # 策略结果 -> 提取策略股票池和信号
            elif hasattr(result, "last_result"):
                last = result.last_result()
                if isinstance(last, dict):
                    if "stock_pool" in last:
                        self._entities["last_strategy_symbols"] = last["stock_pool"]
                    if "signals" in last:
                        self._entities["last_signals_count"] = len(last["signals"])

            # 选股结果 -> 提取选出的股票
            elif hasattr(result, "plain_summary"):
                summary = result.plain_summary or ""
                import re
                symbols = re.findall(r'\b[A-Z]{1,5}\b', summary)
                stop = {"I", "A", "AI", "IT", "OK", "TV", "US", "UK", "EU"}
                symbols = [s for s in symbols if s not in stop]
                if symbols:
                    self._entities["last_screen_symbols"] = symbols[:10]

        except Exception as e:
            logger.debug(f"从结果提取实体失败 (非关键): {e}")

    def get_context_summary(self) -> str:
        """
        获取上下文摘要，供意图识别器使用

        返回上次意图 + 记住的实体，让 LLM 能理解"继续"、"换个股票"等续接语
        """
        lines = []

        if self._last_intent:
            intent_labels = {
                "research": "市场调研", "strategy": "策略生成", "backtest": "回测验证",
                "execute": "执行交易", "monitor": "持仓监控", "stock_screen": "智能选股",
                "full_loop": "全自动循环", "emergency": "紧急止损",
            }
            label = intent_labels.get(self._last_intent, self._last_intent)
            lines.append(f"上次操作: {label}")

        if self._entities:
            entity_parts = []
            for key, val in self._entities.items():
                if isinstance(val, list):
                    val_str = ", ".join(str(v) for v in val[:5])
                    entity_parts.append(f"{key}=[{val_str}]")
                else:
                    entity_parts.append(f"{key}={val}")
            lines.append(f"记住的信息: {'; '.join(entity_parts)}")

        return "\n".join(lines) if lines else ""

    @property
    def last_intent(self) -> str:
        return self._last_intent

    @property
    def entities(self) -> Dict[str, Any]:
        return self._entities.copy()

    @property
    def pending_suggestion(self) -> str:
        return self._pending_suggestion

    @pending_suggestion.setter
    def pending_suggestion(self, value: str) -> None:
        self._pending_suggestion = value


# 全局单例
dialogue_manager = DialogueManager()
