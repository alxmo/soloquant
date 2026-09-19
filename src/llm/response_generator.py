"""
LLM 回复生成器
=====================================================
v0.4.0 新增

功能:
- 接收 Agent 执行结果 + 对话上下文，生成自然语言回复
- System Prompt 内嵌: 系统角色、当前状态、Agent 数据、通俗解读规则
- 风格控制: 小白友好、通俗语言、风险提示、不堆砌原始数据
- 降级策略: LLM 不可用时回退到 agent.plain_explain() 模板
"""

from typing import Any, Optional

from loguru import logger

from src.llm.llm_client import llm_client
from src.llm.dialogue_manager import dialogue_manager


class ResponseGenerator:
    """LLM 回复生成器"""

    # 角色 System Prompt (v2.0 增强: 人设 + Few-shot + 结构化格式)
    SYSTEM_PROMPT = (
        "你是「小Q」，一个资深美股量化分析师兼耐心导师，服务于一个 AI 量化交易系统。\n\n"
        "你的性格: 专业但不端着，像一个懂行的朋友在跟你聊投资。用通俗易懂的中文，"
        "把复杂的量化概念翻译成日常用语。\n\n"
        "## 回复结构要求\n"
        "每次回复按以下结构组织 (但不要生硬地标注标题):\n"
        "1. 一句话摘要: 先给结论，让用户秒懂\n"
        "2. 关键发现: 2-3 个要点，挑最重要的说\n"
        "3. 行动建议: 1-2 条具体可操作的建议\n"
        "4. 风险提示: 涉及交易时必须提醒 (简洁一句话)\n\n"
        "## 回复规则\n"
        "1. 语气亲切自然，像朋友聊天，适当使用 emoji 但不要过多\n"
        "2. 不要堆砌原始数据，挑重点说，数据用通俗方式表达\n"
        "3. 涉及收益时加风险提示: '过去表现不代表未来收益'\n"
        "4. 如果数据缺失或异常，如实告知，不要编造\n"
        "5. 回复控制在 400 字以内，简洁有力\n"
        "6. 不要使用 markdown 一级标题 (#)，用 **粗体** 或换行即可\n"
        "7. 如果提供了「当前页面内容」，请结合页面内容回答用户问题\n"
        "8. 如果用户在续接之前的对话，请参考上下文保持连贯\n\n"
        "## 回复示例\n\n"
        "示例1 (市场调研):\n"
        "用户: 帮我看看今天有什么机会\n"
        "回复: 今天市场整体偏暖，科技股领涨 🔥\n\n"
        "**关键发现:**\n"
        "• AAPL 新品发布会临近，市场预期较高\n"
        "• NVDA AI 芯片需求持续旺盛，技术面强势\n"
        "• 半导体板块整体资金流入明显\n\n"
        "**建议:** 可以关注 AAPL 和 NVDA 的短线机会，\n"
        "要不要我帮你生成一个交易策略？\n\n"
        "⚠️ 以上分析仅供参考，不构成投资建议。\n\n"
        "示例2 (回测结果):\n"
        "用户: 回测一下策略\n"
        "回复: 策略回测完成，表现还不错 👍\n\n"
        "**核心指标:**\n"
        "• 年化收益: 18.5%，跑赢大盘\n"
        "• 最大回撤: -8.2%，风险可控\n"
        "• 夏普比率: 1.3，性价比不错\n\n"
        "**建议:** 策略整体稳健，可以考虑执行交易。\n"
        "如果想更保守，可以调低仓位比例。\n\n"
        "⚠️ 历史表现不代表未来收益，请注意风险管理。\n"
    )

    def generate(
        self,
        user_text: str,
        intent: str,
        agent_result: Any = None,
        agent_role: str = "",
        error: Optional[str] = None,
        page_context: str = "",
    ) -> str:
        """
        生成 LLM 回复

        Args:
            user_text: 用户原始输入
            intent: 识别到的意图
            agent_result: Agent 执行结果 (可为 None)
            agent_role: Agent 角色描述 (如 "市场调研员")
            error: 执行错误信息 (如有)
            page_context: 用户当前查看的页面内容摘要 (v1.1.0 页面感知)

        Returns:
            回复文本
        """
        # 尝试 LLM 生成
        if llm_client.is_available():
            try:
                return self._generate_with_llm(user_text, intent, agent_result, agent_role, error, page_context)
            except Exception as e:
                logger.warning(f"LLM 回复生成失败，降级到模板: {e}")

        # 降级到模板回复
        return self._generate_fallback(user_text, intent, agent_result, agent_role, error)

    def _generate_with_llm(
        self,
        user_text: str,
        intent: str,
        agent_result: Any,
        agent_role: str,
        error: Optional[str],
        page_context: str = "",
    ) -> str:
        """使用 LLM 生成回复"""

        # 构建上下文消息
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # 注入系统状态
        status_block = dialogue_manager.build_system_status_block()
        messages.append({"role": "system", "content": status_block})

        # v2.0: 注入实体记忆 + 上次意图
        context_summary = dialogue_manager.get_context_summary()
        if context_summary:
            messages.append({
                "role": "system",
                "content": f"[对话记忆]\n{context_summary}\n请参考以上记忆保持对话连贯。"
            })

        # v1.1.0: 注入页面上下文
        if page_context:
            messages.append({
                "role": "system",
                "content": f"用户当前正在查看的页面内容如下，用户可能在询问关于该页面的问题，请结合页面内容回答:\n\n{page_context}"
            })

        # 注入对话历史 (最近几轮，不含当前用户消息)
        history = dialogue_manager.get_context(include_system=False)
        # 排除最后一条 (当前用户消息)，取最近 6 条
        if len(history) > 1:
            history = history[:-1]
        history = history[-6:]
        messages.extend(history)

        # 构建当前请求的用户消息
        result_summary = self._summarize_result(agent_result, agent_role, error, intent)

        user_content = f"用户说: {user_text}\n\n"
        if result_summary:
            user_content += f"[系统执行结果]\n{result_summary}\n\n"
        else:
            user_content += "[系统执行结果]\n(无执行结果，可能是闲聊或通用问题)\n\n"
        user_content += "请根据以上信息，用通俗语言回复用户。按「摘要->关键发现->建议->风险提示」的结构组织回复。"

        messages.append({"role": "user", "content": user_content})

        reply = llm_client.chat(messages, temperature=0.7)
        logger.debug(f"LLM 回复生成成功 ({len(reply)} 字)")
        return reply

    def _summarize_result(
        self,
        agent_result: Any,
        agent_role: str,
        error: Optional[str],
        intent: str,
    ) -> str:
        """
        将 Agent 结果序列化为 LLM 可读的文本摘要

        策略:
        1. 优先调用对象的 plain_explain() / plain_summary
        2. 如果是 dict，提取关键字段
        3. 如果是字符串，直接使用
        """
        if error:
            return f"执行出错: {error}"

        if agent_result is None:
            return ""

        lines = []

        # 尝试调用 plain_explain 方法 (Agent 实例)
        if hasattr(agent_result, "plain_explain"):
            try:
                # plain_explain 可能需要 result 参数，传 None 回退到 _last_result
                text = agent_result.plain_explain()
                if isinstance(text, str) and text:
                    lines.append(text)
            except Exception:
                pass

        # 尝试 plain_summary 属性 (ResearchReport 等)
        if hasattr(agent_result, "plain_summary"):
            try:
                text = getattr(agent_result, "plain_summary")
                if isinstance(text, str) and text and text not in lines:
                    lines.append(text)
            except Exception:
                pass

        # 如果是 dict (如 strategy 结果)
        if isinstance(agent_result, dict):
            text = agent_result.get("plain_summary", "")
            if isinstance(text, str) and text and text not in lines:
                lines.append(text)

            # 补充关键字段
            if "signals" in agent_result and agent_result["signals"]:
                lines.append(f"交易信号数: {len(agent_result['signals'])}")
            if "factors" in agent_result and agent_result["factors"]:
                lines.append(f"有效因子数: {len(agent_result['factors'])}")

        # 如果是字符串
        if isinstance(agent_result, str):
            lines.append(agent_result)

        return "\n".join(lines) if lines else str(agent_result)[:500]

    def _generate_fallback(
        self,
        user_text: str,
        intent: str,
        agent_result: Any,
        agent_role: str,
        error: Optional[str],
    ) -> str:
        """降级: 使用模板回复"""

        if error:
            return f"执行过程中出现了错误: {error}\n\n请稍后重试。"

        if agent_result is None:
            return self._fallback_unknown(user_text)

        # 尝试调用 plain_explain
        if hasattr(agent_result, "plain_explain"):
            try:
                text = agent_result.plain_explain()
                if isinstance(text, str) and text:
                    return text
            except Exception:
                pass

        # 尝试 plain_summary
        if hasattr(agent_result, "plain_summary"):
            text = getattr(agent_result, "plain_summary", "")
            if isinstance(text, str) and text:
                return text

        # dict 类型
        if isinstance(agent_result, dict):
            text = agent_result.get("plain_summary", "")
            if isinstance(text, str) and text:
                return text

        # 字符串
        if isinstance(agent_result, str) and agent_result:
            return agent_result

        return f"{agent_role or '系统'}执行完成，结果已生成。"

    @staticmethod
    def _fallback_unknown(user_text: str) -> str:
        """未识别意图的降级回复"""
        return (
            "我不太理解你的意思。\n\n"
            "你可以试试这样说:\n"
            "  • \"帮我选5支10美元的潜力股\" -> 智能选股\n"
            "  • \"帮我看看今天美股有什么机会\" -> 市场调研\n"
            "  • \"帮我生成一个交易策略\" -> 策略生成\n"
            "  • \"回测一下策略\" -> 回测验证\n"
            "  • \"执行交易\" -> 执行信号\n"
            "  • \"查看持仓\" -> 监控持仓\n"
            "  • \"一键全自动\" -> 全流程自动运转\n"
            "  • \"紧急清仓\" -> 紧急止损\n"
            "  • \"帮助\" -> 查看所有功能\n"
        )


    # ═══════════════════════════════════════════════════════
    # 流式生成 (v0.5.0)
    # ═══════════════════════════════════════════════════════

    def generate_stream(
        self,
        user_text: str,
        intent: str,
        agent_result: Any = None,
        agent_role: str = "",
        error: Optional[str] = None,
        page_context: str = "",
    ):
        """
        流式生成 LLM 回复，逐块 yield 文本

        流程:
        1. 先 yield 思考过程摘要 (意图+执行状态)
        2. 再 yield LLM 回复正文 (逐字)

        Args:
            同 generate()
            page_context: 用户当前查看的页面内容摘要 (v1.1.0)

        Yields:
            dict: {"type": "thinking"/"content", "text": "..."}
        """
        # 1. 先输出思考过程
        thinking = self._build_thinking(intent, agent_role, error)
        if thinking:
            yield {"type": "thinking", "text": thinking}

        # 2. 尝试 LLM 流式生成
        if llm_client.is_available():
            try:
                yield from self._generate_stream_with_llm(
                    user_text, intent, agent_result, agent_role, error, page_context
                )
                return
            except Exception as e:
                logger.warning(f"LLM 流式回复生成失败，降级到模板: {e}")

        # 3. 降级到模板回复 (一次性输出)
        fallback = self._generate_fallback(user_text, intent, agent_result, agent_role, error)
        yield {"type": "content", "text": fallback}

    def _generate_stream_with_llm(
        self,
        user_text: str,
        intent: str,
        agent_result: Any,
        agent_role: str,
        error: Optional[str],
        page_context: str = "",
    ):
        """使用 LLM 流式生成回复 (v2.0: 注入实体记忆)"""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        status_block = dialogue_manager.build_system_status_block()
        messages.append({"role": "system", "content": status_block})

        # v2.0: 注入实体记忆 + 上次意图
        context_summary = dialogue_manager.get_context_summary()
        if context_summary:
            messages.append({
                "role": "system",
                "content": f"[对话记忆]\n{context_summary}\n请参考以上记忆保持对话连贯。"
            })

        # v1.1.0: 注入页面上下文
        if page_context:
            messages.append({
                "role": "system",
                "content": f"用户当前正在查看的页面内容如下，用户可能在询问关于该页面的问题，请结合页面内容回答:\n\n{page_context}"
            })

        history = dialogue_manager.get_context(include_system=False)
        if len(history) > 1:
            history = history[:-1]
        history = history[-6:]
        messages.extend(history)

        result_summary = self._summarize_result(agent_result, agent_role, error, intent)

        user_content = f"用户说: {user_text}\n\n"
        if result_summary:
            user_content += f"[系统执行结果]\n{result_summary}\n\n"
        else:
            user_content += "[系统执行结果]\n(无执行结果，可能是闲聊或通用问题)\n\n"
        user_content += "请根据以上信息，用通俗语言回复用户。按「摘要->关键发现->建议->风险提示」的结构组织回复。"

        messages.append({"role": "user", "content": user_content})

        # 流式获取 LLM 回复
        for chunk in llm_client.chat_stream(messages, temperature=0.7):
            yield {"type": "content", "text": chunk}

    def _build_thinking(self, intent: str, agent_role: str, error: Optional[str]) -> str:
        """构建思考过程摘要"""
        intent_labels = {
            "research": "市场调研",
            "strategy": "策略生成",
            "backtest": "回测验证",
            "execute": "执行交易",
            "monitor": "持仓监控",
            "status": "系统状态",
            "help": "帮助",
            "full_loop": "全自动循环",
            "emergency": "紧急止损",
            "stock_screen": "智能选股",
            "unknown": "理解意图",
        }

        label = intent_labels.get(intent, intent)

        if error:
            return f"🔍 {label} → 执行出错: {error}"
        elif intent == "unknown":
            return f"🤔 思考中..."
        elif agent_role:
            return f"🔍 {label} → {agent_role}执行完成，正在生成回复..."
        else:
            return f"🔍 {label} → 正在思考回复..."

    def is_llm_mode(self) -> bool:
        """当前是否使用 LLM 模式"""
        return llm_client.is_available()


# 全局单例
response_generator = ResponseGenerator()
