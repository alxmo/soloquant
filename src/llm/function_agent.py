"""
LLM Function-Calling Agent 推理循环
=====================================================
v1.4.0 新增

功能:
- 完全由 LLM 判断用户意图并选择工具 (无关键词匹配)
- JSON 协议工具调用: LLM 输出 {"tool": "...", "arguments": {...}}
- 多轮推理循环: LLM 可根据工具结果继续调用其他工具，直到给出最终回复
- 流式输出: 每步 yield thinking 事件，最终回复逐块 yield content

推理循环流程:
  用户输入
    ↓
  LLM 思考 (系统提示词: 工具清单 + 对话上下文 + 实体记忆)
    ↓
  ├─ 输出 {"tool": ..., "arguments": ...} → 执行工具 → 结果回传 LLM → 继续循环
  └─ 输出 {"reply": ...} → 最终回复，结束循环

安全限制:
- 最大迭代次数 MAX_ITERATIONS (防死循环)
- emergency_stop 工具调用前需 LLM 确认用户意图 (提示词约束)
"""

import json
def _get_ui_lang():
    """获取 UI 语言 (zh/en), 失败回退中文"""
    try:
        from src.ui.i18n import get_language
        return get_language()
    except Exception:
        return "zh"

from loguru import logger

from src.llm.llm_client import LLMCallError, llm_client
from src.llm.dialogue_manager import dialogue_manager
from src.llm.tool_registry import get_tool_registry


class FunctionAgent:
    """LLM Function-Calling Agent"""

    # 最大推理迭代次数 (每次迭代 = 一次 LLM 调用)
    MAX_ITERATIONS = 6

    def __init__(self):
        self._registry = None

    @property
    def registry(self):
        """工具注册表 (延迟初始化)"""
        if self._registry is None:
            self._registry = get_tool_registry()
        return self._registry

    # ═══════════════════════════════════════════════════════
    # 系统提示词构建
    # ═══════════════════════════════════════════════════════

    def _build_system_prompt(self, page_context: str = "") -> str:
        """构建 Agent 系统提示词"""
        from src.llm.tool_registry import tools_prompt

        parts = [
            "你是一个美股量化交易系统的 AI 助手，通过调用工具来完成用户的请求。",
            "",
            "## 可用工具",
            tools_prompt(),
            "",
            "## 工具调用协议 (JSON)",
            "每次回复必须输出以下两种 JSON 之一 (只输出 JSON，不要其他文字):",
            "",
            '1. 调用工具:',
            '{"tool": "工具名", "arguments": {"参数名": "参数值"}}',
            "",
            '2. 给出最终回复 (不需要再调用工具时):',
            '{"reply": "给用户的最终回复文本"}',
            "",
            "## 决策规则",
            "1. 用户请求涉及系统功能时，必须调用对应工具获取真实数据，禁止编造数据",
            "2. 一句话包含多个操作时 (如'调研AAPL然后生成策略再回测')，逐个调用工具，",
            "   每次调用一个，根据上一步结果决定下一步",
            "3. 工具结果不足以回答时，可以继续调用其他工具补充信息",
            "4. 闲聊、通用问题、与交易无关的对话，直接用 reply 回复，不调用工具",
            "5. emergency_stop (紧急清仓) 是高危操作，仅当用户明确表达清仓意图时才调用",
            "6. 用户说'继续'、'然后呢'、'下一步'时，根据对话上下文推断下一步操作",
            "7. 工具执行失败时，向用户说明原因并给出建议，不要反复重试同一工具",
            "",
            "## 回复风格",
            "- 使用通俗易懂的中文，像朋友聊天一样",
            "- 基于工具返回的真实数据回复，重要数字要准确",
            "- 涉及交易时提醒风险",
            "- 回复简洁，重点突出",
        ]
        # i18n: 回复语言跟随 UI 语言设置
        if _get_ui_lang() == "en":
            parts += [
                "",
                "## Response Language",
                "IMPORTANT: Always respond in English, regardless of the user's input language.",
            ]

        # 注入对话上下文 (实体记忆)
        context_summary = dialogue_manager.get_context_summary()
        if context_summary:
            parts += [
                "",
                "## 对话上下文 (记住的信息)",
                context_summary,
            ]

        # 注入页面上下文 (v1.1.0 页面感知)
        if page_context:
            parts += [
                "",
                "## 用户当前查看的页面",
                page_context[:2000],
            ]

        return "\n".join(parts)

    # ═══════════════════════════════════════════════════════
    # 推理循环 (流式)
    # ═══════════════════════════════════════════════════════

    def run_stream(self, text: str, page_context: str = ""):
        """
        流式执行 Agent 推理循环

        Args:
            text: 用户输入
            page_context: 用户当前页面内容摘要

        Yields:
            dict: {"type": "thinking"/"content", "text": "..."}
        """
        system_prompt = self._build_system_prompt(page_context)

        # Agent 消息序列: system + 对话上下文 + 本轮交互
        agent_messages = self._build_agent_messages(system_prompt, text)

        _lang = _get_ui_lang()
        yield {"type": "thinking", "text": "🧠 Understanding your request..." if _lang == "en" else "🧠 正在理解你的请求..."}

        # i18n: LLM 异常提示文案 (跟随 UI 语言)
        _err_think = "⚠️ AI thinking error" if _lang == "en" else "⚠️ AI 思考过程出现异常"
        _err_retry = "Please try again later, or type \"help\" to see available features." if _lang == "en" else "请稍后重试，或输入\"帮助\"查看功能。"

        tool_calls_log = []  # 本轮工具调用记录

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            # 1. LLM 思考
            try:
                decision = llm_client.chat_json(
                    agent_messages, temperature=0.1
                )
            except LLMCallError:
                # LLM 调用失败 (网络断/超时/未配置): 抛给上层 nl_router 自动降级到关键词规则引擎
                raise
            except Exception as e:
                logger.error(f"Agent LLM 调用失败 (迭代 {iteration}): {e}")
                yield {
                    "type": "content",
                    "text": f"{_err_think}: {e}\n\n{_err_retry}",
                }
                return

            # 2. 解析决策
            tool_name = decision.get("tool", "")
            reply = decision.get("reply", "")
            arguments = decision.get("arguments", {}) or {}

            # 3a. 最终回复 -> 结束循环
            if not tool_name and reply:
                yield {"type": "content", "text": reply}
                self._finalize(text, reply, tool_calls_log)
                return

            # 3b. 工具调用
            if tool_name:
                tool = self.registry.get(tool_name)
                if tool is None:
                    # 未知工具: 告知 LLM 并让其重新决策
                    agent_messages.append({
                        "role": "assistant",
                        "content": json.dumps({"tool": tool_name, "arguments": arguments}, ensure_ascii=False),
                    })
                    agent_messages.append({
                        "role": "user",
                        "content": f"错误: 未知工具 '{tool_name}'。请从可用工具列表中选择，或用 reply 直接回复。",
                    })
                    continue

                # 推送 thinking 进度
                yield {
                    "type": "thinking",
                    "text": (f"⚡ Executing: {tool.label_en or tool.label}...\n" if _lang == "en" else f"⚡ 正在执行: {tool.label}...\n") + "=" * 30 + "\n",
                }

                # 执行工具
                result = self.registry.execute(tool_name, arguments)
                tool_calls_log.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result_preview": result[:200],
                })

                logger.info(
                    f"Agent 工具调用 [{iteration}/{self.MAX_ITERATIONS}]: "
                    f"{tool_name}({arguments}) -> {len(result)} 字符"
                )

                # 工具结果回传 LLM，继续下一轮推理
                agent_messages.append({
                    "role": "assistant",
                    "content": json.dumps(
                        {"tool": tool_name, "arguments": arguments}, ensure_ascii=False
                    ),
                })
                agent_messages.append({
                    "role": "user",
                    "content": f"工具 {tool_name} 执行结果:\n{result}\n\n"
                               "请根据以上结果继续: 需要更多信息就继续调用工具 (JSON)，"
                               "信息足够就用 {\"reply\": \"...\"} 给用户最终回复。",
                })
                continue

            # 3c. 既无 tool 也无 reply: 异常输出，要求 LLM 重新决策
            agent_messages.append({
                "role": "user",
                "content": "你的回复格式不正确。请输出 {\"tool\": ..., \"arguments\": ...} 或 {\"reply\": ...} 之一的 JSON。",
            })

        # 超过最大迭代: 强制生成最终回复
        logger.warning(f"Agent 达到最大迭代次数 {self.MAX_ITERATIONS}，强制总结")
        agent_messages.append({
            "role": "user",
            "content": "已达到工具调用次数上限。请立即用 {\"reply\": \"...\"} 总结目前的结果给用户。",
        })
        try:
            decision = llm_client.chat_json(agent_messages, temperature=0.1)
            reply = decision.get("reply", "") or "已完成部分操作，如需继续请告诉我。"
        except Exception as e:
            logger.error(f"Agent 最终总结失败: {e}")
            reply = "已完成部分操作，如需继续请告诉我。"

        yield {"type": "content", "text": reply}
        self._finalize(text, reply, tool_calls_log)

    # ═══════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════

    def _build_agent_messages(self, system_prompt: str, text: str) -> list:
        """
        构建 Agent 消息序列

        system 提示词 + 最近对话上下文 (多轮记忆) + 当前用户输入
        """
        messages = [{"role": "system", "content": system_prompt}]

        # 注入最近对话上下文 (不含 system 消息，保留多轮记忆)
        try:
            recent = dialogue_manager.get_context(include_system=False)
            # 只保留最近 10 条，控制 token
            messages.extend(recent[-10:])
        except Exception as e:
            logger.debug(f"获取对话上下文失败 (忽略): {e}")

        # 当前用户输入
        messages.append({"role": "user", "content": text})
        return messages

    def _finalize(self, text: str, reply: str, tool_calls_log: list) -> None:
        """
        推理结束后更新对话上下文和实体记忆

        Args:
            text: 用户输入
            reply: 最终回复
            tool_calls_log: 本轮工具调用记录
        """
        try:
            # 更新对话上下文 (多轮记忆)
            dialogue_manager.add_user_message(text)
            dialogue_manager.add_assistant_message(reply)

            # 更新实体记忆 (以最后调用的工具为准)
            if tool_calls_log:
                last_tool = tool_calls_log[-1]["tool"]
                # 工具名到意图的映射 (供 get_context_summary 使用)
                tool_intent_map = {
                    "market_research": "research",
                    "stock_screen": "stock_screen",
                    "generate_strategy": "strategy",
                    "backtest_strategy": "backtest",
                    "execute_trade": "execute",
                    "monitor_positions": "monitor",
                    "full_auto_loop": "full_loop",
                    "emergency_stop": "emergency",
                    "goal_driven_trading": "goal_driven",
                }
                intent = tool_intent_map.get(last_tool, "")
                if intent:
                    # 从工具参数中提取实体
                    params = tool_calls_log[-1]["arguments"] or {}
                    dialogue_manager.update_memory(intent, params, None)
        except Exception as e:
            logger.debug(f"Agent 上下文更新失败 (非关键): {e}")

    def is_available(self) -> bool:
        """LLM 是否可用"""
        return llm_client.is_available()


# 全局单例
function_agent = FunctionAgent()
