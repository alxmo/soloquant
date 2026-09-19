"""
LLM 意图识别器
=====================================================
v0.4.0 新增 / v1.4.0 降级为兜底模块

⚠️ v1.4.0 变更:
- LLM 模式主路径已切换到 Function-Calling Agent
  (src/llm/function_agent.py + src/llm/tool_registry.py)
- LLM 可用时，意图判断 100% 由 Function Agent 完成，本模块不参与
- 本模块仅在 LLM 不可用 (断网/未配置) 时作为规则引擎降级兜底

功能:
- 用 LLM 分析用户自然语言，返回结构化意图 (兜底模式下不再使用 LLM 路径)
- 关键词匹配降级: LLM 不可用时回退到关键词匹配
- 供 nl_router._respond_with_rules 降级模式使用

意图类型 (与 src/ui/nl_router.py Intent 类对齐):
  research   - 市场调研
  strategy   - 策略生成
  backtest   - 回测验证
  execute    - 执行交易
  monitor    - 持仓监控
  status     - 系统状态
  help       - 帮助
  full_loop  - 全自动循环
  emergency  - 紧急止损
  unknown    - 未识别 (含闲聊/通用问题)
"""

from typing import Optional

from loguru import logger

from src.llm.llm_client import llm_client
from src.llm.dialogue_manager import dialogue_manager


class LLMIntent:
    """LLM 识别的意图结果"""

    def __init__(self, intent: str, params: dict = None, confidence: float = 1.0):
        self.intent = intent
        self.params = params or {}
        self.confidence = confidence

    def __repr__(self):
        return f"LLMIntent(intent={self.intent!r}, params={self.params!r}, confidence={self.confidence:.2f})"


class LLMIntentChain:
    """
    v2.0 智能升级: 多意图链

    支持用户一句话包含多个操作:
      "调研AAPL然后生成策略再回测" -> [research, strategy, backtest]
      "选股然后调研" -> [stock_screen, research]
    """

    def __init__(self, intents: list = None, needs_clarification: bool = False,
                 clarification_question: str = ""):
        self.intents: list[LLMIntent] = intents or []
        self.needs_clarification = needs_clarification
        self.clarification_question = clarification_question

    @property
    def is_chain(self) -> bool:
        """是否为多意图链 (>=2 个意图)"""
        return len(self.intents) >= 2

    @property
    def is_empty(self) -> bool:
        return len(self.intents) == 0

    @property
    def first(self) -> Optional[LLMIntent]:
        return self.intents[0] if self.intents else None

    def __repr__(self):
        if self.needs_clarification:
            return f"LLMIntentChain(clarify: {self.clarification_question})"
        chain_str = " -> ".join(i.intent for i in self.intents)
        return f"LLMIntentChain({chain_str})"


class IntentRecognizer:
    """LLM 意图识别器"""

    # 意图说明 (供 LLM 参考)
    INTENT_DESCRIPTIONS = {
        "research": "市场调研: 用户想了解市场动态、新闻、热点股票、投资机会",
        "strategy": "策略生成: 用户想创建、设计交易策略，涉及因子、模型、训练",
        "backtest": "回测验证: 用户想测试策略历史表现、验证效果",
        "execute": "执行交易: 用户想下单、买入、卖出、用模拟盘执行",
        "monitor": "持仓监控: 用户想查看持仓、盈亏、账户状态、交易报告",
        "status": "系统状态: 用户想知道系统配置、当前运行状态",
        "help": "帮助: 用户不知道怎么用，想了解功能",
        "full_loop": "全自动循环: 用户想一键完成全流程 (调研->策略->回测->执行->监控)",
        "emergency": "紧急止损: 用户想紧急清仓、全部卖出",
        "stock_screen": "智能选股: 用户想用自然语言选股，如选N只某价格区间的潜力股/科技股",
        "unknown": "未识别: 闲聊、通用问题、或无法归类的请求",
    }

    def __init__(self):
        self._fallback_router = None

    def recognize(self, text: str) -> LLMIntent:
        """
        识别用户意图 (单意图，向后兼容)

        优先用 LLM，不可用时降级到关键词匹配

        Args:
            text: 用户输入文本

        Returns:
            LLMIntent 实例
        """
        chain = self.recognize_chain(text)
        return chain.first if chain.first else LLMIntent(intent="unknown")

    def recognize_chain(self, text: str) -> LLMIntentChain:
        """
        v2.0 智能升级: 识别用户意图 (支持多意图链 + 上下文感知 + 主动澄清)

        优先用 LLM，不可用时降级到关键词匹配

        Args:
            text: 用户输入文本

        Returns:
            LLMIntentChain 实例
        """
        # 尝试 LLM 识别
        if llm_client.is_available():
            try:
                return self._recognize_chain_with_llm(text)
            except Exception as e:
                logger.warning(f"LLM 意图识别失败，降级到关键词匹配: {e}")

        # 降级到关键词匹配
        intent = self._recognize_with_keywords(text)
        return LLMIntentChain(intents=[intent])

    def _recognize_chain_with_llm(self, text: str) -> LLMIntentChain:
        """使用 LLM 识别意图 (支持多意图链 + 上下文感知 + 澄清)"""
        # 构建意图说明
        intent_list = "\n".join(
            f"  - {key}: {desc}" for key, desc in self.INTENT_DESCRIPTIONS.items()
        )

        # v2.0: 注入对话上下文 (上次意图 + 实体记忆)
        context_summary = dialogue_manager.get_context_summary()

        context_block = ""
        if context_summary:
            context_block = (
                f"\n\n[对话上下文]\n{context_summary}\n"
                "用户可能在续接上一步操作。如果用户说'继续'、'然后呢'、'接着'、"
                "'换个股票试试'、'再看看别的'等续接语，请根据上下文推断下一步意图。\n"
            )

        # v1.5.0: 移除续接词关键词拦截 (_detect_continuation)
        # 续接语判断完全交给 LLM (上方 context_block 已给出规则)，
        # 避免关键词硬匹配绕过 LLM 判断

        system_prompt = (
            "你是一个意图识别器。分析用户的输入，判断其意图类型并提取参数。\n\n"
            "可选的意图类型:\n"
            f"{intent_list}\n"
            f"{context_block}\n"
            "请以 JSON 格式回复，格式如下:\n"
            '{\n'
            '  "intents": [\n'
            '    {\n'
            '      "intent": "意图类型",\n'
            '      "confidence": 0.0-1.0,\n'
            '      "params": {\n'
            '        "symbols": ["股票代码列表"],\n'
            '        "user_interest": "用户关注的行业/主题",\n'
            '        "time_range": "时间范围描述",\n'
            '        "mode": "监控模式 (report/snapshot)",\n'
            '        "count": 选股数量 (整数),\n'
            '        "price_min": 最低价格,\n'
            '        "price_max": 最高价格,\n'
            '        "sector": "行业偏好",\n'
            '        "keywords": "选股关键词"\n'
            '      }\n'
            '    }\n'
            '  ],\n'
            '  "needs_clarification": false,\n'
            '  "clarification_question": ""\n'
            '}\n\n'
            "规则:\n"
            "1. 如果用户一句话包含多个操作 (如'调研AAPL然后生成策略再回测'),\n"
            "   将每个操作作为一个独立的 intent 放入 intents 数组\n"
            "2. 如果意图不明确或缺少必要信息 (如'帮我看看'没说看什么),\n"
            "   设置 needs_clarification=true 并在 clarification_question 中给出提问\n"
            "3. symbols 只在用户明确提到股票代码时填写\n"
            "4. params 中不用的字段留空数组、空字符串或0\n"
            '5. 如果用户在闲聊或问通用问题, intents 设为 [{"intent":"unknown"}]\n'
            "6. '回测策略' 应识别为 backtest 而非 strategy\n"
            "7. '紧急清仓/全部卖出' 应识别为 emergency\n"
            "8. '一键/全自动/全流程' 应识别为 full_loop\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        result = llm_client.chat_json(messages, temperature=0.1)

        # 检查是否需要澄清
        needs_clarify = result.get("needs_clarification", False)
        clarify_q = result.get("clarification_question", "")
        if needs_clarify and clarify_q:
            return LLMIntentChain(
                needs_clarification=True,
                clarification_question=clarify_q
            )

        # 解析意图列表
        raw_intents = result.get("intents", [])
        if not raw_intents:
            raw_intents = [{"intent": result.get("intent", "unknown")}]

        intents = []
        for raw in raw_intents:
            intent_str = raw.get("intent", "unknown").lower().strip()
            confidence = float(raw.get("confidence", 0.8))
            params = raw.get("params", {})

            if intent_str not in self.INTENT_DESCRIPTIONS:
                intent_str = "unknown"
                confidence = min(confidence, 0.5)

            intents.append(LLMIntent(intent=intent_str, params=params, confidence=confidence))

        logger.debug(
            f"LLM 意图识别: {text[:50]}... -> "
            f"{' -> '.join(i.intent for i in intents)} "
            f"(clarify={needs_clarify})"
        )

        return LLMIntentChain(intents=intents, needs_clarification=False)

    # v1.5.0: _detect_continuation 已移除
    # 续接语 ("继续"/"换个股票"等) 的判断完全由 LLM 完成，
    # 不再使用关键词硬匹配拦截 (符合"意图判断 100% 由 LLM 完成"原则)

    @staticmethod
    def _infer_next_intent(last_intent: str) -> Optional[str]:
        """根据上次意图推断工作流中的下一步"""
        workflow_chain = {
            "research": "strategy",
            "strategy": "backtest",
            "backtest": "execute",
            "stock_screen": "research",
        }
        return workflow_chain.get(last_intent)

    def _recognize_with_llm(self, text: str) -> LLMIntent:
        """使用 LLM 识别意图"""
        # 构建意图说明
        intent_list = "\n".join(
            f"  - {key}: {desc}" for key, desc in self.INTENT_DESCRIPTIONS.items()
        )

        system_prompt = (
            "你是一个意图识别器。分析用户的输入，判断其意图类型并提取参数。\n\n"
            "可选的意图类型:\n"
            f"{intent_list}\n\n"
            "请以 JSON 格式回复，格式如下:\n"
            '{\n'
            '  "intent": "意图类型 (上述之一)",\n'
            '  "confidence": 0.0-1.0 的置信度,\n'
            '  "params": {\n'
            '    "symbols": ["股票代码列表"],\n'
            '    "user_interest": "用户关注的行业/主题",\n'
            '    "time_range": "时间范围描述",\n'
            '    "mode": "监控模式 (report/snapshot)",\n'
            '    "count": 选股数量 (整数),\n'
            '    "price_min": 最低价格 (浮点数, 默认0),\n'
            '    "price_max": 最高价格 (浮点数, 0=不限),\n'
            '    "sector": "行业偏好 (如 科技/能源/医疗)",\n'
            '    "keywords": "选股关键词 (如 潜力/低价/成长)"\n'
            '  }\n'
            '}\n\n'
            "规则:\n"
            "1. symbols 只在用户明确提到股票代码时填写\n"
            "2. params 中不用的字段留空数组、空字符串或0\n"
            '3. 如果用户在闲聊或问通用问题 (与交易无关)，intent 设为 "unknown"\n'
            '4. "回测策略" 应识别为 backtest 而非 strategy\n'
            '5. "紧急清仓/全部卖出" 应识别为 emergency\n'
            '6. "一键/全自动/全流程" 应识别为 full_loop\n'
            '7. "选股/筛选股票/推荐股票/找股票" 应识别为 stock_screen\n'
            '8. stock_screen 意图时，从用户输入中提取选股参数:\n'
            '   - count: "选5支" -> 5, "推荐3只" -> 3, 默认5\n'
            '   - price_max: "10美元" -> 10, "50元以下" -> 50, "低于20" -> 20\n'
            '   - price_min: "10-50美元" -> 10, "20元以上" -> 20, 默认0\n'
            '   - sector: "科技股" -> "科技", "能源" -> "能源"\n'
            '   - keywords: "潜力股" -> "潜力", "成长股" -> "成长"\n'
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        result = llm_client.chat_json(messages, temperature=0.1)

        intent_str = result.get("intent", "unknown").lower().strip()
        confidence = float(result.get("confidence", 0.8))
        params = result.get("params", {})

        # 验证意图类型
        if intent_str not in self.INTENT_DESCRIPTIONS:
            intent_str = "unknown"
            confidence = min(confidence, 0.5)

        logger.debug(f"LLM 意图识别: {text[:50]}... -> {intent_str} (conf={confidence:.2f}) params={params}")

        return LLMIntent(intent=intent_str, params=params, confidence=confidence)

    def _recognize_with_keywords(self, text: str) -> LLMIntent:
        """
        降级: 使用关键词匹配 (复用现有 NLRouter 逻辑)

        直接实例化 NLRouter 调用其 _detect_intent 方法，
        避免重复维护关键词表。
        """
        if self._fallback_router is None:
            from src.ui.nl_router import NLRouter
            self._fallback_router = NLRouter()

        text_lower = text.lower().strip()
        intent = self._fallback_router._detect_intent(text_lower)

        # 尝试从文本中提取股票代码 (简单大写字母匹配)
        params = {}
        import re
        # 匹配 1-5 个大写字母的股票代码
        symbols = re.findall(r'\b[A-Z]{1,5}\b', text)
        # 过滤常见非股票词
        stop_words = {"I", "A", "AI", "IT", "OK", "TV", "US", "UK", "EU", "PR", "HR"}
        symbols = [s for s in symbols if s not in stop_words]
        if symbols:
            params["symbols"] = symbols

        # 选股参数提取 (降级模式: 简单正则)
        # 数量: "选5支" "推荐3只" "找10个"
        count_match = re.search(r'(?:选|推荐|找|挑)\s*(\d+)\s*(?:支|只|个|股)', text)
        if count_match:
            params["count"] = int(count_match.group(1))

        # 价格区间: "10-50美元" "5到20美元" (优先匹配)
        price_range_match = re.search(r'(\d+)\s*[-到~]\s*(\d+)\s*(?:美元|元|\$)', text)
        if price_range_match:
            params["price_min"] = float(price_range_match.group(1))
            params["price_max"] = float(price_range_match.group(2))
        else:
            # 价格上限: "10美元以下" "低于20"
            price_max_match = re.search(r'(\d+)\s*(?:美元|元|\$)\s*(?:以下|以内|下)|低于\s*(\d+)', text)
            if price_max_match:
                params["price_max"] = float(price_max_match.group(1) or price_max_match.group(2))
            else:
                # 价格下限: "20美元以上" "高于10"
                price_min_match = re.search(r'(\d+)\s*(?:美元|元|\$)\s*以上|高于\s*(\d+)', text)
                if price_min_match:
                    params["price_min"] = float(price_min_match.group(1) or price_min_match.group(2))
                else:
                    # 纯 "N美元" 作为价格上限 (如 "10美元的潜力股")
                    price_bare_match = re.search(r'(\d+)\s*(?:美元|元|\$)', text)
                    if price_bare_match:
                        params["price_max"] = float(price_bare_match.group(1))

        # 关键词
        if "潜力" in text:
            params["keywords"] = "潜力"
        elif "成长" in text:
            params["keywords"] = "成长"
        elif "低价" in text:
            params["keywords"] = "低价"

        # 行业
        sector_map = {"科技": "科技", "技术": "科技", "能源": "能源", "医疗": "医疗",
                      "金融": "金融", "消费": "消费", "新能源": "能源", "半导体": "科技"}
        for kw, sector in sector_map.items():
            if kw in text:
                params["sector"] = sector
                break

        logger.debug(f"关键词意图识别: {text[:50]}... -> {intent} params={params}")

        return LLMIntent(intent=intent, params=params, confidence=0.6)

    def is_llm_mode(self) -> bool:
        """当前是否使用 LLM 模式"""
        return llm_client.is_available()


# 全局单例
intent_recognizer = IntentRecognizer()
