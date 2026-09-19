"""
AI 分析中心 - 全链路 AI 介入模块
=====================================================
v0.4.1 新增

功能:
- 为 5 个 Agent (Research/Strategy/Backtest/Execution/Monitor) 提供 AI 分析能力
- 基于 LLM (llm_client) 做深度语义分析、风险评估、自然语言报告
- 所有方法都有降级处理: LLM 不可用时返回 None，Agent 自动回退到原有逻辑

设计原则:
1. 安全降级: 所有 LLM 调用 try/except，失败返回 None
2. 中文 prompt: 适配中文量化交易场景
3. 结构化输入: 每个方法接收结构化数据，输出自然语言文本
4. 性能可控: 单次调用超时 30s，控制 token 量
5. 懒加载: llm_client/llm_config 在方法内 import，避免循环导入
"""

from typing import Any, List, Optional

from loguru import logger


class AIAnalyzer:
    """全链路 AI 分析器"""

    def __init__(self):
        self._enabled: Optional[bool] = None

    def _get_llm_client(self):
        """懒加载 llm_client (避免循环导入)"""
        from src.llm.llm_client import llm_client
        return llm_client

    def _get_llm_config(self):
        """懒加载 llm_config (避免循环导入)"""
        from src.llm.llm_config import llm_config
        return llm_config

    def _get_llm_call_error(self):
        """懒加载 LLMCallError (避免循环导入)"""
        from src.llm.llm_client import LLMCallError
        return LLMCallError

    def is_available(self) -> bool:
        """检测 AI 分析是否可用 (LLM 已配置且可连接)"""
        if self._enabled is not None:
            return self._enabled
        try:
            llm_client = self._get_llm_client()
            llm_config = self._get_llm_config()
            self._enabled = llm_client.is_available()
            if self._enabled:
                logger.info(f"🤖 AI 分析已启用 | 模型: {llm_config.get_default_name()}")
            else:
                logger.info("🤖 AI 分析未启用 (LLM 未配置)，将使用规则降级模式")
        except Exception as e:
            logger.warning(f"🤖 AI 分析初始化失败 (降级): {e}")
            self._enabled = False
        return self._enabled

    def _safe_chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> Optional[str]:
        """
        安全调用 LLM 对话 (带降级处理)

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            LLM 回复文本，失败返回 None
        """
        if not self.is_available():
            return None
        try:
            llm_client = self._get_llm_client()
            LLMCallError = self._get_llm_call_error()
            return llm_client.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except LLMCallError as e:
            logger.warning(f"🤖 AI 分析调用失败 (降级): {e}")
            return None
        except Exception as e:
            logger.warning(f"🤖 AI 分析异常 (降级): {e}")
            return None

    # ═══════════════════════════════════════════════════════
    # 1. Research Agent AI 分析
    # ═══════════════════════════════════════════════════════

    def analyze_research_sentiment(
        self,
        news_list: List[dict],
    ) -> Optional[dict]:
        """
        AI 新闻语义分析 (替代关键词匹配)

        分析新闻标题和摘要的深层语义，返回:
        - sentiment_score: 情绪分数 (-1.0 ~ 1.0)
        - sentiment_label: 情绪标签
        - analysis: AI 分析文本

        Args:
            news_list: 新闻列表 [{"headline": str, "summary": str, ...}]

        Returns:
            {"sentiment_score": float, "sentiment_label": str, "analysis": str}
            LLM 不可用时返回 None
        """
        if not news_list:
            return None

        # 截取前 20 条新闻，控制 token 量
        sample_news = news_list[:20]
        news_text = "\n".join(
            f"- {n.get('headline', '')}"
            for n in sample_news
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位专业的金融市场分析师，擅长新闻语义分析。"
                    "请分析以下新闻的整体市场情绪，返回 JSON 格式。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"以下是今日市场新闻标题:\n{news_text}\n\n"
                    "请分析整体市场情绪，返回 JSON:\n"
                    '{"sentiment_score": -1.0到1.0的浮点数, '
                    '"sentiment_label": "极度悲观/悲观/中性/乐观/极度乐观", '
                    '"analysis": "简要分析当前市场情绪和主要影响因素(100字以内)"}'
                ),
            },
        ]

        try:
            llm_client = self._get_llm_client()
            result = llm_client.chat_json(messages, temperature=0.2, max_tokens=500)
            score = float(result.get("sentiment_score", 0.0))
            label = result.get("sentiment_label", "中性")
            analysis = result.get("analysis", "")
            logger.info(f"🤖 AI 新闻语义分析完成: {label} ({score:+.2f})")
            return {
                "sentiment_score": max(-1.0, min(1.0, score)),
                "sentiment_label": label,
                "analysis": analysis,
            }
        except Exception as e:
            logger.warning(f"🤖 AI 新闻语义分析失败 (降级): {e}")
            return None

    def analyze_research_report(
        self,
        sentiment_label: str,
        sentiment_score: float,
        candidates: list,
        key_events: list,
        ai_sentiment_analysis: str = "",
    ) -> Optional[str]:
        """
        AI 生成深度调研报告

        基于市场情绪、候选股票、关键事件，生成自然语言调研报告。

        Args:
            sentiment_label: 情绪标签
            sentiment_score: 情绪分数
            candidates: 候选股票列表 (StockCandidate 或 dict)
            key_events: 关键事件列表
            ai_sentiment_analysis: AI 情绪分析文本 (如有)

        Returns:
            AI 调研报告文本，LLM 不可用时返回 None
        """
        if not self.is_available():
            return None

        # 构建候选股票摘要
        candidate_lines = []
        for c in candidates[:10]:
            sym = c.symbol if hasattr(c, "symbol") else c.get("symbol", "?")
            name = c.name if hasattr(c, "name") else c.get("name", sym)
            mentions = c.mention_count if hasattr(c, "mention_count") else c.get("mention_count", 0)
            candidate_lines.append(f"  - {sym} ({name}): 新闻提及 {mentions} 次")

        # 构建关键事件摘要
        event_lines = []
        for e in key_events[:5]:
            title = e.get("title", "") if isinstance(e, dict) else str(e)
            event_lines.append(f"  - {title}")

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位资深量化交易分析师，请基于以下市场调研数据，"
                    "生成一份简洁但有深度的中文调研报告。"
                    "报告应包含: 市场情绪解读、热点板块分析、候选股票点评、投资建议。"
                    "控制在 300 字以内，使用自然流畅的语言。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"市场情绪: {sentiment_label} ({sentiment_score:+.2f})\n"
                    f"AI情绪分析: {ai_sentiment_analysis or '无'}\n\n"
                    f"候选股票:\n" + "\n".join(candidate_lines) + "\n\n"
                    f"关键事件:\n" + "\n".join(event_lines) + "\n\n"
                    "请生成深度调研报告:"
                ),
            },
        ]

        report = self._safe_chat(messages, temperature=0.4, max_tokens=800)
        if report:
            logger.info("🤖 AI 调研报告生成完成")
        return report

    def analyze_research_candidate_reason(
        self,
        symbol: str,
        name: str,
        sector: str,
        sentiment_score: float,
        mention_count: int,
        news_headlines: List[str] = None,
    ) -> Optional[str]:
        """
        AI 为单只候选股票生成推荐理由

        Args:
            symbol: 股票代码
            name: 股票名称
            sector: 行业
            sentiment_score: 情绪分数
            mention_count: 新闻提及次数
            news_headlines: 相关新闻标题列表

        Returns:
            AI 生成的推荐理由，LLM 不可用时返回 None
        """
        if not self.is_available():
            return None

        news_str = ""
        if news_headlines:
            news_str = "相关新闻:\n" + "\n".join(f"  - {h}" for h in news_headlines[:3])

        messages = [
            {
                "role": "system",
                "content": "你是量化交易分析师，请用一句话点评候选股票的投资价值，50字以内。",
            },
            {
                "role": "user",
                "content": (
                    f"股票: {symbol} ({name})\n"
                    f"行业: {sector or '未知'}\n"
                    f"情绪分数: {sentiment_score:+.2f}\n"
                    f"新闻提及: {mention_count} 次\n"
                    f"{news_str}\n"
                    "请生成投资点评:"
                ),
            },
        ]

        return self._safe_chat(messages, temperature=0.5, max_tokens=150)

    # ═══════════════════════════════════════════════════════
    # 2. Strategy Agent AI 分析
    # ═══════════════════════════════════════════════════════

    def analyze_strategy(
        self,
        strategy_name: str,
        model_type: str,
        stock_pool: List[str],
        factors: list,
        model_result: Any,
        signals: list,
    ) -> Optional[str]:
        """
        AI 策略分析报告

        分析因子组合有效性、模型预测能力、信号合理性，生成策略分析文本。

        Args:
            strategy_name: 策略名称
            model_type: 模型类型
            stock_pool: 股票池
            factors: 因子列表
            model_result: 模型训练结果
            signals: 交易信号列表

        Returns:
            AI 策略分析文本，LLM 不可用时返回 None
        """
        if not self.is_available():
            return None

        # 因子信息
        factor_lines = []
        for f in factors[:8]:
            fname = f.name if hasattr(f, "name") else str(f)
            ic = f"{f.ic:.4f}" if hasattr(f, "ic") and f.ic else "N/A"
            factor_lines.append(f"  - {fname} (IC={ic})")

        # 模型信息
        model_info = "无模型结果"
        if model_result:
            valid_ic = getattr(model_result, "valid_ic", 0)
            train_ic = getattr(model_result, "train_ic", 0)
            model_info = f"训练IC={train_ic:.4f}, 验证IC={valid_ic:.4f}"

        # 信号信息
        signal_lines = []
        for s in signals[:8]:
            sym = s.symbol if hasattr(s, "symbol") else "?"
            direction = s.direction.value if hasattr(s, "direction") else "?"
            weight = f"{s.target_weight:.0%}" if hasattr(s, "target_weight") else "?"
            signal_lines.append(f"  - {sym}: {direction} (仓位 {weight})")

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位资深量化策略分析师。请分析以下策略的因子组合、模型表现和信号质量，"
                    "生成一份策略分析报告。报告应包含: 因子评价、模型诊断、信号合理性判断、优化建议。"
                    "控制在 300 字以内。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"策略: {strategy_name}\n"
                    f"模型: {model_type}\n"
                    f"股票池: {', '.join(stock_pool)}\n\n"
                    f"有效因子:\n" + "\n".join(factor_lines) + "\n\n"
                    f"模型表现: {model_info}\n\n"
                    f"交易信号:\n" + "\n".join(signal_lines) + "\n\n"
                    "请生成策略分析报告:"
                ),
            },
        ]

        report = self._safe_chat(messages, temperature=0.4, max_tokens=800)
        if report:
            logger.info("🤖 AI 策略分析报告生成完成")
        return report

    # ═══════════════════════════════════════════════════════
    # 3. Backtest Agent AI 分析
    # ═══════════════════════════════════════════════════════

    def analyze_backtest(
        self,
        strategy_name: str,
        grade: str,
        annual_return: float,
        max_drawdown: float,
        sharpe_ratio: float,
        win_rate: float,
        total_return: float,
        initial_capital: float,
        final_value: float,
        period: tuple,
        stock_pool: List[str],
    ) -> Optional[str]:
        """
        AI 回测深度解读

        分析回测指标，识别过拟合风险，提供优化建议。

        Args:
            strategy_name: 策略名称
            grade: 策略评级
            annual_return: 年化收益率
            max_drawdown: 最大回撤
            sharpe_ratio: 夏普比率
            win_rate: 胜率
            total_return: 总收益率
            initial_capital: 初始资金
            final_value: 最终资金
            period: 回测区间 (start, end)
            stock_pool: 股票池

        Returns:
            AI 回测解读文本，LLM 不可用时返回 None
        """
        if not self.is_available():
            return None

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位资深量化回测分析师。请基于回测数据进行深度解读，包含:\n"
                    "1. 核心指标评价 (收益/风险/稳定性)\n"
                    "2. 过拟合风险评估 (IC是否合理、回撤是否异常)\n"
                    "3. 策略优化建议 (因子/参数/风控)\n"
                    "4. 实盘部署建议\n"
                    "控制在 300 字以内，语言简洁专业。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"策略: {strategy_name}\n"
                    f"评级: {grade}\n"
                    f"回测区间: {period[0]} ~ {period[1]}\n"
                    f"股票池: {', '.join(stock_pool)}\n"
                    f"初始资金: ${initial_capital:,.2f}\n"
                    f"最终资金: ${final_value:,.2f}\n\n"
                    f"核心指标:\n"
                    f"  年化收益: {annual_return:.1%}\n"
                    f"  最大回撤: {max_drawdown:.1%}\n"
                    f"  夏普比率: {sharpe_ratio:.2f}\n"
                    f"  胜率: {win_rate:.1%}\n"
                    f"  总收益: {total_return:.1%}\n\n"
                    "请进行深度解读:"
                ),
            },
        ]

        report = self._safe_chat(messages, temperature=0.3, max_tokens=800)
        if report:
            logger.info("🤖 AI 回测深度解读生成完成")
        return report

    # ═══════════════════════════════════════════════════════
    # 4. Execution Agent AI 分析
    # ═══════════════════════════════════════════════════════

    def analyze_execution(
        self,
        total_signals: int,
        executed: int,
        rejected: int,
        skipped: int,
        details: list,
        trading_mode: str = "paper",
        simulated: bool = False,
    ) -> Optional[str]:
        """
        AI 交易执行审查

        分析执行结果，评估订单质量，识别异常情况。

        Args:
            total_signals: 信号总数
            executed: 成功执行数
            rejected: 风控拒绝数
            skipped: 跳过数
            details: 执行明细列表
            trading_mode: 交易模式 (paper/live)
            simulated: 是否模拟执行

        Returns:
            AI 执行审查文本，LLM 不可用时返回 None
        """
        if not self.is_available():
            return None

        # 构建执行明细摘要
        detail_lines = []
        for d in details[:10]:
            sym = d.get("symbol", "?")
            direction = d.get("direction", "?")
            status = d.get("status", "?")
            reason = d.get("reason", "")[:60]
            detail_lines.append(f"  - {sym} {direction}: {status} | {reason}")

        mode_label = "模拟盘" if trading_mode == "paper" else "实盘"

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位专业的交易执行审查员。请审查以下交易执行结果，包含:\n"
                    "1. 执行效率评价 (成功率/拒绝原因)\n"
                    "2. 风控有效性评估\n"
                    "3. 异常订单识别\n"
                    "4. 后续操作建议\n"
                    "控制在 200 字以内。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"交易模式: {mode_label}{' (模拟执行)' if simulated else ''}\n"
                    f"信号总数: {total_signals}\n"
                    f"成功执行: {executed}\n"
                    f"风控拒绝: {rejected}\n"
                    f"跳过: {skipped}\n\n"
                    f"执行明细:\n" + "\n".join(detail_lines) + "\n\n"
                    "请进行执行审查:"
                ),
            },
        ]

        report = self._safe_chat(messages, temperature=0.3, max_tokens=500)
        if report:
            logger.info("🤖 AI 执行审查生成完成")
        return report

    # ═══════════════════════════════════════════════════════
    # 5. Monitor Agent AI 分析
    # ═══════════════════════════════════════════════════════

    def analyze_monitor(
        self,
        equity: float,
        cash: float,
        positions: list,
        risk_status: dict,
        alerts: list,
        stop_loss_triggered: list,
        mode: str = "snapshot",
    ) -> Optional[str]:
        """
        AI 持仓分析报告

        分析持仓盈亏、风控状态，生成自然语言监控报告。

        Args:
            equity: 总资产
            cash: 现金
            positions: 持仓列表
            risk_status: 风控状态
            alerts: 告警列表
            stop_loss_triggered: 止损触发列表
            mode: 报告模式 (snapshot/report)

        Returns:
            AI 监控分析文本，LLM 不可用时返回 None
        """
        if not self.is_available():
            return None

        # 持仓摘要
        pos_lines = []
        total_pnl = 0
        for p in positions[:10]:
            sym = p.get("symbol", "?")
            pnl = p.get("unrealized_pnl", 0)
            pnl_pct = p.get("unrealized_pnl_pct", 0)
            weight = p.get("weight", 0)
            total_pnl += pnl
            icon = "🟢" if pnl >= 0 else "🔴"
            pos_lines.append(f"  {icon} {sym}: 盈亏 ${pnl:+,.0f} ({pnl_pct:+.1%}) 仓位 {weight:.1%}")

        # 风控摘要
        dd = risk_status.get("current_drawdown", 0)
        daily_loss = risk_status.get("current_daily_loss", 0)
        is_alert = risk_status.get("is_alert", False)

        # 告警摘要
        alert_lines = [f"  - {a}" for a in alerts[:5]] if alerts else ["  无告警"]

        # 止损摘要
        stop_lines = [f"  - {s.get('symbol', '?')}: {s.get('reason', '')}" for s in stop_loss_triggered[:5]]

        report_type = "日报" if mode == "report" else "快照"

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位专业的量化交易风控监控员。请基于以下持仓和风控数据，"
                    "生成一份自然语言监控报告。报告应包含:\n"
                    "1. 组合整体表现评价\n"
                    "2. 个股盈亏原因分析 (基于行业/仓位/趋势)\n"
                    "3. 风控状态评估\n"
                    "4. 操作建议 (减仓/加仓/止损/持有)\n"
                    "控制在 300 字以内。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"📊 组合监控{report_type}\n"
                    f"总资产: ${equity:,.2f}\n"
                    f"现金: ${cash:,.2f}\n"
                    f"持仓数: {len(positions)}\n"
                    f"总浮动盈亏: ${total_pnl:+,.2f}\n\n"
                    f"持仓明细:\n" + "\n".join(pos_lines) + "\n\n"
                    f"风控状态:\n"
                    f"  当前回撤: {dd:.1%}\n"
                    f"  今日盈亏: {daily_loss:+.1%}\n"
                    f"  风控告警: {'是' if is_alert else '否'}\n\n"
                    f"告警 ({len(alerts)} 条):\n" + "\n".join(alert_lines) + "\n\n"
                    + (f"止损触发 ({len(stop_loss_triggered)} 只):\n" + "\n".join(stop_lines) + "\n\n" if stop_loss_triggered else "")
                    + "请生成监控分析报告:"
                ),
            },
        ]

        report = self._safe_chat(messages, temperature=0.3, max_tokens=800)
        if report:
            logger.info(f"🤖 AI 监控{report_type}生成完成")
        return report

    def analyze_crypto_monitor(
        self,
        equity: float,
        cash: float,
        positions: list,
        risk_status: dict,
        alerts: list,
        stop_loss_triggered: list,
    ) -> Optional[str]:
        """
        AI 加密货币持仓分析报告

        Args:
            equity: 总资产
            cash: 现金
            positions: 加密货币持仓列表
            risk_status: 风控状态
            alerts: 告警列表
            stop_loss_triggered: 止损触发列表

        Returns:
            AI 监控分析文本，LLM 不可用时返回 None
        """
        if not self.is_available():
            return None

        pos_lines = []
        total_pnl = 0
        for p in positions[:10]:
            sym = p.get("symbol", "?")
            pnl = p.get("unrealized_pnl", 0)
            pnl_pct = p.get("unrealized_pnl_pct", 0)
            total_pnl += pnl
            icon = "🟢" if pnl >= 0 else "🔴"
            pos_lines.append(f"  {icon} {sym}: 盈亏 ${pnl:+,.0f} ({pnl_pct:+.1%})")

        dd = risk_status.get("current_drawdown", 0)
        is_alert = risk_status.get("is_alert", False)
        alert_lines = [f"  - {a}" for a in alerts[:5]] if alerts else ["  无告警"]

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位专业的加密货币交易风控监控员。"
                    "加密货币 24/7 交易，波动性远高于股票。"
                    "请基于持仓数据生成自然语言监控报告，包含:\n"
                    "1. 加密货币组合表现\n"
                    "2. 波动性风险评估\n"
                    "3. 止损/止盈建议\n"
                    "控制在 250 字以内。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"🔒 加密货币监控报告\n"
                    f"总资产: ${equity:,.2f}\n"
                    f"持仓数: {len(positions)}\n"
                    f"总浮动盈亏: ${total_pnl:+,.2f}\n\n"
                    f"持仓明细:\n" + "\n".join(pos_lines) + "\n\n"
                    f"风控状态: 回撤 {dd:.1%}, 告警 {'是' if is_alert else '否'}\n"
                    f"告警: " + "; ".join(alerts[:3]) + "\n\n"
                    "请生成加密货币监控分析:"
                ),
            },
        ]

        report = self._safe_chat(messages, temperature=0.3, max_tokens=600)
        if report:
            logger.info("🤖 AI 加密货币监控分析生成完成")
        return report


# 全局单例
ai_analyzer = AIAnalyzer()
