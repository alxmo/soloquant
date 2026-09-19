"""
Research Agent - 市场调研分析师
=====================================================
职责:
1. 扫描市场新闻 (Finnhub)
2. 新闻情绪分析
3. 热点股票发现与筛选
4. 生成调研报告 (含小白解读)

工具链:
- data_collector.get_market_news() -> 市场新闻
- data_collector.get_company_news() -> 公司新闻
- data_collector.get_stock_overview() -> 公司综合信息
- data_collector.get_company_profile() -> 公司概况
- data_collector.get_earnings_calendar() -> 财报日历
"""

from datetime import datetime
from typing import Any
from collections import Counter

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.data.collector import data_collector
from src.models import AssetType, CryptoCandidate, ResearchReport, StockCandidate
from src.utils.config import config

try:
    from src.ui.i18n import get_language
except ImportError:  # 避免循环依赖时的回退
    def get_language():
        return "zh"


class ResearchAgent(BaseAgent):
    """市场调研员 Agent"""

    # 默认关注列表 (引用 config, 保持向后兼容)
    DEFAULT_WATCHLIST = config.DEFAULT_WATCHLIST

    # 行业关键词映射 (用于新闻情绪粗分析)
    SECTOR_KEYWORDS = {
        "科技": ["AI", "芯片", "软件", "云", "科技", "tech", "chip", "software", "cloud", "semiconductor"],
        "金融": ["银行", "利率", "金融", "bank", "interest", "financial", "Fed"],
        "能源": ["石油", "天然气", "能源", "oil", "energy", "gas"],
        "医疗": ["药", "医疗", "生物", "pharma", "health", "biotech", "drug"],
        "消费": ["零售", "消费", "电商", "retail", "consumer", "e-commerce"],
    }

    # 加密货币关键词映射 (Phase 8 新增)
    CRYPTO_KEYWORDS = {
        "比特币": ["BTC", "Bitcoin", "比特币"],
        "以太坊": ["ETH", "Ethereum", "以太坊"],
        "DeFi": ["DeFi", "去中心化金融", "Uniswap", "Aave"],
        "NFT": ["NFT", "非同质化代币"],
        "Layer2": ["Layer2", "L2", "Arbitrum", "Optimism", "Polygon"],
        "Solana": ["SOL", "Solana"],
        "稳定币": ["USDT", "USDC", "stablecoin", "稳定币"],
    }

    # 加密货币交易对 -> 名称映射
    CRYPTO_NAME_MAP = {
        "BTC/USD": "Bitcoin", "ETH/USD": "Ethereum", "SOL/USD": "Solana",
        "ADA/USD": "Cardano", "AVAX/USD": "Avalanche", "DOT/USD": "Polkadot",
        "MATIC/USD": "Polygon", "LINK/USD": "Chainlink",
        "BTC/USDT": "Bitcoin", "ETH/USDT": "Ethereum",
    }

    def __init__(self):
        super().__init__(name="research", role="市场调研员")

    def _execute(
        self,
        user_interest: str = "",
        scan_scope: str = "full",
        max_candidates: int = None,
        asset_type: str = "stock",
        **kwargs,
    ) -> ResearchReport:
        """
        执行市场调研

        Args:
            user_interest: 用户关注的行业/主题 (如 "AI", "新能源")
            scan_scope: 扫描范围 "full" | "tech" | "finance"
            max_candidates: 最大候选股票数
            asset_type: 资产类型 "stock" | "crypto" (Phase 8 新增)

        Returns:
            ResearchReport
        """
        # Phase 8: 加密货币调研路由
        if asset_type == "crypto":
            return self._execute_crypto_research(user_interest, max_candidates)

        # 使用配置值（如果未指定）
        if max_candidates is None:
            max_candidates = config.MAX_CANDIDATES

        logger.info(f"📋 开始市场调研 | 关注: {user_interest or '全市场'} | 范围: {scan_scope} | 候选上限: {max_candidates}")

        # 1. 获取市场新闻
        market_news = self._fetch_market_news()
        company_news = self._fetch_company_news()
        all_news = market_news + company_news

        logger.info(f"  获取新闻: 市场 {len(market_news)} 条, 公司 {len(company_news)} 条")

        # 2. 情绪分析 (AI 增强: 优先使用 LLM 语义分析，降级到关键词匹配)
        from src.llm.ai_analyzer import ai_analyzer
        ai_sentiment = ai_analyzer.analyze_research_sentiment(all_news)
        if ai_sentiment:
            sentiment_score = ai_sentiment["sentiment_score"]
            sentiment_label = ai_sentiment["sentiment_label"]
            ai_sentiment_analysis = ai_sentiment.get("analysis", "")
            logger.info(f"  市场情绪 (AI): {sentiment_label} ({sentiment_score:+.2f})")
        else:
            sentiment_score, sentiment_label = self._analyze_sentiment(all_news)
            ai_sentiment_analysis = ""
            logger.info(f"  市场情绪 (规则): {sentiment_label} ({sentiment_score:+.2f})")

        # 3. 候选股票筛选
        candidates = self._discover_candidates(all_news, user_interest, max_candidates)
        logger.info(f"  候选股票: {len(candidates)} 只")

        # 4. 补充候选股票详情
        candidates = self._enrich_candidates(candidates)

        # 5. 关键事件提取
        key_events = self._extract_key_events(all_news)

        # 6. 生成小白解读 (AI 增强: 优先使用 AI 调研报告)
        ai_report = ai_analyzer.analyze_research_report(
            sentiment_label, sentiment_score, candidates, key_events,
            ai_sentiment_analysis,
        )
        if ai_report:
            plain_summary = self._build_plain_summary(
                sentiment_label, sentiment_score, candidates, key_events
            )
            plain_summary += f"\n\n🤖 AI 深度分析:\n{'─' * 40}\n{ai_report}"
        else:
            plain_summary = self._build_plain_summary(
                sentiment_label, sentiment_score, candidates, key_events
            )

        report = ResearchReport(
            report_id=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            date=datetime.now(),
            candidate_stocks=candidates,
            market_sentiment=sentiment_score,
            key_events=key_events,
            asset_type=AssetType.STOCK,
            plain_summary=plain_summary,
        )

        logger.info(f"📋 调研报告生成完成: {report.report_id}")
        return report

    def _fetch_market_news(self) -> list[dict]:
        """获取市场新闻"""
        try:
            news = data_collector.get_market_news(category="general", count=50)
            return news if news else []
        except Exception as e:
            logger.warning(f"获取市场新闻失败: {e}")
            return []

    def _fetch_company_news(self, max_symbols: int = None) -> list[dict]:
        """
        获取关注列表的公司新闻 (并行获取，提升速度)

        Args:
            max_symbols: 最多获取几只股票的新闻 (None=全部关注列表)
        """
        from src.utils.parallel import parallel_map

        watchlist = config.WATCHLIST if config.WATCHLIST else config.DEFAULT_WATCHLIST
        if max_symbols is not None:
            symbols = watchlist[:max_symbols]
        else:
            symbols = watchlist  # 获取全部关注列表新闻

        def _fetch_one(sym: str) -> list[dict]:
            """获取单只股票的公司新闻"""
            try:
                news = data_collector.get_company_news(sym, days=3)
                # 给每条新闻标注股票代码
                for n in news:
                    n["_symbol"] = sym
                return news[:3]  # 每只取3条
            except Exception as e:
                logger.debug(f"获取 {sym} 公司新闻失败: {e}")
                return []

        # 并行获取 (max_workers 动态调整: 5 ~ 10)
        num_workers = min(max(5, len(symbols)), 10)
        results = parallel_map(
            _fetch_one, symbols,
            max_workers=num_workers,
            desc="获取公司新闻",
            timeout=60,
        )

        all_news = []
        for news_list in results:
            if news_list:
                all_news.extend(news_list)

        return all_news

    def research_single_stock(self, symbol: str) -> dict:
        """
        对单只股票进行深度调研，返回结构化调研结果

        Args:
            symbol: 股票代码 (如 "AAPL")

        Returns:
            dict: {
                "symbol": str, "name": str, "sector": str,
                "sentiment_score": float, "news": list[dict],
                "plain_reason": str, "profile": dict|None,
            }
        """
        symbol = symbol.upper().strip()
        logger.info(f"🔍 单股深度调研: {symbol}")

        # 1. 获取公司新闻
        try:
            news = data_collector.get_company_news(symbol, days=7)
        except Exception as e:
            logger.warning(f"获取 {symbol} 新闻失败: {e}")
            news = []
        # 标注来源
        for n in news:
            n["_symbol"] = symbol

        # 2. 情绪分析
        sentiment_score, sentiment_label = self._analyze_sentiment(news)

        # 3. 获取公司概况
        profile = None
        try:
            profile = data_collector.get_company_profile(symbol)
        except Exception as e:
            logger.debug(f"获取 {symbol} 概况失败: {e}")

        # 4. 获取公司名称
        name = self._get_stock_name(symbol)
        sector = None
        if profile:
            name = profile.get("name", name)
            sector = profile.get("finnhubIndustry", "") or None

        # 5. 生成推荐理由
        if profile:
            candidate = StockCandidate(
                symbol=symbol, name=name, sector=sector,
                sentiment_score=sentiment_score, mention_count=len(news),
                key_events=[], plain_reason="",
            )
            plain_reason = self._generate_plain_reason(candidate, profile)
        else:
            plain_reason = (
                    f"{name} has {len(news)} related news recently, market sentiment {sentiment_label}." if get_language() == "en"
                    else f"{name} 近期有 {len(news)} 条相关新闻，市场情绪{sentiment_label}。"
                )

        # 6. 截取最多5条新闻摘要
        news_items = []
        for n in news[:5]:
            news_items.append({
                "headline": n.get("headline", ""),
                "source": n.get("source", ""),
                "url": n.get("url", ""),
                "datetime": n.get("datetime", ""),
            })

        return {
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "news": news_items,
            "plain_reason": plain_reason,
            "profile": profile,
        }

    def research_watchlist(self) -> list[dict]:
        """
        对关注列表中的所有股票进行调研，返回每只股票的调研摘要

        Returns:
            list[dict]: 每只股票的调研结果 (同 research_single_stock 返回格式)
        """
        watchlist = config.WATCHLIST if config.WATCHLIST else config.DEFAULT_WATCHLIST
        logger.info(f"⭐ 关注列表调研: {len(watchlist)} 只股票")

        from src.utils.parallel import parallel_map

        def _research_one(sym: str) -> dict:
            try:
                return self.research_single_stock(sym)
            except Exception as e:
                logger.warning(f"调研 {sym} 失败: {e}")
                return {
                    "symbol": sym, "name": self._get_stock_name(sym),
                    "sector": None, "sentiment_score": 0.0,
                    "sentiment_label": "中性", "news": [],
                    "plain_reason": f"调研失败: {e}", "profile": None,
                }

        # 并行调研所有关注股票
        results = parallel_map(
            _research_one, watchlist,
            max_workers=min(len(watchlist), 8),
            desc="关注列表调研",
            timeout=120,
        )

        return results

    def _analyze_sentiment(self, news_list: list[dict]) -> tuple[float, str]:
        """
        简易情绪分析 (基于关键词)

        Returns:
            (sentiment_score, sentiment_label)
            score: -1.0 (极度悲观) ~ 1.0 (极度乐观)
            label: "极度悲观" / "悲观" / "中性" / "乐观" / "极度乐观"
        """
        if not news_list:
            return 0.0, ("Neutral" if get_language() == "en" else "中性")

        # 积极和消极关键词
        positive_words = [
            "surge", "soar", "rally", "gain", "beat", "upgrade", "bullish",
            "growth", "record", "high", "strong", "buy", "outperform",
            "profit", "rise", "jump", "boom", "breakthrough", "positive",
        ]
        negative_words = [
            "drop", "fall", "decline", "loss", "miss", "downgrade", "bearish",
            "weak", "sell", "plunge", "crash", "fear", "concern", "risk",
            "cut", "low", "warning", "lawsuit", "investigation", "negative",
        ]

        pos_count = 0
        neg_count = 0

        for news in news_list:
            # 合并标题和摘要
            text = (news.get("headline", "") + " " + news.get("summary", "")).lower()
            for w in positive_words:
                if w in text:
                    pos_count += 1
            for w in negative_words:
                if w in text:
                    neg_count += 1

        total = pos_count + neg_count
        if total == 0:
            return 0.0, "中性"

        score = (pos_count - neg_count) / total

        # 按当前 UI 语言输出情绪标签
        if get_language() == "en":
            if score > 0.5:
                label = "Very Bullish"
            elif score > 0.15:
                label = "Bullish"
            elif score > -0.15:
                label = "Neutral"
            elif score > -0.5:
                label = "Bearish"
            else:
                label = "Very Bearish"
        else:
            if score > 0.5:
                label = "极度乐观"
            elif score > 0.15:
                label = "乐观"
            elif score > -0.15:
                label = "中性"
            elif score > -0.5:
                label = "悲观"
            else:
                label = "极度悲观"

        return round(score, 2), label

    def _match_user_interest(self, user_interest: str) -> list[str]:
        """
        将用户兴趣文本匹配到行业关键词列表

        Args:
            user_interest: 用户输入的兴趣文本 (如 "AI", "新能源", "科技股")

        Returns:
            匹配到的关键词列表 (用于新闻标题匹配)
        """
        if not user_interest or not user_interest.strip():
            return []

        interest_lower = user_interest.lower().strip()
        matched_keywords: list[str] = []

        # 1. 先匹配预定义行业关键词
        for sector, keywords in self.SECTOR_KEYWORDS.items():
            # 用户输入包含行业名 (中文或英文)
            if sector in user_interest or sector.lower() in interest_lower:
                matched_keywords.extend(keywords)
                continue
            # 用户输入包含某个行业关键词
            for kw in keywords:
                if kw.lower() in interest_lower:
                    matched_keywords.extend(keywords)
                    break

        # 2. 将用户兴趣本身也作为关键词 (支持自定义主题如 "AI", "新能源")
        # 提取用户输入中的有效词 (长度 >= 2 的英文词或中文词)
        raw_words = user_interest.replace(",", " ").replace("，", " ").split()
        for w in raw_words:
            w = w.strip()
            if len(w) >= 2 and w not in matched_keywords:
                matched_keywords.append(w)

        return matched_keywords

    def _discover_candidates(
        self,
        news_list: list[dict],
        user_interest: str,
        max_candidates: int,
    ) -> list[StockCandidate]:
        """
        从新闻中发现候选股票

        策略:
        1. 如果有用户兴趣，先匹配行业关键词
        2. 统计新闻中出现的股票代码频率
        3. 如果有用户兴趣，优先保留与兴趣相关的新闻中的股票
        4. 按频率 + 兴趣匹配度排序
        """
        # 统计新闻中提到的股票
        symbol_mentions: Counter = Counter()
        symbol_interest_match: dict[str, int] = {}  # 股票 -> 兴趣匹配次数

        # 解析用户兴趣为关键词列表
        interest_keywords = self._match_user_interest(user_interest)
        has_interest = len(interest_keywords) > 0

        watchlist = config.WATCHLIST if config.WATCHLIST else config.DEFAULT_WATCHLIST

        for news in news_list:
            headline = news.get("headline", "")
            headline_lower = headline.lower()

            # 检查这条新闻是否匹配用户兴趣
            news_matches_interest = False
            if has_interest:
                for kw in interest_keywords:
                    if kw.lower() in headline_lower:
                        news_matches_interest = True
                        break

            # 从公司新闻中提取股票代码
            sym = news.get("_symbol", "")
            if sym:
                symbol_mentions[sym] += 1
                if news_matches_interest:
                    symbol_interest_match[sym] = symbol_interest_match.get(sym, 0) + 1

            # 从标题中提取股票代码 (匹配 watchlist 中的代码)
            words = headline.split()
            for w in words:
                cleaned = w.strip(".,;:!?()[]{}").upper()
                if cleaned in watchlist:
                    symbol_mentions[cleaned] += 1
                    if news_matches_interest:
                        symbol_interest_match[cleaned] = symbol_interest_match.get(cleaned, 0) + 1

        # 如果新闻数据不足，用默认列表补充
        if len(symbol_mentions) < max_candidates:
            for sym in watchlist:
                if sym not in symbol_mentions:
                    symbol_mentions[sym] = 0
                if len(symbol_mentions) >= max_candidates * 2:
                    break

        # 排序: 如果有用户兴趣，兴趣匹配的股票优先; 否则纯按提及次数
        if has_interest:
            # 先按兴趣匹配次数降序, 再按总提及次数降序
            sorted_symbols = sorted(
                symbol_mentions.items(),
                key=lambda x: (symbol_interest_match.get(x[0], 0), x[1]),
                reverse=True,
            )
        else:
            sorted_symbols = symbol_mentions.most_common(max_candidates * 2)

        candidates = []
        for sym, count in sorted_symbols:
            if len(candidates) >= max_candidates:
                break

            # 简单的情绪分数 (基于提及次数和整体情绪)
            sentiment = 0.3 if count > 2 else 0.0

            # 如果有用户兴趣且该股票匹配了兴趣, 提升情绪分数
            interest_boost = symbol_interest_match.get(sym, 0)
            if interest_boost > 0:
                sentiment = min(0.5, sentiment + 0.1 * interest_boost)

            candidate = StockCandidate(
                symbol=sym,
                name=self._get_stock_name(sym),
                sentiment_score=sentiment,
                mention_count=count,
                key_events=[],
                plain_reason="",
            )
            candidates.append(candidate)

        return candidates

    def _enrich_candidates(self, candidates: list[StockCandidate]) -> list[StockCandidate]:
        """补充候选股票详情 (公司概况，并行获取)"""
        from src.utils.parallel import parallel_map

        if not candidates:
            return candidates

        def _fetch_profile(c: StockCandidate) -> dict | None:
            """获取单只股票的公司概况"""
            try:
                return data_collector.get_company_profile(c.symbol)
            except Exception as e:
                logger.debug(f"获取 {c.symbol} 概况失败: {e}")
                return None

        # 并行获取所有候选股票的公司概况
        profiles = parallel_map(
            _fetch_profile, candidates,
            max_workers=min(5, len(candidates)),
            desc="获取公司概况",
            timeout=30,
        )

        # 回填结果
        for c, profile in zip(candidates, profiles):
            if profile:
                c.name = profile.get("name", c.name)
                c.sector = profile.get("finnhubIndustry", "") or None
                c.plain_reason = self._generate_plain_reason(c, profile)
            else:
                c.plain_reason = (
                    f"{c.name} mentioned in {c.mention_count} recent news" if get_language() == "en"
                    else f"{c.name} 近期被新闻提及 {c.mention_count} 次"
                )

        return candidates

    def _generate_plain_reason(self, candidate: StockCandidate, profile: dict) -> str:
        """生成小白友好的推荐理由 (按当前 UI 语言)"""
        name = candidate.name or candidate.symbol
        is_en = get_language() == "en"
        sector = candidate.sector or ("Unknown sector" if is_en else "未知行业")
        mentions = candidate.mention_count

        parts = [f"{name}"]

        if mentions > 5:
            parts.append(
                f"heavily covered by recent news ({mentions} mentions)" if is_en
                else f"近期被新闻大量报道 ({mentions} 次)"
            )
        elif mentions > 2:
            parts.append(
                f"has some recent news coverage ({mentions} mentions)" if is_en
                else f"近期有一定新闻关注度 ({mentions} 次)"
            )
        else:
            parts.append(f"in the {sector} sector" if is_en else f"属于 {sector} 板块")

        if candidate.sentiment_score > 0.2:
            parts.append("market sentiment is positive" if is_en else "市场情绪偏积极")
        elif candidate.sentiment_score < -0.2:
            parts.append("market sentiment is cautious" if is_en else "市场情绪偏谨慎")
        else:
            parts.append("market sentiment is neutral" if is_en else "市场情绪中性")

        if is_en:
            # 每段首字母大写, 组成规范英文句子
            parts = [p[0].upper() + p[1:] if p else p for p in parts]
            return ". ".join(parts) + "."
        return "，".join(parts) + "。"

    def _extract_key_events(self, news_list: list[dict]) -> list[dict]:
        """提取关键事件"""
        events = []
        seen = set()

        for news in news_list[:20]:
            headline = news.get("headline", "")
            if not headline or headline in seen:
                continue
            seen.add(headline)

            events.append({
                "title": headline,
                "source": news.get("source", ""),
                "time": news.get("datetime", ""),
                "url": news.get("url", ""),
            })
            if len(events) >= 5:
                break

        return events

    def _build_plain_summary(
        self,
        sentiment_label: str,
        sentiment_score: float,
        candidates: list[StockCandidate],
        key_events: list[dict],
    ) -> str:
        """构建小白友好的调研总结"""
        # i18n: 按当前语言生成双语调研总结
        is_en = get_language() == "en"
        lines = []
        lines.append("📊 今日市场调研报告" if not is_en else "📊 Daily Market Research Report")
        lines.append(f"{'─' * 40}")
        lines.append(
            f"Market sentiment: {sentiment_label} ({sentiment_score:+.2f})" if is_en
            else f"市场情绪: {sentiment_label} ({sentiment_score:+.2f})"
        )
        lines.append("")

        if candidates:
            lines.append(
                f"🔍 Found {len(candidates)} noteworthy stocks:" if is_en
                else f"🔍 发现 {len(candidates)} 只值得关注的股票:"
            )
            for i, c in enumerate(candidates, 1):
                lines.append(f"  {i}. {c.symbol} ({c.name})")
                if c.plain_reason:
                    lines.append(f"     → {c.plain_reason}")
        else:
            lines.append("⚠️ No notable hot stocks found" if is_en else "⚠️ 未发现明显的热点股票")

        lines.append("")
        if key_events:
            lines.append("📰 Recent key events:" if is_en else "📰 近期关键事件:")
            for e in key_events[:3]:
                lines.append(f"  • {e['title']}")

        lines.append("")
        if is_en:
            lines.append("💡 Tip: Pick a stock you like and I'll generate a trading strategy and backtest it.")
            lines.append("⚠️ Disclaimer: For reference only, not investment advice.")
        else:
            lines.append("💡 建议: 可以选择感兴趣的股票，让我帮你生成交易策略并回测验证。")
            lines.append("⚠️ 风险提示: 以上分析仅供参考，不构成投资建议。")

        return "\n".join(lines)

    # 常用股票代码 -> 中文名映射 (优先查表, 避免测试依赖外部 API)
    STOCK_NAME_MAP = {
        "AAPL": "苹果", "MSFT": "微软", "GOOGL": "谷歌", "AMZN": "亚马逊",
        "NVDA": "英伟达", "META": "Meta", "TSLA": "特斯拉", "AMD": "AMD",
        "NFLX": "奈飞", "JPM": "摩根大通", "V": "Visa", "DIS": "迪士尼",
        "BABA": "阿里巴巴", "INTC": "英特尔", "CRM": "Salesforce",
        "ORCL": "甲骨文", "ADBE": "Adobe", "PYPL": "PayPal",
        "UBER": "Uber", "COIN": "Coinbase",
    }

    def _get_stock_name(self, symbol: str) -> str:
        """股票代码 -> 名称映射 (优先查中文名表, 找不到再从 Finnhub 获取)"""
        # 1. 先查本地中文名映射表 (确定性, 不依赖外部 API)
        if symbol in self.STOCK_NAME_MAP:
            return self.STOCK_NAME_MAP[symbol]

        # 2. 本地没有, 尝试从 Finnhub 获取公司名
        try:
            profile = data_collector.get_company_profile(symbol)
            if profile and profile.get("name"):
                return profile["name"]
        except Exception:
            pass

        # 3. 都没有, 返回原始代码
        return symbol

    def plain_explain(self, result: Any = None) -> str:
        """通俗解读"""
        if result is None:
            result = self._last_result
        if isinstance(result, ResearchReport):
            return result.plain_summary
        return "调研完成，请查看报告。"

    # ═══════════════════════════════════════════════════════
    # 加密货币调研 (Phase 8 新增)
    # ═══════════════════════════════════════════════════════

    def _execute_crypto_research(
        self,
        user_interest: str = "",
        max_candidates: int = None,
    ) -> ResearchReport:
        """
        加密货币市场调研

        流程:
        1. 获取加密货币新闻 (Finnhub crypto category)
        2. 分析市场情绪
        3. 发现候选币种 (新闻频率 + 价格变化)
        4. 补充市场数据 (市值/成交量)
        5. 生成调研报告

        Args:
            user_interest: 用户关注的加密货币主题 (如 "DeFi", "Layer2")
            max_candidates: 最大候选数

        Returns:
            ResearchReport (asset_type=CRYPTO, crypto_candidates 填充)
        """
        if max_candidates is None:
            max_candidates = min(config.MAX_CANDIDATES, 8)

        logger.info(f"🔒 开始加密货币市场调研 | 关注: {user_interest or '全市场'} | 候选上限: {max_candidates}")

        # 1. 获取加密货币新闻
        crypto_news = self._fetch_crypto_news()
        logger.info(f"  获取加密货币新闻: {len(crypto_news)} 条")

        # 2. 情绪分析 (复用股票情绪分析逻辑)
        sentiment_score, sentiment_label = self._analyze_sentiment(crypto_news)
        logger.info(f"  加密货币市场情绪: {sentiment_label} ({sentiment_score:+.2f})")

        # 3. 获取市场总览
        overview = data_collector.get_crypto_market_overview()
        logger.info(f"  市场总览: 情绪={overview.get('sentiment', 'N/A')}")

        # 4. 发现候选币种
        candidates = self._discover_crypto_candidates(
            crypto_news, overview, user_interest, max_candidates
        )
        logger.info(f"  加密货币候选: {len(candidates)} 个")

        # 5. 补充候选币种详情
        candidates = self._enrich_crypto_candidates(candidates)

        # 6. 提取关键事件
        key_events = self._extract_key_events(crypto_news)

        # 7. 生成小白解读
        plain_summary = self._build_crypto_summary(
            sentiment_label, sentiment_score, candidates, key_events, overview
        )

        report = ResearchReport(
            report_id=f"crypto_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            date=datetime.now(),
            crypto_candidates=candidates,
            market_sentiment=sentiment_score,
            key_events=key_events,
            asset_type=AssetType.CRYPTO,
            plain_summary=plain_summary,
        )

        logger.info(f"🔒 加密货币调研报告生成完成: {report.report_id}")
        return report

    def _fetch_crypto_news(self) -> list[dict]:
        """获取加密货币新闻"""
        try:
            news = data_collector.get_crypto_news(count=50)
            return news if news else []
        except Exception as e:
            logger.warning(f"获取加密货币新闻失败: {e}")
            return []

    def _discover_crypto_candidates(
        self,
        news_list: list[dict],
        overview: dict,
        user_interest: str,
        max_candidates: int,
    ) -> list[CryptoCandidate]:
        """
        从新闻和市场数据中发现候选加密货币

        策略:
        1. 统计新闻中提到的币种频率
        2. 结合市场总览中的涨跌幅数据
        3. 如果有用户兴趣, 优先匹配相关币种
        4. 按综合得分排序
        """
        # 统计新闻中提到的币种
        symbol_mentions: Counter = Counter()
        symbol_interest_match: dict[str, int] = {}

        # 解析用户兴趣
        interest_keywords = self._match_crypto_interest(user_interest)
        has_interest = len(interest_keywords) > 0

        # 默认交易对列表
        default_pairs = config.get_crypto_pairs()

        for news in news_list:
            headline = news.get("headline", "")
            headline_lower = headline.lower()

            # 检查是否匹配用户兴趣
            news_matches_interest = False
            if has_interest:
                for kw in interest_keywords:
                    if kw.lower() in headline_lower:
                        news_matches_interest = True
                        break

            # 从标题中提取币种代码
            for pair in default_pairs:
                base = pair.split("/")[0]  # 如 "BTC"
                if base.lower() in headline_lower or base in headline:
                    symbol_mentions[pair] += 1
                    if news_matches_interest:
                        symbol_interest_match[pair] = symbol_interest_match.get(pair, 0) + 1

        # 从市场总览中获取价格变化
        price_changes = {}
        for item in overview.get("top_gainers", []) + overview.get("top_losers", []):
            price_changes[item["symbol"]] = item

        # 如果新闻数据不足, 用默认列表补充
        for pair in default_pairs:
            if pair not in symbol_mentions:
                symbol_mentions[pair] = 0

        # 排序: 兴趣匹配 > 新闻频率 > 价格变化幅度
        if has_interest:
            sorted_symbols = sorted(
                symbol_mentions.items(),
                key=lambda x: (symbol_interest_match.get(x[0], 0), x[1]),
                reverse=True,
            )
        else:
            sorted_symbols = sorted(
                symbol_mentions.items(),
                key=lambda x: (x[1], abs(price_changes.get(x[0], {}).get("change_pct", 0))),
                reverse=True,
            )

        candidates = []
        for sym, count in sorted_symbols:
            if len(candidates) >= max_candidates:
                break

            price_info = price_changes.get(sym, {})
            sentiment = 0.3 if count > 2 else 0.0
            change_pct = price_info.get("change_pct", 0)
            if change_pct > 5:
                sentiment = min(0.5, sentiment + 0.2)
            elif change_pct < -5:
                sentiment = max(-0.3, sentiment - 0.2)

            interest_boost = symbol_interest_match.get(sym, 0)
            if interest_boost > 0:
                sentiment = min(0.6, sentiment + 0.1 * interest_boost)

            candidate = CryptoCandidate(
                symbol=sym,
                name=self._get_crypto_name(sym),
                price_change_24h=change_pct,
                sentiment_score=sentiment,
                mention_count=count,
                key_events=[],
                plain_reason="",
            )
            candidates.append(candidate)

        return candidates

    def _enrich_crypto_candidates(
        self, candidates: list[CryptoCandidate]
    ) -> list[CryptoCandidate]:
        """补充候选币种详情 (获取最新价格和24h数据)"""
        if not candidates:
            return candidates

        for c in candidates:
            try:
                price = data_collector.get_crypto_latest_price(c.symbol)
                if price and price > 0:
                    c.market_cap = 0.0  # Alpaca 不提供市值, 留空

                # 生成推荐理由
                c.plain_reason = self._generate_crypto_reason(c)
            except Exception as e:
                logger.debug(f"获取 {c.symbol} 详情失败: {e}")
                c.plain_reason = (
                    f"{c.name} mentioned in {c.mention_count} recent news" if get_language() == "en"
                    else f"{c.name} 近期被新闻提及 {c.mention_count} 次"
                )

        return candidates

    def _generate_crypto_reason(self, candidate: CryptoCandidate) -> str:
        """生成加密货币候选的小白友好推荐理由 (按当前 UI 语言)"""
        name = candidate.name or candidate.symbol
        is_en = get_language() == "en"
        mentions = candidate.mention_count
        change = candidate.price_change_24h

        parts = [f"{name}"]

        if mentions > 5:
            parts.append(
                f"heavily mentioned in recent news ({mentions} mentions)" if is_en
                else f"近期被大量新闻提及 ({mentions} 次)"
            )
        elif mentions > 2:
            parts.append(
                f"has some recent news coverage ({mentions} mentions)" if is_en
                else f"近期有一定新闻关注度 ({mentions} 次)"
            )
        else:
            parts.append("a mainstream cryptocurrency" if is_en else "属于主流加密货币")

        if change > 5:
            parts.append(
                f"24h change {change:+.1f}%, strong momentum" if is_en
                else f"24h 涨幅 {change:+.1f}%, 动能强劲"
            )
        elif change < -5:
            parts.append(
                f"24h change {change:+.1f}%, possible rebound opportunity" if is_en
                else f"24h 跌幅 {change:+.1f}%, 可能存在反弹机会"
            )
        else:
            parts.append("24h price stable" if is_en else "24h 价格波动平稳")

        if candidate.sentiment_score > 0.2:
            parts.append("market sentiment is positive" if is_en else "市场情绪偏积极")
        elif candidate.sentiment_score < -0.2:
            parts.append("market sentiment is cautious" if is_en else "市场情绪偏谨慎")
        else:
            parts.append("market sentiment is neutral" if is_en else "市场情绪中性")

        if is_en:
            # 每段首字母大写, 组成规范英文句子
            parts = [p[0].upper() + p[1:] if p else p for p in parts]
            return ". ".join(parts) + "."
        return "，".join(parts) + "。"

    def _match_crypto_interest(self, user_interest: str) -> list[str]:
        """将用户兴趣匹配到加密货币关键词"""
        if not user_interest or not user_interest.strip():
            return []

        interest_lower = user_interest.lower().strip()
        matched: list[str] = []

        for category, keywords in self.CRYPTO_KEYWORDS.items():
            if category in user_interest or category.lower() in interest_lower:
                matched.extend(keywords)
                continue
            for kw in keywords:
                if kw.lower() in interest_lower:
                    matched.extend(keywords)
                    break

        # 用户输入本身也作为关键词
        raw_words = user_interest.replace(",", " ").replace("，", " ").split()
        for w in raw_words:
            w = w.strip()
            if len(w) >= 2 and w not in matched:
                matched.append(w)

        return matched

    def _get_crypto_name(self, symbol: str) -> str:
        """交易对 -> 币种名称映射"""
        if symbol in self.CRYPTO_NAME_MAP:
            return self.CRYPTO_NAME_MAP[symbol]
        # 从交易对提取 base currency
        base = symbol.split("/")[0] if "/" in symbol else symbol
        return base

    def _build_crypto_summary(
        self,
        sentiment_label: str,
        sentiment_score: float,
        candidates: list[CryptoCandidate],
        key_events: list[dict],
        overview: dict,
    ) -> str:
        """构建加密货币调研的小白友好总结"""
        lines = []
        lines.append(f"📊 加密货币市场调研报告")
        lines.append(f"{'─' * 40}")
        lines.append(
            f"Market sentiment: {sentiment_label} ({sentiment_score:+.2f})" if get_language() == "en"
            else f"市场情绪: {sentiment_label} ({sentiment_score:+.2f})"
        )
        lines.append(f"市场总览: {overview.get('sentiment', 'N/A')}")
        lines.append("")

        if candidates:
            lines.append(f"🔍 发现 {len(candidates)} 个值得关注的币种:")
            for i, c in enumerate(candidates, 1):
                lines.append(f"  {i}. {c.symbol} ({c.name})")
                if c.plain_reason:
                    lines.append(f"     -> {c.plain_reason}")
        else:
            lines.append("⚠️ 未发现明显的热点币种")

        lines.append("")
        if key_events:
            lines.append("📰 近期关键事件:")
            for e in key_events[:3]:
                lines.append(f"  • {e['title']}")

        lines.append("")
        lines.append("💡 建议: 可以选择感兴趣的币种, 让我帮你生成加密货币交易策略并回测验证。")
        lines.append("⚠️ 风险提示: 加密货币波动性远高于股票, 请务必使用 Paper Trading 模式验证策略。")

        return "\n".join(lines)
