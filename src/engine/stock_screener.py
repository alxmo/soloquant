"""
自然语言选股引擎
=====================================================
v0.5.0 新增

功能:
- 从全美股实时行情中筛选符合条件的股票
- 多维度综合评分: 技术面 + 基本面 + 情绪面 + 动量面
- LLM 辅助判断: 让 LLM 根据候选股数据给出"潜力"判断和通俗推荐理由
- 降级策略: AKShare 不可用 -> 扩展关注列表; Finnhub 不可用 -> 跳过基本面/情绪; LLM 不可用 -> 纯评分排序

使用示例:
    from src.engine.stock_screener import stock_screener

    result = stock_screener.screen({
        "count": 5,
        "price_max": 10,
        "keywords": "潜力",
    })
    print(result.plain_summary)
"""

import time
import numpy as np
import pandas as pd
from loguru import logger

from src.data.collector import data_collector
from src.data.akshare_collector import akshare_collector
from src.data.finnhub_client import finnhub_client
from src.models import StockPick, StockScreenResult
from src.utils.config import config

try:
    from src.ui.i18n import get_language
except ImportError:  # 避免循环依赖时的回退
    def get_language():
        return "zh"


class StockScreener:
    """自然语言选股引擎"""

    # AKShare stock_us_spot_em 列名映射 -> 标准化英文
    SPOT_COLUMN_MAP = {
        "名称": "name",
        "最新价": "price",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "成交量": "volume",
        "成交额": "turnover",
        "振幅": "amplitude",
        "换手率": "turnover_rate",
        "简称": "symbol_short",
        "编码": "symbol_code",
        "最高价": "high",
        "最低价": "low",
        "开盘价": "open",
        "昨收价": "prev_close",
        "总市值": "market_cap",
        "市盈率": "pe_ratio",
    }

    # 扩展关注列表 (全量扫描失败时的降级池, 约 80 只覆盖各行业)
    FALLBACK_UNIVERSE = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "JPM",
        "V", "DIS", "BABA", "INTC", "CRM", "ORCL", "ADBE", "PYPL", "UBER", "COIN",
        "F", "GM", "NIO", "XPEV", "PLTR", "SOFI", "WBD", "SNAP", "PINS", "RBLX",
        "SQ", "SHOP", "SQ", "ROKU", "ZM", "DOCU", "TWTR", "SPCE", "NKLA", "FCEL",
        "PLUG", "BLNK", "SIRI", "BBD", "ITUB", "VALE", "PBR", "GOLD", "AUY", "NEM",
        "WBA", "DDS", "M", "KSS", "JCP", "RAD", "SURG", "GE", "BAC", "C",
        "KEY", "RF", "HBAN", "FITB", "CFG", "NCLH", "CCL", "RCL", "DAL", "AAL",
        "UAL", "LUV", "SAVE", "SABR", "MARA", "RIOT", "HUT", "BTBT", "WISH", "CGC",
    ]

    def __init__(self):
        from src.llm.llm_client import llm_client
        self._llm = llm_client

    # ═══════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════

    def screen(self, params: dict, query: str = "") -> StockScreenResult:
        """
        执行选股

        Args:
            params: 筛选参数 (由 LLM 从自然语言提取)
                - count: int         选股数量 (默认 5)
                - price_min: float   最低价格 (默认 0)
                - price_max: float   最高价格 (默认不限)
                - sector: str        行业偏好 (如 "科技")
                - keywords: str      关键词 (如 "潜力")
            query: 用户原始查询文本

        Returns:
            StockScreenResult
        """
        count = int(params.get("count", 5))
        price_min = float(params.get("price_min", 0))
        price_max = float(params.get("price_max", 0))  # 0 = 不限
        sector = params.get("sector", "")
        keywords = params.get("keywords", "")

        logger.info(
            f"🔍 选股引擎启动 | 数量={count} | 价格={price_min}~{price_max or '不限'} | "
            f"行业={sector or '不限'} | 关键词={keywords or '无'}"
        )

        # 1. 获取全美股实时行情
        universe_df = self._fetch_universe()
        total_scanned = len(universe_df)

        if total_scanned == 0:
            return self._empty_result(query, params, "无法获取美股行情数据，请检查网络连接")

        # 2. 价格筛选
        filtered = self._filter_by_price(universe_df, price_min, price_max)
        total_filtered = len(filtered)

        if total_filtered == 0:
            range_desc = f"{price_min}~{price_max}美元" if price_max else f">{price_min}美元"
            return self._empty_result(query, params, f"在{range_desc}价格范围内没有找到股票")

        logger.info(f"  价格筛选: {total_scanned} -> {total_filtered} 只")

        # 3. 预筛选: 按成交额取 Top N 进入详细分析
        universe_size = config.SCREEN_UNIVERSE_SIZE
        if "turnover" in filtered.columns and filtered["turnover"].notna().any():
            filtered = filtered.nlargest(universe_size, "turnover")
        else:
            filtered = filtered.head(universe_size)

        logger.info(f"  预筛选: 取成交额 Top {len(filtered)} 进入详细分析")

        # 4. 多维度评分
        scored_picks = self._score_stocks(filtered, sector)

        if not scored_picks:
            return self._empty_result(query, params, "评分阶段未能生成结果")

        # 5. 按综合评分排序取 Top count
        scored_picks.sort(key=lambda p: p.composite_score, reverse=True)
        top_picks = scored_picks[:count]

        # 6. LLM 辅助判断
        top_picks = self._llm_enhance(top_picks, params, query)

        # 7. 生成小白解读
        plain_summary = self._build_summary(top_picks, params, total_scanned, total_filtered)

        result = StockScreenResult(
            query=query,
            params=params,
            picks=top_picks,
            total_scanned=total_scanned,
            total_filtered=total_filtered,
            plain_summary=plain_summary,
        )

        logger.info(f"🔍 选股完成: {len(top_picks)} 只入选 (扫描 {total_scanned}, 筛选后 {total_filtered})")
        return result

    # ═══════════════════════════════════════════════════════
    # Step 1: 获取全美股行情
    # ═══════════════════════════════════════════════════════

    def _fetch_universe(self) -> pd.DataFrame:
        """
        获取全美股实时行情快照

        优先 AKShare stock_us_spot_em, 失败则降级到扩展关注列表
        """
        # 尝试全量扫描 (带多源降级, 应对 AKShare 连接不稳定)
        try:
            logger.info("  尝试获取全美股实时行情 (AKShare + 多源降级)...")
            raw = akshare_collector.get_us_stock_spot_with_fallback()

            if raw is not None and len(raw) > 0:
                df = self._normalize_spot(raw)
                logger.info(f"  全美股行情: {len(df)} 只")
                return df
        except Exception as e:
            logger.warning(f"  全美股行情获取失败: {e}")

        # 降级: 从扩展关注列表逐只获取最新价格
        logger.info("  降级: 使用扩展关注列表获取行情...")
        return self._fetch_fallback_universe()

    def _normalize_spot(self, raw: pd.DataFrame) -> pd.DataFrame:
        """标准化 AKShare spot 数据"""
        df = raw.rename(columns=self.SPOT_COLUMN_MAP)

        # 构造标准 symbol: 从编码提取 (如 105.AAPL -> AAPL)
        if "symbol_code" in df.columns:
            df["symbol"] = df["symbol_code"].astype(str).str.split(".").str[-1]
        elif "symbol_short" in df.columns:
            df["symbol"] = df["symbol_short"]
        else:
            df["symbol"] = ""

        # 数值列转换
        numeric_cols = ["price", "pct_change", "volume", "turnover",
                        "market_cap", "pe_ratio", "high", "low", "open", "prev_close"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 过滤无效价格
        df = df[df["price"].notna() & (df["price"] > 0)]

        # 只保留需要的列
        keep = ["symbol", "name", "price", "pct_change", "volume", "turnover",
                "market_cap", "pe_ratio"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()

        # 去重 (按 symbol)
        df = df.drop_duplicates(subset=["symbol"], keep="first")
        return df.reset_index(drop=True)

    def _fetch_fallback_universe(self) -> pd.DataFrame:
        """降级: 从扩展关注列表获取最新价格"""
        rows = []
        for sym in self.FALLBACK_UNIVERSE:
            try:
                price = akshare_collector.get_latest_price(sym)
                if price and price > 0:
                    rows.append({
                        "symbol": sym,
                        "name": sym,
                        "price": price,
                        "pct_change": None,
                        "volume": None,
                        "turnover": None,
                        "market_cap": None,
                        "pe_ratio": None,
                    })
                time.sleep(0.3)
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    # ═══════════════════════════════════════════════════════
    # Step 2: 价格筛选
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _filter_by_price(df: pd.DataFrame, price_min: float, price_max: float) -> pd.DataFrame:
        """按价格范围筛选"""
        mask = df["price"] >= price_min
        if price_max > 0:
            mask &= df["price"] <= price_max
        return df[mask].reset_index(drop=True)

    # ═══════════════════════════════════════════════════════
    # Step 3: 多维度评分
    # ═══════════════════════════════════════════════════════

    def _score_stocks(self, df: pd.DataFrame, sector: str) -> list[StockPick]:
        """
        对候选股票进行四维评分

        Returns:
            list[StockPick]
        """
        picks = []
        finnhub_ok = finnhub_client.available
        total = len(df)

        for i, row in df.iterrows():
            symbol = row["symbol"]
            price = float(row.get("price", 0))
            name = str(row.get("name", symbol))

            logger.debug(f"  评分 [{i+1}/{total}] {symbol} @ ${price:.2f}")

            # 技术面评分
            tech_score, tech_data = self._score_technical(symbol)

            # 动量面评分
            mom_score, mom_data = self._score_momentum(symbol, row)

            # 基本面评分 (需要 Finnhub)
            if finnhub_ok:
                fund_score, fund_data = self._score_fundamental(symbol, row)
            else:
                fund_score, fund_data = 50.0, {}  # 中性

            # 情绪面评分 (需要 Finnhub)
            if finnhub_ok:
                sent_score, sent_data = self._score_sentiment(symbol)
            else:
                sent_score, sent_data = 50.0, {}

            # 行业匹配加分
            if sector and fund_data.get("sector", "").lower().find(sector.lower()) >= 0:
                fund_score = min(fund_score + 10, 100)

            # 综合评分 (加权)
            composite = (
                tech_score * config.SCREEN_WEIGHT_TECH
                + fund_score * config.SCREEN_WEIGHT_FUND
                + sent_score * config.SCREEN_WEIGHT_SENT
                + mom_score * config.SCREEN_WEIGHT_MOM
            )

            pick = StockPick(
                symbol=symbol,
                name=name,
                price=round(price, 2),
                sector=fund_data.get("sector", ""),
                composite_score=round(composite, 1),
                tech_score=round(tech_score, 1),
                fundamental_score=round(fund_score, 1),
                sentiment_score=round(sent_score, 1),
                momentum_score=round(mom_score, 1),
            )
            picks.append(pick)

            # 避免 API 限流
            if finnhub_ok and i < total - 1:
                time.sleep(0.2)

        return picks

    def _score_technical(self, symbol: str) -> tuple[float, dict]:
        """
        技术面评分 (0~100)

        指标: RSI(14), MA比率(5/20), 量比
        """
        try:
            df = akshare_collector.get_us_stock_daily(symbol)
            if len(df) < 30:
                return 50.0, {}

            close = df["close"]
            volume = df["volume"]

            # RSI
            rsi = self._calc_rsi(close, 14)

            # MA 比率 (短期均线/长期均线)
            ma5 = close.rolling(5).mean().iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma_ratio = ma5 / ma20 if ma20 > 0 else 1.0

            # 量比
            vol_avg = volume.rolling(20).mean().iloc[-1]
            vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 1.0

            # 评分逻辑
            score = 50.0

            # RSI: 30-50 超卖区 (买入机会), 50-70 正常, >70 超买
            if rsi < 30:
                score += 15  # 深度超卖
            elif rsi < 50:
                score += 10  # 超卖区
            elif rsi > 70:
                score -= 10  # 超买
            elif 50 <= rsi <= 65:
                score += 5   # 健康上升

            # MA 比率: >1 短期均线上穿长期均线 (多头排列)
            if ma_ratio > 1.05:
                score += 10
            elif ma_ratio > 1.0:
                score += 5
            elif ma_ratio < 0.95:
                score -= 5

            # 量比: 1.5-3 放量上涨 (有资金关注)
            if 1.5 <= vol_ratio <= 3.0:
                score += 10
            elif vol_ratio > 3.0:
                score += 5  # 巨量 (可能异常)
            elif vol_ratio < 0.5:
                score -= 5  # 缩量

            score = max(0, min(100, score))
            return score, {"rsi": round(rsi, 1), "ma_ratio": round(ma_ratio, 3), "vol_ratio": round(vol_ratio, 2)}

        except Exception as e:
            logger.debug(f"  {symbol} 技术面评分失败: {e}")
            return 50.0, {}

    def _score_momentum(self, symbol: str, spot_row: pd.Series) -> tuple[float, dict]:
        """
        动量面评分 (0~100)

        指标: 5日收益率, 20日收益率, 当日涨跌幅
        """
        try:
            df = akshare_collector.get_us_stock_daily(symbol)
            if len(df) < 25:
                # 数据不足, 用当日涨跌幅替代
                pct = float(spot_row.get("pct_change", 0) or 0)
                return 50 + max(-20, min(20, pct * 2)), {"pct_today": round(pct, 2)}

            close = df["close"]

            # 5日收益率
            if len(close) >= 6:
                ret_5d = (close.iloc[-1] / close.iloc[-6]) - 1
            else:
                ret_5d = 0.0

            # 20日收益率
            if len(close) >= 21:
                ret_20d = (close.iloc[-1] / close.iloc[-21]) - 1
            else:
                ret_20d = 0.0

            # 评分: 正收益加分, 负收益减分
            score = 50.0
            score += max(-20, min(20, ret_5d * 100))   # 5日收益: ±20
            score += max(-15, min(15, ret_20d * 50))   # 20日收益: ±15

            # 当日涨跌
            pct_today = float(spot_row.get("pct_change", 0) or 0)
            score += max(-10, min(10, pct_today))

            score = max(0, min(100, score))
            return score, {
                "ret_5d": round(ret_5d * 100, 2),
                "ret_20d": round(ret_20d * 100, 2),
                "pct_today": round(pct_today, 2),
            }

        except Exception as e:
            logger.debug(f"  {symbol} 动量面评分失败: {e}")
            return 50.0, {}

    def _score_fundamental(self, symbol: str, spot_row: pd.Series) -> tuple[float, dict]:
        """
        基本面评分 (0~100)

        指标: P/E 比率, 分析师推荐, 营收增长 (Finnhub)
        """
        try:
            score = 50.0
            data = {}

            # P/E 比率 (从 spot 数据)
            pe = spot_row.get("pe_ratio", None)
            if pe is not None and not np.isnan(pe) and pe > 0:
                data["pe"] = round(float(pe), 1)
                if pe < 15:
                    score += 10  # 低估
                elif pe < 25:
                    score += 5   # 合理
                elif pe > 50:
                    score -= 5   # 高估

            # 公司概况 (Finnhub)
            profile = data_collector.get_company_profile(symbol)
            if profile:
                sector = profile.get("finnhubIndustry", "")
                data["sector"] = sector
                market_cap = profile.get("marketCapitalization", 0)
                if market_cap:
                    data["market_cap"] = market_cap

            # 分析师推荐 (Finnhub)
            recs = data_collector.get_recommendations(symbol)
            if recs and len(recs) > 0:
                rec = recs[0]  # 最新一期
                buy = rec.get("buy", 0)
                hold = rec.get("hold", 0)
                sell = rec.get("sell", 0)
                total = buy + hold + sell
                if total > 0:
                    buy_ratio = buy / total
                    sell_ratio = sell / total
                    data["buy_ratio"] = round(buy_ratio, 2)
                    score += max(-15, min(20, (buy_ratio - 0.3) * 40))
                    score -= max(-10, min(10, sell_ratio * 20))

            # 基本面财务指标 (Finnhub)
            financials = data_collector.get_financials(symbol)
            if financials:
                metrics = financials.get("metric", {})
                revenue_growth = metrics.get("revenueGrowth5Y", 0) or 0
                if revenue_growth:
                    data["rev_growth_5y"] = round(revenue_growth, 2)
                    score += max(-10, min(15, revenue_growth * 30))

                roe = metrics.get("roeTTM", 0) or 0
                if roe:
                    data["roe"] = round(roe, 2)
                    score += max(-5, min(10, roe * 10))

            score = max(0, min(100, score))
            return score, data

        except Exception as e:
            logger.debug(f"  {symbol} 基本面评分失败: {e}")
            return 50.0, {}

    def _score_sentiment(self, symbol: str) -> tuple[float, dict]:
        """
        情绪面评分 (0~100)

        指标: 新闻关键词情绪分析
        """
        try:
            news = data_collector.get_company_news(symbol, days=7)
            if not news:
                return 50.0, {}

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
            for n in news[:10]:
                text = (n.get("headline", "") + " " + n.get("summary", "")).lower()
                pos_count += sum(1 for w in positive_words if w in text)
                neg_count += sum(1 for w in negative_words if w in text)

            total = pos_count + neg_count
            if total == 0:
                return 50.0, {"news_count": len(news)}

            score = 50 + ((pos_count - neg_count) / total) * 40
            score = max(0, min(100, score))
            return score, {
                "news_count": len(news),
                "positive": pos_count,
                "negative": neg_count,
            }

        except Exception as e:
            logger.debug(f"  {symbol} 情绪面评分失败: {e}")
            return 50.0, {}

    # ═══════════════════════════════════════════════════════
    # Step 4: LLM 辅助判断
    # ═══════════════════════════════════════════════════════

    def _llm_enhance(self, picks: list[StockPick], params: dict, query: str) -> list[StockPick]:
        """
        让 LLM 根据候选股数据判断"潜力"并生成通俗推荐理由

        降级: LLM 不可用时跳过, 使用模板理由
        """
        if not picks:
            return picks

        # LLM 不可用 -> 模板理由
        if not self._llm.is_available():
            for p in picks:
                p.llm_reason = self._template_reason(p, params)
            return picks

        try:
            # 构建候选股摘要
            stock_summaries = []
            for i, p in enumerate(picks):
                stock_summaries.append(
                    f"{i+1}. {p.symbol} ({p.name}) - 价格 ${p.price}\n"
                    f"   综合{p.composite_score} | 技术{p.tech_score} | 基本面{p.fundamental_score} | "
                    f"情绪{p.sentiment_score} | 动量{p.momentum_score}"
                )

            keywords = params.get("keywords", "")
            # 按当前 UI 语言生成推荐理由 (zh=中文 / en=英文)
            if get_language() == "en":
                system_prompt = (
                    "You are a US stock investment analysis assistant. Based on the provided stock scoring data, "
                    "generate a plain-English recommendation reason (30-50 words) for each stock.\n"
                    "Requirements:\n"
                    "1. Analyze why the stock has potential based on dimension scores\n"
                    "2. Use plain language that beginners can understand\n"
                    f"3. If the user mentioned '{keywords}', echo it in the reason\n"
                    "4. Return as a JSON array, each element contains symbol and reason\n"
                    'Format: [{"symbol":"AAPL","reason":"..."}, ...]'
                )
            else:
                system_prompt = (
                    "你是一个美股投资分析助手。根据提供的股票评分数据，"
                    "为每只股票生成一段通俗的中文推荐理由（30-50字）。\n"
                    "要求：\n"
                    "1. 结合各维度评分分析为什么有潜力\n"
                    "2. 语言通俗，小白也能理解\n"
                    f"3. 如果用户提到了'{keywords}'，在理由中呼应\n"
                    "4. 以 JSON 数组格式返回，每个元素含 symbol 和 reason\n"
                    '格式: [{"symbol":"AAPL","reason":"..."}, ...]'
                )

            user_content = (
                f"用户需求: {query or '选股'}\n"
                f"筛选条件: 数量={params.get('count',5)}, "
                f"价格范围={params.get('price_min',0)}~{params.get('price_max','不限')}美元\n\n"
                f"候选股票评分:\n" + "\n".join(stock_summaries)
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            result = self._llm.chat_json(messages, temperature=0.3)

            # 解析 LLM 返回的推荐理由
            if isinstance(result, list):
                reason_map = {item.get("symbol", ""): item.get("reason", "") for item in result}
            elif isinstance(result, dict) and "reasons" in result:
                reason_map = {item.get("symbol", ""): item.get("reason", "") for item in result["reasons"]}
            else:
                reason_map = {}

            for p in picks:
                p.llm_reason = reason_map.get(p.symbol, "") or self._template_reason(p, params)

            logger.info(f"  LLM 辅助判断完成: {len(reason_map)} 只股票获得推荐理由")

        except Exception as e:
            logger.warning(f"  LLM 辅助判断失败，使用模板理由: {e}")
            for p in picks:
                p.llm_reason = self._template_reason(p, params)

        return picks

    @staticmethod
    def _template_reason(pick: StockPick, params: dict) -> str:
        """LLM 不可用时的模板推荐理由 (按当前 UI 语言生成)"""
        parts = []
        is_en = get_language() == "en"

        # 找最高分维度
        scores = {
            "技术面": pick.tech_score,
            "基本面": pick.fundamental_score,
            "情绪面": pick.sentiment_score,
            "动量": pick.momentum_score,
        }
        dim_names = {
            "技术面": ("技术面", "Technical"),
            "基本面": ("基本面", "Fundamentals"),
            "情绪面": ("情绪面", "Sentiment"),
            "动量": ("动量", "Momentum"),
        }
        best_dim = max(scores, key=scores.get)
        best_name = dim_names[best_dim][1] if is_en else dim_names[best_dim][0]

        if scores[best_dim] >= 65:
            if is_en:
                parts.append(f"{best_name} stands out ({scores[best_dim]:.0f} pts)")
            else:
                parts.append(f"{best_dim}表现突出({scores[best_dim]:.0f}分)")
        elif pick.composite_score >= 60:
            if is_en:
                parts.append(f"Balanced across dimensions, composite score {pick.composite_score:.0f}")
            else:
                parts.append(f"各维度均衡发展，综合评分{pick.composite_score:.0f}")
        else:
            if is_en:
                parts.append(f"Composite score {pick.composite_score:.0f}, worth watching")
            else:
                parts.append(f"综合评分{pick.composite_score:.0f}，有一定关注价值")

        if pick.momentum_score >= 60:
            parts.append("Strong recent momentum" if is_en else "近期动量较强")
        if pick.fundamental_score >= 60:
            parts.append("Solid fundamentals" if is_en else "基本面稳健")

        return ", ".join(parts) + "." if is_en else "，".join(parts) + "。"

    # ═══════════════════════════════════════════════════════
    # Step 5: 生成小白解读
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _build_summary(
        picks: list[StockPick],
        params: dict,
        total_scanned: int,
        total_filtered: int,
    ) -> str:
        """构建小白友好的选股结果解读"""
        lines = []
        lines.append("📊 AI 智能选股结果")
        lines.append(f"{'─' * 40}")

        count = params.get("count", 5)
        price_max = params.get("price_max", 0)
        price_min = params.get("price_min", 0)
        if price_max:
            lines.append(f"筛选条件: {price_min}~{price_max}美元, 选 {count} 只")
        else:
            lines.append(f"筛选条件: >{price_min}美元, 选 {count} 只")
        lines.append(f"扫描: {total_scanned} 只 -> 筛选后: {total_filtered} 只 -> 入选: {len(picks)} 只")
        lines.append("")

        if not picks:
            lines.append("⚠️ 未找到符合条件的股票")
            return "\n".join(lines)

        for i, p in enumerate(picks, 1):
            lines.append(f"  {i}. {p.symbol} ({p.name}) - ${p.price}")
            lines.append(f"     综合: {p.composite_score} | 技术: {p.tech_score} | 基本面: {p.fundamental_score} | 情绪: {p.sentiment_score} | 动量: {p.momentum_score}")
            if p.llm_reason:
                lines.append(f"     💡 {p.llm_reason}")

        lines.append("")
        lines.append("💡 建议: 选择感兴趣的股票，我可以帮你生成交易策略并回测验证。")
        lines.append("⚠️ 风险提示: 以上分析仅供参考，不构成投资建议。")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _empty_result(query: str, params: dict, reason: str) -> StockScreenResult:
        """生成空结果"""
        return StockScreenResult(
            query=query,
            params=params,
            picks=[],
            total_scanned=0,
            total_filtered=0,
            plain_summary=f"⚠️ {reason}",
        )

    @staticmethod
    def _calc_rsi(prices: pd.Series, period: int = 14) -> float:
        """计算 RSI 指标"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean().iloc[-1]
        avg_loss = loss.rolling(window=period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


# 全局单例
stock_screener = StockScreener()
