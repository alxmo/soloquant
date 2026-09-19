"""
加密货币信号生成器
=====================================================
Phase 8: 加密货币量化交易扩展

功能:
- 基于模型预测生成加密货币交易信号
- 动量信号 (24h 交易, 支持小时级数据)
- 复用 SignalGenerator 的特征计算框架, 数据源切换为 CryptoDataCollector

与股票信号生成器的区别:
1. 数据源: 使用 crypto_collector 而非 akshare_collector
2. 时间周期: 支持小时级 lookback (24/7 交易)
3. 资产标记: 信号 asset_type = AssetType.CRYPTO
4. 仓位限制: 使用加密货币专属 max_position
"""

from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from src.data.crypto_collector import crypto_collector
from src.models import AssetType, Direction, StrategyConfig, TradeSignal
from src.utils.config import config
from src.utils.parallel import parallel_map


class CryptoSignalGenerator:
    """加密货币信号生成器"""

    # 模型缓存最大数量
    _MODEL_CACHE_MAX = 10

    def __init__(self):
        self.collector = crypto_collector
        # 复用股票信号生成器的模型加载逻辑
        from src.engine.signal_generator import signal_generator
        self._stock_sg = signal_generator

    def generate_signals(
        self,
        strategy: StrategyConfig,
        model_path: str,
        symbols: list[str],
        top_k: int = None,
    ) -> list[TradeSignal]:
        """
        生成加密货币交易信号

        流程:
        1. 加载模型 (复用股票信号生成器的模型缓存)
        2. 对币种池计算特征 (数据来自 crypto_collector)
        3. 模型预测打分
        4. 选 Top-K 做多, 底部卖出
        5. 生成信号 (asset_type=CRYPTO)

        Args:
            strategy: 策略配置
            model_path: 模型文件路径
            symbols: 加密货币交易对列表 (如 ["BTC/USD", "ETH/USD"])
            top_k: 选前 K 个做多
        """
        if top_k is None:
            top_k = min(config.TOP_K_SIGNALS, len(symbols))

        logger.info(f"🔒 加密货币信号生成: {strategy.name}, 模型={strategy.model_type}, "
                    f"币种数={len(symbols)}, top_k={top_k}")

        # 加载模型 (复用股票信号生成器的 LRU 缓存)
        model = self._stock_sg._load_model(model_path, strategy.model_type)
        if model is None:
            logger.error("模型加载失败, 无法生成加密货币信号")
            return []

        # 计算特征 (使用加密货币数据源)
        features = self._calc_crypto_features(symbols)
        if features.empty:
            logger.error("加密货币特征计算失败")
            return []

        # 预测
        scores = self._stock_sg._predict(model, features, strategy.model_type)

        # 云端模式降级
        if scores is None and self._stock_sg.inference is not None:
            logger.info("尝试使用云端推理加载器预测加密货币...")
            scores = self._stock_sg.inference.predict(features)

        if scores is None or len(scores) == 0:
            logger.error("加密货币预测失败")
            return []

        # 排序选 Top-K
        ranked = scores.sort_values(ascending=False)
        top_symbols = ranked.head(top_k).index.tolist()
        bottom_symbols = ranked.tail(top_k).index.tolist()

        # 生成信号
        signals = []
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # 加密货币使用专属仓位上限
        weight = min(1.0 / top_k, config.CRYPTO_MAX_POSITION_PCT)

        for symbol in top_symbols:
            score = ranked[symbol]
            signal = TradeSignal(
                signal_id=f"csig_buy_{symbol.replace('/', '_')}_{timestamp}",
                strategy_id=strategy.strategy_id,
                symbol=symbol,
                direction=Direction.BUY,
                target_weight=weight,
                reason=f"加密货币模型预测排名前{top_k}, score={score:.4f}",
                confidence=min(0.5 + abs(score) * 5, 0.95),
                asset_type=AssetType.CRYPTO,
            )
            signals.append(signal)

        # 底部币种卖出信号
        for symbol in bottom_symbols:
            signal = TradeSignal(
                signal_id=f"csig_sell_{symbol.replace('/', '_')}_{timestamp}",
                strategy_id=strategy.strategy_id,
                symbol=symbol,
                direction=Direction.SELL,
                target_weight=0.0,
                reason=f"加密货币模型预测排名后{top_k}, score={scores[symbol]:.4f}",
                confidence=0.7,
                asset_type=AssetType.CRYPTO,
            )
            signals.append(signal)

        logger.info(f"🔒 加密货币信号生成完成: {len(signals)} 个 "
                    f"(买入 {len(top_symbols)}, 卖出 {len(bottom_symbols)})")
        return signals

    def generate_momentum_signals(
        self,
        symbols: list[str],
        lookback_hours: int = 24,
        top_k: int = 5,
        strategy_id: str = "crypto_momentum",
    ) -> list[TradeSignal]:
        """
        加密货币动量信号 (不依赖模型)

        按过去 N 小时收益率排序, 买入 Top-K, 卖出底部

        Args:
            symbols: 交易对列表
            lookback_hours: 回看小时数 (默认 24h)
            top_k: 选前 K 个做多
            strategy_id: 策略 ID
        """
        logger.info(f"🔒 加密货币动量信号: lookback={lookback_hours}h, top_k={top_k}")

        scores = {}
        for symbol in symbols:
            try:
                # 获取小时线数据
                df = self.collector.get_crypto_bars(
                    symbol, timeframe="1Hour", limit=lookback_hours + 5
                )
                if len(df) < lookback_hours + 2:
                    # 降级到日线
                    df = self.collector.get_crypto_bars(
                        symbol, timeframe="1Day", limit=max(lookback_hours // 24 + 5, 7)
                    )
                    if len(df) < 3:
                        continue

                recent = df.tail(lookback_hours + 1) if len(df) > lookback_hours else df
                ret = (recent.iloc[-1]["close"] / recent.iloc[0]["close"]) - 1
                scores[symbol] = ret
            except Exception as e:
                logger.debug(f"{symbol} 加密货币动量计算失败: {e}")

        if not scores:
            return []

        # 排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = ranked[:top_k]
        bottom = ranked[-top_k:]

        signals = []
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        weight = min(1.0 / top_k, config.CRYPTO_MAX_POSITION_PCT)

        for symbol, ret in top:
            signals.append(TradeSignal(
                signal_id=f"cmom_buy_{symbol.replace('/', '_')}_{timestamp}",
                strategy_id=strategy_id,
                symbol=symbol,
                direction=Direction.BUY,
                target_weight=weight,
                reason=f"加密货币动量策略: 过去{lookback_hours}h收益 {ret:.1%}",
                confidence=min(0.5 + abs(ret) * 2, 0.9),
                asset_type=AssetType.CRYPTO,
            ))

        for symbol, ret in bottom:
            signals.append(TradeSignal(
                signal_id=f"cmom_sell_{symbol.replace('/', '_')}_{timestamp}",
                strategy_id=strategy_id,
                symbol=symbol,
                direction=Direction.SELL,
                target_weight=0.0,
                reason=f"加密货币动量策略: 过去{lookback_hours}h收益 {ret:.1%} (弱势)",
                confidence=0.6,
                asset_type=AssetType.CRYPTO,
            ))

        logger.info(f"🔒 加密货币动量信号: {len(signals)} 个")
        return signals

    # ─── 内部方法 ───

    def _calc_crypto_features(self, symbols: list[str]) -> pd.DataFrame:
        """
        计算加密货币特征矩阵 (并行获取行情数据)

        特征与股票保持一致, 确保模型可通用:
        - rsi_14: 14 周期 RSI
        - ma_ratio: 5 周期均线 / 20 周期均线
        - vol_ratio: 最新成交量 / 20 周期均量
        - momentum_5: 5 周期动量
        - momentum_20: 20 周期动量
        - volatility_20: 20 周期波动率
        - price_ma60: 价格 / 60 周期均线
        """
        def _calc_one(symbol: str) -> Optional[pd.Series]:
            """计算单个币种的特征"""
            try:
                df = self.collector.get_crypto_bars(
                    symbol, timeframe="1Day", limit=120
                )
                if len(df) < 60:
                    return None

                close = df["close"]
                volume = df["volume"] if "volume" in df.columns else pd.Series([1] * len(df))
                returns = close.pct_change()

                feat = pd.Series({
                    "rsi_14": self._rsi(close, 14),
                    "ma_ratio": close.rolling(5).mean().iloc[-1] / close.rolling(20).mean().iloc[-1],
                    "vol_ratio": volume.iloc[-1] / volume.rolling(20).mean().iloc[-1],
                    "momentum_5": close.pct_change(5).iloc[-1],
                    "momentum_20": close.pct_change(20).iloc[-1],
                    "volatility_20": returns.rolling(20).std().iloc[-1],
                    "price_ma60": close.iloc[-1] / close.rolling(60).mean().iloc[-1],
                }, name=symbol)
                return feat
            except Exception as e:
                logger.debug(f"{symbol} 加密货币特征计算失败: {e}")
                return None

        # 并行计算所有币种特征
        results = parallel_map(
            _calc_one, symbols,
            max_workers=min(4, len(symbols)),
            desc="计算加密货币特征矩阵",
            timeout=120,
        )

        all_features = [r for r in results if r is not None]

        if not all_features:
            return pd.DataFrame()

        df = pd.DataFrame(all_features)
        logger.info(f"🔒 加密货币特征矩阵: {df.shape}")
        return df

    def _rsi(self, prices: pd.Series, period: int = 14) -> float:
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


# 全局实例
crypto_signal_generator = CryptoSignalGenerator()
