"""
信号生成器 - 模型预测转换为交易信号
=====================================================
T2.5: 信号生成器
- 加载训练好的模型
- 对股票池生成预测
- 转换为交易信号 (买入/卖出/调仓)
"""

import pickle
from collections import OrderedDict
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from src.models import Direction, StrategyConfig, TradeSignal
from src.utils.config import config


class SignalGenerator:
    """信号生成器"""

    # 模型缓存最大数量 (防止长期运行内存泄漏)
    _MODEL_CACHE_MAX = 10

    def __init__(self):
        # 云端模式: 使用 model_inference 推理加载器 (不依赖 Qlib/torch)
        # 本地模式: 使用 qlib_wrapper (完整功能)
        if config.DEPLOY_MODE == "cloud":
            try:
                from src.engine.model_inference import model_inference
                self.inference = model_inference
                logger.info("信号生成器: 云端模式 (使用 model_inference 推理加载器)")
            except Exception as e:
                logger.warning(f"model_inference 加载失败，降级到标准模式: {e}")
                self.inference = None
        else:
            self.inference = None

        from src.engine.qlib_wrapper import qlib_wrapper
        self.qlib = qlib_wrapper
        # LRU 模型缓存 (有上限，淘汰最久未使用)
        self._model_cache: OrderedDict[str, object] = OrderedDict()

    def generate_signals(
        self,
        strategy: StrategyConfig,
        model_path: str,
        stock_pool: list[str],
        top_k: int = None,
    ) -> list[TradeSignal]:
        """
        生成交易信号

        流程:
        1. 加载模型
        2. 对股票池计算特征
        3. 模型预测打分
        4. 选 Top-K 做多
        5. 生成买入信号

        Args:
            strategy: 策略配置
            model_path: 模型文件路径
            stock_pool: 股票池
            top_k: 选前 K 只做多
        """
        # 使用配置值（如果未指定）
        if top_k is None:
            top_k = config.TOP_K_SIGNALS

        logger.info(f"信号生成: {strategy.name}, 模型={strategy.model_type}, top_k={top_k}")

        # 加载模型
        model = self._load_model(model_path, strategy.model_type)
        if model is None:
            logger.error("模型加载失败, 无法生成信号")
            return []

        # 计算特征
        features = self._calc_features(stock_pool)
        if features.empty:
            logger.error("特征计算失败")
            return []

        # 预测
        scores = self._predict(model, features, strategy.model_type)

        # 云端模式降级: 如果标准 _predict 失败，尝试使用 model_inference
        if scores is None and self.inference is not None:
            logger.info("尝试使用云端推理加载器预测...")
            scores = self.inference.predict(features)

        if scores is None or len(scores) == 0:
            logger.error("预测失败")
            return []

        # 排序选 Top-K
        ranked = scores.sort_values(ascending=False)
        top_symbols = ranked.head(top_k).index.tolist()
        bottom_symbols = ranked.tail(top_k).index.tolist()

        # 生成信号
        signals = []
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        weight = min(1.0 / top_k, strategy.max_position)

        for symbol in top_symbols:
            score = ranked[symbol]
            signal = TradeSignal(
                signal_id=f"sig_buy_{symbol}_{timestamp}",
                strategy_id=strategy.strategy_id,
                symbol=symbol,
                direction=Direction.BUY,
                target_weight=weight,
                reason=f"模型预测打分排名前{top_k}, score={score:.4f}",
                confidence=min(0.5 + abs(score) * 5, 0.95),
            )
            signals.append(signal)

        # 底部股票如果有持仓则卖出
        for symbol in bottom_symbols:
            signal = TradeSignal(
                signal_id=f"sig_sell_{symbol}_{timestamp}",
                strategy_id=strategy.strategy_id,
                symbol=symbol,
                direction=Direction.SELL,
                target_weight=0.0,
                reason=f"模型预测打分排名后{top_k}, score={scores[symbol]:.4f}",
                confidence=0.7,
            )
            signals.append(signal)

        logger.info(f"生成 {len(signals)} 个信号: "
                    f"买入 {len(top_symbols)}, 卖出 {len(bottom_symbols)}")
        return signals

    def generate_momentum_signals(
        self,
        stock_pool: list[str],
        lookback_days: int = 20,
        top_k: int = 5,
        strategy_id: str = "momentum",
    ) -> list[TradeSignal]:
        """
        动量信号 (简化版, 不依赖模型)
        - 按过去 N 天收益率排序
        - 买入 Top-K
        """
        from src.data.akshare_collector import akshare_collector

        logger.info(f"动量信号生成: lookback={lookback_days}天, top_k={top_k}")

        scores = {}
        for symbol in stock_pool:
            try:
                df = akshare_collector.get_us_stock_daily(symbol)
                if len(df) < lookback_days + 5:
                    continue
                recent = df.tail(lookback_days + 1)
                ret = (recent.iloc[-1]["close"] / recent.iloc[0]["close"]) - 1
                scores[symbol] = ret
            except Exception as e:
                logger.debug(f"{symbol} 动量计算失败: {e}")

        if not scores:
            return []

        # 排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = ranked[:top_k]
        bottom = ranked[-top_k:]

        signals = []
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        weight = min(1.0 / top_k, 0.30)

        for symbol, ret in top:
            signals.append(TradeSignal(
                signal_id=f"mom_buy_{symbol}_{timestamp}",
                strategy_id=strategy_id,
                symbol=symbol,
                direction=Direction.BUY,
                target_weight=weight,
                reason=f"动量策略: 过去{lookback_days}天收益 {ret:.1%}",
                confidence=min(0.5 + abs(ret) * 2, 0.9),
            ))

        for symbol, ret in bottom:
            signals.append(TradeSignal(
                signal_id=f"mom_sell_{symbol}_{timestamp}",
                strategy_id=strategy_id,
                symbol=symbol,
                direction=Direction.SELL,
                target_weight=0.0,
                reason=f"动量策略: 过去{lookback_days}天收益 {ret:.1%} (弱势)",
                confidence=0.6,
            ))

        logger.info(f"动量信号: {len(signals)} 个")
        return signals

    # ─── 内部方法 ───

    def _load_model(self, model_path: str, model_type: str):
        """加载模型 (带 LRU 缓存淘汰)"""
        if model_path in self._model_cache:
            # 命中缓存: 移到末尾 (最近使用)
            self._model_cache.move_to_end(model_path)
            return self._model_cache[model_path]

        try:
            if model_type == "lightgbm":
                import lightgbm as lgb
                model = lgb.Booster(model_file=model_path)
            elif model_type == "linear":
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            elif model_type == "lstm":
                # LSTM 模型通过 LSTMTrainer.predict() 预测，不需要单独加载
                # 缓存 model_path 作为标记，实际预测在 _predict 中处理
                model = model_path  # 保存路径，_predict 中调用 LSTMTrainer
            else:
                logger.error(f"不支持的模型类型: {model_type}")
                return None

            # 写入 LRU 缓存 (超限时淘汰最久未使用)
            self._model_cache[model_path] = model
            self._model_cache.move_to_end(model_path)
            while len(self._model_cache) > self._MODEL_CACHE_MAX:
                evicted_key, _ = self._model_cache.popitem(last=False)
                logger.debug(f"模型缓存淘汰: {evicted_key}")

            logger.info(f"模型已加载: {model_path} (类型: {model_type}, 缓存: {len(self._model_cache)}/{self._MODEL_CACHE_MAX})")
            return model
        except Exception as e:
            logger.error(f"模型加载失败 [{model_path}]: {e}")
            return None

    def _calc_features(self, stock_pool: list[str]) -> pd.DataFrame:
        """计算特征矩阵 (并行获取行情数据)"""
        from src.data.akshare_collector import akshare_collector
        from src.utils.parallel import parallel_map

        def _calc_one(symbol: str) -> Optional[pd.Series]:
            """计算单只股票的特征"""
            try:
                df = akshare_collector.get_us_stock_daily(symbol)
                if len(df) < 60:
                    return None

                close = df["close"]
                volume = df["volume"]
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
                logger.debug(f"{symbol} 特征计算失败: {e}")
                return None

        # 并行计算所有股票特征
        results = parallel_map(
            _calc_one, stock_pool,
            max_workers=min(4, len(stock_pool)),
            desc="计算特征矩阵",
            timeout=60,
        )

        all_features = [r for r in results if r is not None]

        if not all_features:
            return pd.DataFrame()

        df = pd.DataFrame(all_features)
        logger.info(f"特征矩阵: {df.shape}")
        return df

    def _predict(self, model, features: pd.DataFrame, model_type: str) -> Optional[pd.Series]:
        """
        模型预测

        Args:
            model: 加载的模型对象。
                   - lightgbm: lgb.Booster
                   - linear: sklearn LinearRegression (pickle)
                   - lstm: model_path 字符串 (实际预测通过 LSTMTrainer)
            features: 特征 DataFrame
            model_type: 模型类型
        """
        try:
            if model_type == "lightgbm":
                pred = model.predict(features.values)
                scores = pd.Series(pred, index=features.index)
                return scores

            elif model_type == "linear":
                pred = model.predict(features.values)
                scores = pd.Series(pred, index=features.index)
                return scores

            elif model_type == "lstm":
                # LSTM 通过 LSTMTrainer.predict() 预测 (Phase 6 新增)
                # 修复: 原实现传入每只股票仅 1 行的特征, 而 LSTM 需要 lookback+1 行时序,
                # 导致预测必然返回 None。现改为逐股票取时序特征分别预测。
                from src.engine.lstm_model import LSTMTrainer, is_lstm_available

                if not is_lstm_available():
                    logger.error("PyTorch 不可用, 无法使用 LSTM 模型预测")
                    return None

                model_path = model  # model 是 model_path 字符串
                trainer = LSTMTrainer()

                scores_dict: dict[str, float] = {}
                for symbol in features.index:
                    feat_series = features.loc[symbol]
                    # 单行特征 -> 需要重新获取时序数据构建序列
                    pred = self._predict_lstm_single(trainer, model_path, symbol)
                    if pred is not None:
                        scores_dict[symbol] = float(pred)

                if not scores_dict:
                    logger.error("LSTM 预测: 所有股票均失败")
                    return None

                # 补齐未预测成功的股票为 0 分 (中性)
                for symbol in features.index:
                    scores_dict.setdefault(symbol, 0.0)

                scores = pd.Series(scores_dict)
                logger.info(f"LSTM 预测完成: {len(scores_dict)} 只股票")
                return scores

            else:
                logger.error(f"不支持的模型类型: {model_type}")
                return None

        except Exception as e:
            logger.error(f"预测失败: {e}")
            return None

    def _predict_lstm_single(self, trainer, model_path: str, symbol: str) -> Optional[float]:
        """对单只股票做 LSTM 预测 (取最新一个序列的预测值)

        修复: LSTM 需要 lookback+1 行时序特征, 原实现只传 1 行导致必然失败。
        """
        try:
            from src.data.akshare_collector import akshare_collector
            from src.engine.lstm_model import build_lstm_features

            # 获取足够的时序数据 (lookback=20, 取 120 行留足余量)
            df = akshare_collector.get_us_stock_daily(symbol)
            if df is None or len(df) < 80:
                logger.debug(f"{symbol} LSTM 预测数据不足: {len(df) if df is not None else 0} 行")
                return None

            # 构建与训练一致的特征
            feat = build_lstm_features(df)
            feat = feat.dropna()
            if len(feat) < 25:
                logger.debug(f"{symbol} LSTM 特征不足: {len(feat)} 行")
                return None

            # 取最近 60 行做预测, 只取最后一个预测值作为该股分数
            recent = feat.tail(60)
            preds = trainer.predict(model_path, recent)
            if preds is None or len(preds) == 0:
                return None
            return float(preds[-1])
        except Exception as e:
            logger.debug(f"{symbol} LSTM 预测失败: {e}")
            return None

    def _predict_with_model_file(self, model_result, symbols: list[str]) -> dict[str, float]:
        """用已训练的模型对股票列表打分 (回测加权用)

        Args:
            model_result: ModelResult (含 model_path 和 model_type)
            symbols: 股票列表

        Returns:
            {symbol: score} 字典, 失败返回空字典
        """
        try:
            model_type = model_result.model_type
            model_path = model_result.model_path

            if model_type == "lstm":
                # LSTM: 逐股票预测
                from src.engine.lstm_model import LSTMTrainer, is_lstm_available
                if not is_lstm_available():
                    return {}
                trainer = LSTMTrainer()
                scores = {}
                for symbol in symbols:
                    pred = self._predict_lstm_single(trainer, model_path, symbol)
                    if pred is not None:
                        scores[symbol] = pred
                return scores

            # lightgbm / linear: 加载模型 + 计算特征 + 批量预测
            model = self._load_model(model_path, model_type)
            if model is None:
                return {}
            features = self._calc_features(symbols)
            if features.empty:
                return {}
            pred = self._predict(model, features, model_type)
            if pred is None:
                return {}
            return {s: float(v) for s, v in pred.items() if pd.notna(v)}
        except Exception as e:
            logger.warning(f"模型打分失败: {e}")
            return {}

    def _rsi(self, prices: pd.Series, period: int = 14) -> float:
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
signal_generator = SignalGenerator()
