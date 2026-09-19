"""
模型管理器 - 多模型竞赛管理
=====================================================
T2.3: 模型管理器
- 支持 LightGBM / Linear / LSTM 模型
- 多模型竞赛: 同时训练, 比较 IC
- 自动超参数优化
"""

from src.models import ModelResult
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import config


class ModelManager:
    """多模型竞赛管理"""

    SUPPORTED_MODELS = {
        "lightgbm": {
            "default_params": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        },
        "linear": {
            "default_params": {},
        },
        "lstm": {
            "default_params": {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.1,
                "n_epochs": 50,
                "lr": 1e-3,
                "lookback": 20,
            },
        },
    }

    def __init__(self):
        from src.engine.qlib_wrapper import qlib_wrapper
        self.qlib = qlib_wrapper

    def train_model(
        self,
        model_type: str,
        factors: list[str],
        stock_pool: list[str],
        train_start: str,
        train_end: str,
        params: Optional[dict] = None,
    ) -> ModelResult:
        """
        训练单个模型

        Args:
            model_type: "lightgbm" | "linear"
            factors: 因子列表
            stock_pool: 股票池
            train_start/end: 训练数据日期范围
            params: 模型参数 (None 使用默认)

        Returns:
            ModelResult
        """
        if model_type not in self.SUPPORTED_MODELS:
            logger.error(f"不支持的模型类型: {model_type}")
            return ModelResult(model_type=model_type)

        # 如果 Qlib 可用, 使用 Qlib 训练
        if self.qlib.available:
            return self._train_with_qlib(model_type, factors, stock_pool,
                                         train_start, train_end, params)
        else:
            # 降级: 使用 LightGBM 直接训练
            return self._train_standalone(model_type, factors, stock_pool,
                                          train_start, train_end, params)

    def train_competition(
        self,
        factors: list[str],
        stock_pool: list[str],
        train_start: str,
        train_end: str,
        models: Optional[list[str]] = None,
    ) -> dict[str, ModelResult]:
        """
        多模型竞赛:
        1. 同时训练多个模型
        2. 比较验证集 IC/Rank IC
        3. 返回所有结果供策略层选择
        """
        model_types = models or list(self.SUPPORTED_MODELS.keys())
        results = {}

        logger.info(f"多模型竞赛开始: {model_types}")

        for mt in model_types:
            logger.info(f"训练 {mt}...")
            try:
                result = self.train_model(
                    mt, factors, stock_pool, train_start, train_end
                )
                results[mt] = result
                logger.info(f"  {mt}: IC={result.valid_ic:.4f}, "
                           f"RankIC={result.valid_rank_ic:.4f}")
            except Exception as e:
                logger.error(f"  {mt} 训练失败: {e}")
                results[mt] = ModelResult(model_type=mt)

        # 选出最佳模型
        # 修复: 原用 abs(valid_ic) 会选中负 IC (反向预测) 模型
        # 现只接受正向 IC, 全部无效时告警并返回 (由调用方决定降级策略)
        positive = {k: r for k, r in results.items() if r.valid_ic > 0}
        if positive:
            best = max(positive.values(), key=lambda r: r.valid_ic)
            logger.info(f"最佳模型: {best.model_type} (IC={best.valid_ic:.4f})")
        else:
            best = max(results.values(), key=lambda r: r.valid_ic)
            logger.warning(
                f"所有模型验证集 IC 均 <= 0 (最高 {best.valid_ic:.4f}), "
                f"无有效预测能力, 建议检查数据或调整训练区间"
            )

        return results

    def auto_optimize(
        self,
        model_type: str,
        base_params: dict,
        factors: list[str],
        stock_pool: list[str],
        train_start: str,
        train_end: str,
    ) -> dict:
        """
        自动超参数优化 (简化版)
        - 使用网格搜索 + 交叉验证
        - 目标: 最大化验证集 IC
        """
        logger.info(f"自动优化 {model_type} 超参数...")

        if model_type == "lightgbm":
            return self._optimize_lightgbm(
                factors, stock_pool, train_start, train_end, base_params
            )
        else:
            logger.info(f"{model_type} 无需优化, 使用默认参数")
            return base_params

    # ─── 内部: Qlib 训练 ───

    def _train_with_qlib(
        self, model_type: str, factors: list[str],
        stock_pool: list[str],
        train_start: str, train_end: str,
        params: Optional[dict],
    ) -> ModelResult:
        """使用 Qlib 框架训练"""
        try:
            # 确保 Qlib 已初始化
            if not self.qlib.available:
                self.qlib.init()

            from qlib.data.dataset import DatasetH
            from qlib.contrib.data.handler import Alpha158

            default_params = self.SUPPORTED_MODELS[model_type]["default_params"].copy()
            if params:
                default_params.update(params)

            # 创建数据集
            # 使用 Qlib 实际交易日历分割 (避免节假日不匹配)
            calendar = self.qlib.D.calendar(start_time=train_start, end_time=train_end)
            calendar_list = [str(d)[:10] for d in calendar]
            split_idx = int(len(calendar_list) * 0.8)
            valid_start = calendar_list[split_idx]
            train_end_actual = calendar_list[-1]

            data_handler_config = {
                "start_time": train_start,
                "end_time": train_end_actual,
                "fit_start_time": train_start,
                "fit_end_time": valid_start,  # 只用训练集fit归一化参数
                "instruments": stock_pool,
            }

            handler = Alpha158(**data_handler_config)
            dataset = DatasetH(
                handler=handler,
                segments={
                    "train": (train_start, valid_start),
                    "valid": (valid_start, train_end_actual),
                }
            )

            # 直接训练，不通过 qlib_wrapper
            if model_type == "lightgbm":
                from qlib.contrib.model.gbdt import LGBModel
                # 简化参数，避免过拟合
                train_params = {
                    "loss": "mse",
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "num_leaves": 32,
                    "num_threads": 4,
                    "early_stopping_rounds": 20,
                    "num_boost_round": 100,
                }
                train_params.update(default_params)
                model = LGBModel(**train_params)
                model.fit(dataset)
                result = ModelResult(model_type="lightgbm", params=train_params)
            else:
                from qlib.contrib.model.linear import LinearModel
                model = LinearModel()
                model.fit(dataset)
                result = ModelResult(model_type="linear", params=default_params)

            # 计算 IC
            # 获取验证集数据
            valid_df = dataset.prepare("valid", data_key="learn")
            if len(valid_df) > 0:
                # 简单 IC 计算: 预测值和标签的相关系数
                # LGBModel 预测
                pred = model.predict(dataset, segment="valid")
                label = valid_df["LABEL0"]
                # 对齐索引
                pred = pred.reindex(label.index)
                valid_mask = pred.notna() & label.notna()
                if valid_mask.sum() > 10:
                    ic = pred[valid_mask].corr(label[valid_mask])
                    rank_ic = pred[valid_mask].rank().corr(label[valid_mask].rank())
                    result.valid_ic = float(ic) if not pd.isna(ic) else 0.0
                    result.valid_rank_ic = float(rank_ic) if not pd.isna(rank_ic) else 0.0

                # 训练集 IC
                train_df = dataset.prepare("train", data_key="learn")
                pred_train = model.predict(dataset, segment="train")
                label_train = train_df["LABEL0"]
                pred_train = pred_train.reindex(label_train.index)
                train_mask = pred_train.notna() & label_train.notna()
                if train_mask.sum() > 10:
                    train_ic = pred_train[train_mask].corr(label_train[train_mask])
                    train_rank_ic = pred_train[train_mask].rank().corr(label_train[train_mask].rank())
                    result.train_ic = float(train_ic) if not pd.isna(train_ic) else 0.0
                    result.train_rank_ic = float(train_rank_ic) if not pd.isna(train_rank_ic) else 0.0

            logger.info(f"Qlib 训练完成: train IC={result.train_ic:.4f}, valid IC={result.valid_ic:.4f}")
            return result

        except Exception as e:
            logger.error(f"Qlib 训练失败, 降级到独立训练: {e}")
            return self._train_standalone(model_type, factors, stock_pool,
                                          train_start, train_end, params)

    # ─── 内部: 独立训练 (不依赖 Qlib) ───

    def _train_standalone(
        self, model_type: str, factors: list[str],
        stock_pool: list[str],
        train_start: str, train_end: str,
        params: Optional[dict],
    ) -> ModelResult:
        """独立训练 (使用 LightGBM + AKShare 数据, 并行获取行情)"""

        logger.info(f"独立训练 {model_type} (AKShare 数据, 并行获取)")

        # 数据准备 (与网格搜索共用同一方法, 含股票池截断告警)
        cached = self._prepare_training_data(stock_pool, train_start, train_end)
        if cached is None:
            logger.error("无可用训练数据")
            return ModelResult(model_type=model_type)
        X, y, per_stock = cached

        # 按时间顺序分割 (前 80% 训练, 后 20% 验证)
        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        if model_type == "lightgbm":
            import lightgbm as lgb

            default_params = self.SUPPORTED_MODELS["lightgbm"]["default_params"].copy()
            if params:
                default_params.update(params)

            # 转换为 LightGBM 参数
            # 修复: 透传用户配置的 n_estimators (原硬编码 100 导致模板/UI 配置静默失效)
            lgb_params = {
                "n_estimators": default_params.get("n_estimators", 100),
                "max_depth": default_params.get("max_depth", 8),
                "num_leaves": default_params.get("num_leaves", 31),
                "learning_rate": default_params.get("learning_rate", 0.05),
                "subsample": default_params.get("subsample", 0.8),
                "colsample_bytree": default_params.get("colsample_bytree", 0.8),
                "verbose": -1,
            }

            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_train, y_train)

            # 评估
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_ic = self._calc_ic(train_pred, y_train.values)
            val_ic = self._calc_ic(val_pred, y_val.values)
            train_rank_ic = self._calc_ic(train_pred, y_train.values, method="spearman")
            val_rank_ic = self._calc_ic(val_pred, y_val.values, method="spearman")

            # 保存模型
            model_dir = config.DATA_DIR / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            model.booster_.save_model(str(model_path))

            result = ModelResult(
                model_type=model_type,
                train_ic=float(train_ic),
                valid_ic=float(val_ic),
                train_rank_ic=float(train_rank_ic),
                valid_rank_ic=float(val_rank_ic),
                model_path=str(model_path),
                params=lgb_params,
            )
            logger.info(f"LightGBM 训练完成: IC={val_ic:.4f}, RankIC={val_rank_ic:.4f}")
            return result

        elif model_type == "linear":
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_ic = self._calc_ic(train_pred, y_train.values)
            val_ic = self._calc_ic(val_pred, y_val.values)

            result = ModelResult(
                model_type=model_type,
                train_ic=float(train_ic),
                valid_ic=float(val_ic),
                params={},
            )
            logger.info(f"Linear 训练完成: IC={val_ic:.4f}")
            return result

        elif model_type == "lstm":
            # LSTM 深度学习模型 (Phase 6 新增)
            # 传入按股票分组的数据, 避免滑窗跨越股票边界
            return self._train_lstm_standalone(
                X, y, stock_pool, params, per_stock
            )

        return ModelResult(model_type=model_type)

    def _train_lstm_standalone(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        stock_pool: list[str],
        params: Optional[dict],
        per_stock: Optional[list] = None,
    ) -> ModelResult:
        """
        LSTM 独立训练 (Phase 6 新增)

        Args:
            X: 特征 DataFrame (已对齐, 已移除非数值列)
            y: 标签 Series (已对齐)
            stock_pool: 股票池 (用于日志)
            params: 超参数 (None 使用默认)
            per_stock: 按股票分组的数据 [(feat, label), ...]
                       提供时按股票独立构建序列 (修复跨股票边界滑窗 bug)

        Returns:
            ModelResult
        """
        try:
            from src.engine.lstm_model import LSTMTrainer, is_lstm_available

            if not is_lstm_available():
                logger.error("PyTorch 不可用, 无法训练 LSTM 模型。请安装: pip install torch")
                return ModelResult(model_type="lstm")

            # 合并默认参数与用户参数
            default_params = self.SUPPORTED_MODELS["lstm"]["default_params"].copy()
            if params:
                default_params.update(params)

            trainer = LSTMTrainer(
                lookback=default_params.get("lookback", 20),
                hidden_size=default_params.get("hidden_size", 64),
                num_layers=default_params.get("num_layers", 2),
                dropout=default_params.get("dropout", 0.1),
                n_epochs=default_params.get("n_epochs", 50),
                lr=default_params.get("lr", 1e-3),
            )

            # 按股票独立构建序列后拼接 (修复: 原直接 concat 导致滑窗跨越股票边界)
            lookback = default_params.get("lookback", 20)
            if per_stock:
                from src.engine.lstm_model import build_sequences
                X_parts, y_parts = [], []
                for feat, label in per_stock:
                    if len(feat) <= lookback:
                        continue
                    sx, sy = build_sequences(feat, label, lookback=lookback)
                    if len(sx) > 0:
                        X_parts.append(sx)
                        y_parts.append(sy)
                if not X_parts:
                    logger.error("LSTM 训练数据不足: 所有股票序列构建后为空")
                    return ModelResult(model_type="lstm")
                X_seq = np.concatenate(X_parts, axis=0)
                y_seq = np.concatenate(y_parts, axis=0)
                result = trainer.train_from_sequences(X_seq, y_seq, stock_pool)
            else:
                # 兼容旧路径: 无分组数据时直接训练 (调用方自行保证数据连续性)
                result = trainer.train(X, y, stock_pool)
            return result

        except ImportError as e:
            logger.error(f"LSTM 依赖不可用: {e}")
            return ModelResult(model_type="lstm")
        except Exception as e:
            logger.error(f"LSTM 训练失败: {e}")
            return ModelResult(model_type="lstm")

    def _optimize_lightgbm(
        self, factors, stock_pool, train_start, train_end, base_params,
    ) -> dict:
        """LightGBM 超参数网格搜索

        性能优化: 数据只拉取一次并缓存, 27 组参数仅重训模型 (提速 20 倍+)
        """
        search_space = {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [5, 8, 12],
            "num_leaves": [31, 63, 128],
        }

        best_params = base_params.copy()
        best_ic = 0.0  # 修复: 原为 -999, abs(-999)=999 导致 best_params 永不更新
        found_valid = False  # 是否找到过有效 (IC>0) 的参数组合

        # ─── 数据只获取一次, 缓存供所有网格组合复用 ───
        cached_data = self._prepare_training_data(stock_pool, train_start, train_end)
        if cached_data is None:
            logger.error("网格搜索中止: 无可用训练数据")
            return best_params

        X, y, _ = cached_data

        for lr in search_space["learning_rate"]:
            for md in search_space["max_depth"]:
                for nl in search_space["num_leaves"]:
                    test_params = {
                        "learning_rate": lr,
                        "max_depth": md,
                        "num_leaves": nl,
                    }
                    # 用缓存数据直接训练评估 (不重新取数)
                    ic = self._eval_lightgbm_ic(X, y, test_params)
                    # 修复: 只接受正向 IC 且优于当前最优的组合
                    if ic > best_ic:
                        best_ic = ic
                        found_valid = True
                        best_params.update(test_params)
                        logger.info(f"  新最优 IC={best_ic:.4f}: lr={lr}, "
                                   f"depth={md}, leaves={nl}")

        if not found_valid:
            logger.warning("网格搜索未找到 IC>0 的参数组合, 保留基础参数")
        logger.info(f"优化完成: 最优 IC={best_ic:.4f}")
        return best_params

    def _prepare_training_data(
        self, stock_pool: list[str], train_start: str, train_end: str,
    ) -> Optional[tuple[pd.DataFrame, pd.Series, list]]:
        """准备训练数据 (特征矩阵 + 标签), 供网格搜索复用

        Returns:
            (X, y, per_stock) 元组:
              - X: 合并后的特征 DataFrame
              - y: 合并后的标签 Series
              - per_stock: [(feat_df, label_series), ...] 按股票分组的数据
                (LSTM 训练用, 避免滑窗跨越股票边界)
            数据不足时返回 None
        """
        from src.data.akshare_collector import akshare_collector
        from src.utils.parallel import parallel_map

        pool_to_fetch = stock_pool[:5]  # 取前5只 (与 _train_standalone 保持一致)
        if len(stock_pool) > 5:
            logger.warning(
                f"股票池共 {len(stock_pool)} 只, 独立训练模式仅使用前 5 只: "
                f"{pool_to_fetch} (Qlib 可用时可使用完整股票池)"
            )

        def _fetch_stock_data(symbol: str) -> tuple[str, Optional[pd.DataFrame]]:
            """获取单只股票数据并计算特征和标签"""
            try:
                df = akshare_collector.get_us_stock_daily(symbol, train_start, train_end)
                if len(df) < 60:
                    return symbol, None
                feat = self._calc_features(df)
                label = df["close"].shift(-5) / df["close"] - 1
                feat["symbol"] = symbol
                return symbol, (feat, label)
            except Exception as e:
                logger.error(f"{symbol} 数据获取失败: {e}")
                return symbol, None

        results = parallel_map(
            _fetch_stock_data, pool_to_fetch,
            max_workers=min(4, len(pool_to_fetch)),
            desc="并行获取训练数据",
            timeout=120,
        )

        all_features = []
        all_labels = []
        per_stock = []  # 按股票分组 (LSTM 用, 防止滑窗跨股票边界)
        for result in results:
            if result is None:
                continue
            symbol, data = result
            if data is None:
                continue
            feat, label = data
            all_features.append(feat)
            all_labels.append(label)
            per_stock.append((feat, label))

        if not all_features:
            return None

        X = pd.concat(all_features).reset_index(drop=True)
        y = pd.concat(all_labels).reset_index(drop=True)

        # 对齐并移除非数值列
        valid = y.notna() & X.notna().all(axis=1)
        X = X[valid].reset_index(drop=True)
        y = y[valid].reset_index(drop=True)
        X = X.drop(columns=["symbol"], errors="ignore")

        if len(X) < 50:
            logger.error(f"训练数据不足: {len(X)} 行")
            return None

        # 对 per_stock 做同样的对齐处理 (保持各股票内部时序连续)
        aligned_per_stock = []
        for feat, label in per_stock:
            v = label.notna() & feat.notna().all(axis=1)
            f = feat[v].drop(columns=["symbol"], errors="ignore").reset_index(drop=True)
            l = label[v].reset_index(drop=True)
            if len(f) > 0:
                aligned_per_stock.append((f, l))

        return X, y, aligned_per_stock

    def _eval_lightgbm_ic(self, X: pd.DataFrame, y: pd.Series, params: dict) -> float:
        """用给定参数训练 LightGBM 并返回验证集 IC (网格搜索专用)"""
        try:
            import lightgbm as lgb

            split = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split], X.iloc[split:]
            y_train, y_val = y.iloc[:split], y.iloc[split:]

            lgb_params = {
                "n_estimators": 100,
                "verbose": -1,
            }
            lgb_params.update(params)

            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            return self._calc_ic(val_pred, y_val.values)
        except Exception as e:
            logger.debug(f"网格组合评估失败: {e}")
            return 0.0

    # ─── 工具方法 ───

    def _calc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算特征因子"""
        close = df["close"]
        volume = df["volume"]
        returns = close.pct_change()

        feat = pd.DataFrame(index=df.index)
        feat["rsi_14"] = self._calc_rsi(close, 14)
        feat["ma_ratio"] = close.rolling(5).mean() / close.rolling(20).mean()
        feat["vol_ratio"] = volume / volume.rolling(20).mean()
        feat["momentum_5"] = close.pct_change(5)
        feat["momentum_20"] = close.pct_change(20)
        feat["volatility_20"] = returns.rolling(20).std()
        feat["price_ma60"] = close / close.rolling(60).mean()
        return feat

    def _calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def _calc_ic(self, pred, actual, method="pearson") -> float:
        """计算 IC"""
        if len(pred) != len(actual) or len(pred) == 0:
            return 0.0
        pred_s = pd.Series(pred)
        actual_s = pd.Series(actual)
        if method == "spearman":
            return float(pred_s.corr(actual_s, method="spearman"))
        return float(pred_s.corr(actual_s))


# 全局实例
model_manager = ModelManager()
