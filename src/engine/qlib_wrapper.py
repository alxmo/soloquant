"""
Qlib 量化引擎封装
=====================================================
T2.1: Qlib 封装层
- 初始化 Qlib 环境
- 因子搜索 (Alpha158)
- 模型训练 (LightGBM / Linear)
- 回测运行
- 信号生成
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.models import BacktestResult, ModelResult, StrategyConfig, StrategyGrade
from src.utils.config import config


class QlibWrapper:
    """Qlib 量化引擎封装"""

    _initialized = False
    _provider_uri = ""

    def __init__(self):
        self._qlib = None
        self._data_ops = None

    def init(self, provider_uri: Optional[str] = None) -> bool:
        """
        初始化 Qlib 环境

        Args:
            provider_uri: Qlib 数据目录路径, None 则使用 config 配置
        """
        if QlibWrapper._initialized and provider_uri == QlibWrapper._provider_uri:
            return True

        provider_uri = provider_uri or str(config.QLIB_DATA_PATH)
        provider_uri = os.path.expanduser(provider_uri)

        if not Path(provider_uri).exists():
            logger.warning(f"Qlib 数据目录不存在: {provider_uri}")
            logger.info("请先运行数据初始化: python -m src.data.qlib_adapter")
            return False

        try:
            # 解决 MLflow 文件存储警告 + 禁用不必要的日志
            os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
            os.environ['QLIB_DISABLE_MLFLOW'] = '1'  # 禁用 MLflow 实验跟踪

            import qlib
            from qlib.config import REG_US, C

            # Windows 兼容配置: 禁用多进程，避免 spawn 模式报错
            # 必须在 qlib.init() 之前设置
            import platform
            if platform.system() == "Windows":
                C["joblib_backend"] = "threading"
                C["maxtasksperchild"] = None

            qlib.init(provider_uri=provider_uri, region=REG_US)

            # init 之后再设置一次，防止被重置
            if platform.system() == "Windows":
                C["joblib_backend"] = "threading"
            QlibWrapper._initialized = True
            QlibWrapper._provider_uri = provider_uri
            self._qlib = qlib

            from qlib.data import D
            self._data_ops = D

            logger.info(f"Qlib 初始化成功 (provider_uri={provider_uri}, region=US)")
            return True
        except Exception as e:
            logger.error(f"Qlib 初始化失败: {e}")
            return False

    @property
    def D(self):
        """Qlib 数据操作器"""
        if self._data_ops is None:
            raise RuntimeError("Qlib 未初始化, 请先调用 init()")
        return self._data_ops

    @property
    def available(self) -> bool:
        return QlibWrapper._initialized

    # ─── 数据操作 ───

    def get_features(
        self,
        symbols: list[str],
        fields: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        获取特征数据

        Args:
            symbols: 股票代码列表
            fields: 字段列表 (如 ["$close", "$volume"])
            start: 开始日期 "YYYY-MM-DD"
            end: 结束日期 "YYYY-MM-DD"
        """
        if not self.available:
            return pd.DataFrame()
        try:
            df = self.D.features(symbols, fields, start_time=start, end_time=end)
            return df
        except Exception as e:
            logger.error(f"Qlib 获取特征失败: {e}")
            return pd.DataFrame()

    def get_calendar(self, start: str, end: str) -> list[str]:
        """获取交易日历"""
        if not self.available:
            return []
        try:
            return list(self.D.calendar(start_time=start, end_time=end))
        except Exception as e:
            logger.error(f"Qlib 获取日历失败: {e}")
            return []

    def get_instruments(self) -> list[str]:
        """获取所有可交易股票"""
        if not self.available:
            return []
        try:
            instruments = self.D.instruments()
            return list(self.D.list_instruments(instruments=instruments,
                                                as_list=True))
        except Exception as e:
            logger.error(f"Qlib 获取股票列表失败: {e}")
            return []

    # ─── 因子计算 ───

    def calc_alpha158_factors(
        self,
        symbols: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        计算 Alpha158 因子集

        Returns:
            DataFrame: MultiIndex(instrument, datetime), columns=因子名
        """
        if not self.available:
            return pd.DataFrame()

        try:
            from qlib.contrib.data.handler import Alpha158

            handler = Alpha158(
                instruments=symbols,
                start_time=start,
                end_time=end,
                fit_start_time=start,
                fit_end_time=end,
            )
            df = handler.fetch()
            logger.info(f"Alpha158 因子计算完成: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Alpha158 因子计算失败: {e}")
            return pd.DataFrame()

    # ─── 模型训练 ───

    def train_lightgbm(
        self,
        dataset,  # Qlib DatasetH
        params: Optional[dict] = None,
    ) -> ModelResult:
        """训练 LightGBM 模型"""
        try:
            from qlib.contrib.model.gbdt import LGBModel

            default_params = {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            }
            if params:
                default_params.update(params)

            model = LGBModel(**default_params)
            model.fit(dataset)

            result = ModelResult(
                model_type="lightgbm",
                params=default_params,
            )
            logger.info("LightGBM 模型训练完成")
            return result
        except Exception as e:
            logger.error(f"LightGBM 训练失败: {e}")
            return ModelResult(model_type="lightgbm")

    def train_linear(
        self,
        dataset,
        params: Optional[dict] = None,
    ) -> ModelResult:
        """训练线性模型"""
        try:
            from qlib.contrib.model.linear import LinearModel

            model = LinearModel(**(params or {}))
            model.fit(dataset)

            result = ModelResult(
                model_type="linear",
                params=params or {},
            )
            logger.info("Linear 模型训练完成")
            return result
        except Exception as e:
            logger.error(f"Linear 训练失败: {e}")
            return ModelResult(model_type="linear")

    # ─── 回测 ───

    def run_backtest(
        self,
        model,
        dataset,
        strategy_config: StrategyConfig,
        start: str,
        end: str,
        initial_capital: float = 100000,
    ) -> BacktestResult:
        """运行 Qlib 回测"""
        if not self.available:
            return BacktestResult(strategy_id=strategy_config.strategy_id)

        try:
            from qlib.contrib.evaluate import backtest_daily, risk_analysis
            from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

            # 生成预测
            pred = model.predict(dataset, segment="test")

            # 创建策略实例 (Qlib 0.9.7 API)
            topk = min(5, len(strategy_config.stock_pool))
            strategy = TopkDropoutStrategy(
                signal=pred,
                topk=topk,
                n_drop=2,
            )

            # 交易所参数
            exchange_kwargs = {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            }

            # 运行回测 (Qlib 0.9.7 新 API)
            portfolio_metric = backtest_daily(
                start_time=start,
                end_time=end,
                strategy=strategy,
                account=initial_capital,
                benchmark="AAPL",
                exchange_kwargs=exchange_kwargs,
            )

            # 分析结果
            if isinstance(portfolio_metric, tuple):
                report_normal, positions_normal = portfolio_metric
            else:
                report_normal = portfolio_metric
            analysis = risk_analysis(report_normal)

            # 提取指标 (Qlib 0.9.7 risk_analysis 返回格式不同)
            metrics = {}
            if isinstance(analysis, pd.DataFrame):
                for idx, row in analysis.iterrows():
                    # 处理多层索引和单层索引
                    if isinstance(idx, tuple):
                        metric_name = idx[0] if len(idx) > 0 else str(idx)
                    else:
                        metric_name = str(idx)
                    # 获取第一列的值
                    val = row.iloc[0] if len(row) > 0 else 0.0
                    try:
                        metrics[metric_name] = float(val)
                    except (TypeError, ValueError):
                        metrics[metric_name] = 0.0
            else:
                # 直接是字典的情况
                metrics = analysis

            total_return = metrics.get("return", 0.0)
            annual_return = metrics.get("annualized_return", 0.0)
            sharpe = metrics.get("information_ratio", 0.0)
            max_dd = abs(metrics.get("max_drawdown", 0.0))

            # 评级
            grade = self._grade_strategy(annual_return, sharpe, max_dd)

            result = BacktestResult(
                strategy_id=strategy_config.strategy_id,
                period=(start, end),
                initial_capital=initial_capital,
                final_value=initial_capital * (1 + total_return),
                total_return=total_return,
                annual_return=annual_return,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                grade=grade,
                plain_summary=self._plain_summary(annual_return, sharpe, max_dd, grade),
            )

            logger.info(f"回测完成: {grade}级, 年化 {annual_return:.1%}, "
                        f"夏普 {sharpe:.2f}, 回撤 {max_dd:.1%}")
            return result

        except Exception as e:
            logger.error(f"回测失败: {e}")
            import traceback
            traceback.print_exc()
            return BacktestResult(strategy_id=strategy_config.strategy_id)

    # ─── 内部方法 ───

    def _grade_strategy(
        self, annual_return: float, sharpe: float, max_dd: float,
    ) -> StrategyGrade:
        """策略评级"""
        if annual_return > 0.20 and sharpe > 1.5 and max_dd < 0.10:
            return StrategyGrade.A
        elif annual_return > 0.15 and sharpe > 1.0 and max_dd < 0.15:
            return StrategyGrade.B
        elif annual_return > 0.10 and sharpe > 0.5 and max_dd < 0.20:
            return StrategyGrade.C
        else:
            return StrategyGrade.D

    def _plain_summary(
        self, annual_return: float, sharpe: float,
        max_dd: float, grade: StrategyGrade,
    ) -> str:
        """生成小白解读"""
        return (
            f"策略评级: {grade}级\n"
            f"年化收益 {annual_return:.1%} - "
            f"{'优秀' if annual_return > 0.2 else '良好' if annual_return > 0.15 else '一般' if annual_return > 0.1 else '不理想'}\n"
            f"最大回撤 {max_dd:.1%} - "
            f"{'很小' if max_dd < 0.05 else '可控' if max_dd < 0.15 else '较大'}\n"
            f"夏普比率 {sharpe:.2f} - "
            f"{'优秀' if sharpe > 1.5 else '不错' if sharpe > 1.0 else '一般' if sharpe > 0.5 else '较差'}\n"
            f"⚠️ 过去的收益不代表未来"
        )


# 全局实例
qlib_wrapper = QlibWrapper()
