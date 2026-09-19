"""
因子自动搜索与筛选
=====================================================
T2.2: 因子自动搜索
- 使用 Qlib Alpha158 因子库
- 计算因子 IC (信息系数) / IR (信息比率)
- 按 |IR| 排序, 去相关, 筛选 Top-N 有效因子
"""


import numpy as np
import pandas as pd
from loguru import logger

from src.models import FactorResult


class FactorSearcher:
    """自动因子搜索与筛选"""

    # Alpha158 主要因子分类 (按类型分组)
    FACTOR_GROUPS = {
        "price": [
            "KMID", "KLEN", "KMID2", "KUPP", "KLOW", "KUPP2", "KLOW2",
            "OPEN0", "HIGH0", "LOW0", "VWAP0",
        ],
        "momentum": [
            "ROC", "VROC", "MA", "STD", "BETA", "RSQR", "RESI",
            "ROC5", "ROC10", "ROC20", "ROC30", "ROC60",
            "MA5", "MA10", "MA20", "MA30", "MA60",
        ],
        "volume": [
            "VSTD", "WVMA", "VSUMP", "VSUMN", "VDUMP", "VDUMN",
            "VSTD5", "VSTD10", "VSTD20", "VSTD30", "VSTD60",
        ],
        "technical": [
            "RSI", "KDJ_K", "KDJ_D", "KDJ_J", "MACD", "CCI", "DMI",
        ],
    }

    def __init__(self):
        from src.engine.qlib_wrapper import qlib_wrapper
        self.qlib = qlib_wrapper

    def search(
        self,
        stock_pool: list[str],
        start: str,
        end: str,
        top_n: int = 20,
        ic_threshold: float = 0.03,
        correlation_threshold: float = 0.7,
    ) -> list[FactorResult]:
        """
        自动因子搜索流程:
        1. 计算所有 Alpha158 因子
        2. 计算每个因子的 IC / Rank IC / IR
        3. 按 |IR| 排序
        4. 去相关 (因子间相关性 < 阈值)
        5. 返回 Top-N 因子

        Args:
            stock_pool: 股票池
            start/end: 回测日期范围
            top_n: 返回前 N 个因子
            ic_threshold: IC 绝对值阈值 (低于此值认为无效)
            correlation_threshold: 因子间相关性阈值 (高于此值去重)

        Returns:
            list[FactorResult]: 按有效性排序的因子列表
        """
        logger.info(f"因子搜索开始: {len(stock_pool)} 只股票, {start} ~ {end}")

        # 1. 计算因子
        factor_df = self._calc_factors(stock_pool, start, end)
        if factor_df.empty:
            logger.warning("因子数据为空, 无法搜索")
            return []

        # 2. 计算前向收益 (标签)
        returns = self._calc_forward_returns(stock_pool, start, end)
        if returns.empty:
            logger.warning("收益数据为空, 无法计算 IC")
            return []

        # 3. 计算 IC / IR
        factor_results = self._calc_ic_ir(factor_df, returns, ic_threshold)

        # 4. 按 |IR| 排序
        factor_results.sort(key=lambda x: abs(x.ir), reverse=True)

        # 5. 去相关
        selected = self._decorrelate(factor_results, factor_df, correlation_threshold)

        # 6. 取 Top-N
        top_factors = selected[:top_n]
        effective = [f for f in top_factors if f.is_effective]

        logger.info(f"因子搜索完成: {len(factor_results)} 个因子 -> "
                     f"筛选出 {len(effective)} 个有效因子")

        # 打印结果
        for f in top_factors[:10]:
            status = "✅" if f.is_effective else "❌"
            logger.info(f"  {status} {f.name}: IC={f.ic:.4f}, "
                        f"RankIC={f.rank_ic:.4f}, IR={f.ir:.4f}")

        return top_factors

    def search_simple(
        self,
        stock_pool: list[str],
        start: str,
        end: str,
    ) -> list[FactorResult]:
        """
        简化版因子搜索 (不依赖 Qlib, 使用自行计算的因子)

        适用于 Qlib 数据未初始化时的降级方案
        """
        logger.info("使用简化因子搜索 (不依赖 Qlib)")
        from src.data.akshare_collector import akshare_collector

        all_factors = []
        for symbol in stock_pool[:3]:  # 取前3只做快速验证
            try:
                df = akshare_collector.get_us_stock_daily(symbol, start=start, end=end)
                if len(df) < 60:
                    continue

                factors = self._calc_simple_factors(df)
                returns = df["close"].pct_change(5).shift(-5)  # 5日前向收益
                all_factors.append((symbol, factors, returns))
            except Exception as e:
                logger.error(f"{symbol} 因子计算失败: {e}")

        if not all_factors:
            return []

        # 合并计算 IC
        results = []
        factor_names = all_factors[0][1].columns.tolist()
        for fname in factor_names:
            ic_list = []
            rank_ic_list = []
            for symbol, factors, returns in all_factors:
                aligned = pd.concat([factors[fname], returns], axis=1, keys=["factor", "return"]).dropna()
                if len(aligned) < 20:
                    continue
                ic = aligned["factor"].corr(aligned["return"])
                rank_ic = aligned["factor"].corr(aligned["return"], method="spearman")
                if not np.isnan(ic):
                    ic_list.append(ic)
                    rank_ic_list.append(rank_ic)

            if ic_list:
                ic_mean = np.mean(ic_list)
                rank_ic_mean = np.mean(rank_ic_list)
                ir = ic_mean / (np.std(ic_list) + 1e-8)

                results.append(FactorResult(
                    name=fname,
                    ic=ic_mean,
                    rank_ic=rank_ic_mean,
                    ir=ir,
                    is_effective=abs(ir) > 0.3,
                ))

        results.sort(key=lambda x: abs(x.ir), reverse=True)
        logger.info(f"简化因子搜索完成: {len(results)} 个因子")
        return results[:20]

    # ─── 内部方法 ───

    def _calc_factors(self, stock_pool: list[str], start: str, end: str) -> pd.DataFrame:
        """使用 Qlib 计算 Alpha158 因子"""
        if not self.qlib.available:
            return pd.DataFrame()
        try:
            return self.qlib.calc_alpha158_factors(stock_pool, start, end)
        except Exception as e:
            logger.error(f"Qlib 因子计算失败: {e}")
            return pd.DataFrame()

    def _calc_forward_returns(
        self, stock_pool: list[str], start: str, end: str,
    ) -> pd.DataFrame:
        """计算前向收益"""
        if not self.qlib.available:
            return pd.DataFrame()

        try:
            df = self.qlib.get_features(
                stock_pool, ["$close"], start, end
            )
            if df.empty:
                return pd.DataFrame()
            # 5日前向收益
            df["return"] = df["$close"].groupby(level=0).pct_change(5).shift(-5)
            return df["return"]
        except Exception as e:
            logger.error(f"前向收益计算失败: {e}")
            return pd.Series(dtype=float)

    def _calc_ic_ir(
        self, factor_df: pd.DataFrame,
        returns: pd.Series, ic_threshold: float,
    ) -> list[FactorResult]:
        """计算每个因子的 IC / IR"""
        results = []

        # 确保索引对齐
        factor_df = factor_df.copy()
        # 去掉非因子列 (如 $close, $volume, label 等)
        factor_cols = [c for c in factor_df.columns
                       if not c.startswith("$") and c not in ("label",)]

        for col in factor_cols:
            try:
                aligned = pd.concat([factor_df[col], returns], axis=1,
                                    keys=["factor", "return"]).dropna()
                if len(aligned) < 20:
                    continue

                # 按日期截面计算 IC
                ic_series = aligned.groupby(level=1)["factor"].corr(aligned["return"])
                rank_ic_series = aligned.groupby(level=1)["factor"].corr(
                    aligned["return"], method="spearman"
                )

                ic_mean = ic_series.mean()
                rank_ic_mean = rank_ic_series.mean()
                ic_std = ic_series.std()
                ir = ic_mean / (ic_std + 1e-8)

                results.append(FactorResult(
                    name=col,
                    ic=float(ic_mean) if not np.isnan(ic_mean) else 0.0,
                    rank_ic=float(rank_ic_mean) if not np.isnan(rank_ic_mean) else 0.0,
                    ir=float(ir) if not np.isnan(ir) else 0.0,
                    is_effective=abs(ir) > 0.3,
                ))
            except Exception as e:
                logger.debug(f"因子 {col} IC 计算失败: {e}")
                continue

        return results

    def _decorrelate(
        self,
        factor_results: list[FactorResult],
        factor_df: pd.DataFrame,
        correlation_threshold: float,
    ) -> list[FactorResult]:
        """去相关: 相似因子只保留 IR 最高的"""
        selected = []
        selected_names = []

        for fr in factor_results:
            if fr.name not in factor_df.columns:
                selected.append(fr)
                selected_names.append(fr.name)
                continue

            # 检查与已选因子的相关性
            is_correlated = False
            for sn in selected_names:
                if sn not in factor_df.columns:
                    continue
                try:
                    corr = factor_df[fr.name].corr(factor_df[sn])
                    if abs(corr) > correlation_threshold:
                        is_correlated = True
                        logger.debug(f"去相关: {fr.name} 与 {sn} 相关性={corr:.3f}, 跳过")
                        break
                except Exception:
                    continue

            if not is_correlated:
                selected.append(fr)
                selected_names.append(fr.name)

        return selected

    def _calc_simple_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算简化因子 (不依赖 Qlib)"""
        close = df["close"]
        volume = df["volume"]
        high = df["high"]
        low = df["low"]
        returns = close.pct_change()

        factors = pd.DataFrame(index=df.index)
        factors["RSI_14"] = self._calc_rsi(close, 14)
        factors["MA5_MA20"] = close.rolling(5).mean() / close.rolling(20).mean()
        factors["VOL_RATIO"] = volume / volume.rolling(20).mean()
        factors["MOMENTUM_5"] = close.pct_change(5)
        factors["MOMENTUM_20"] = close.pct_change(20)
        factors["VOLATILITY_20"] = returns.rolling(20).std()
        factors["HIGH_LOW_RATIO"] = (high - low) / close
        factors["PRICE_MA60"] = close / close.rolling(60).mean()
        factors["VOL_CHANGE_5"] = volume.pct_change(5)
        factors["STD_10"] = returns.rolling(10).std()

        return factors

    def _calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算 RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - (100 / (1 + rs))


# 全局实例
factor_searcher = FactorSearcher()
