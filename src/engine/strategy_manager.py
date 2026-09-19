"""
策略管理器 - 策略生命周期管理
=====================================================
T2.6: 策略管理器
- 创建 / 更新 / 删除策略
- 策略状态机: draft -> testing -> active -> retired
- 策略持久化 (JSON)
"""

import json
from datetime import datetime
from typing import Optional

from src.models import BacktestResult, StrategyConfig
from loguru import logger

from src.utils.config import config


class StrategyManager:
    """策略管理器"""

    def __init__(self):
        self.strategies_dir = config.DATA_DIR / "strategies"
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
        self._strategies: dict[str, StrategyConfig] = {}
        self._backtest_results: dict[str, BacktestResult] = {}
        self._load_all()

    def create_strategy(
        self,
        strategy_id: str,
        name: str,
        model_type: str = "lightgbm",
        factors: Optional[list[str]] = None,
        stock_pool: Optional[list[str]] = None,
        max_position: float = 0.30,
        hyper_params: Optional[dict] = None,
    ) -> StrategyConfig:
        """创建新策略"""
        if strategy_id in self._strategies:
            logger.warning(f"策略 {strategy_id} 已存在, 返回已有策略")
            return self._strategies[strategy_id]

        strategy = StrategyConfig(
            strategy_id=strategy_id,
            name=name,
            model_type=model_type,
            factors=factors or [],
            stock_pool=stock_pool or [],
            max_position=max_position,
            hyper_params=hyper_params or {},
            status="draft",
        )
        self._strategies[strategy_id] = strategy
        self._save(strategy)
        logger.info(f"策略创建: {strategy_id} ({name})")
        return strategy

    def get_strategy(self, strategy_id: str) -> Optional[StrategyConfig]:
        """获取策略"""
        return self._strategies.get(strategy_id)

    def list_strategies(self, status: Optional[str] = None) -> list[StrategyConfig]:
        """列出策略"""
        strategies = list(self._strategies.values())
        if status:
            strategies = [s for s in strategies if s.status == status]
        return strategies

    def update_strategy(self, strategy_id: str, **kwargs) -> Optional[StrategyConfig]:
        """更新策略"""
        strategy = self._strategies.get(strategy_id)
        if not strategy:
            logger.error(f"策略不存在: {strategy_id}")
            return None

        for key, value in kwargs.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)
        strategy.updated_at = datetime.now()
        self._save(strategy)
        logger.info(f"策略更新: {strategy_id}")
        return strategy

    def delete_strategy(self, strategy_id: str) -> bool:
        """删除策略 (同时清理回测结果文件, 避免残留孤儿数据)"""
        if strategy_id not in self._strategies:
            return False
        del self._strategies[strategy_id]
        # 删除策略配置文件
        filepath = self.strategies_dir / f"{strategy_id}.json"
        if filepath.exists():
            filepath.unlink()
        # 修复: 同步删除回测结果文件, 否则重启后 _load_all 会加载成孤儿数据
        backtest_file = self.strategies_dir / f"{strategy_id}_backtest.json"
        if backtest_file.exists():
            backtest_file.unlink()
        # 清理内存中的回测结果缓存
        self._backtest_results.pop(strategy_id, None)
        logger.info(f"策略删除: {strategy_id}")
        return True

    def activate_strategy(self, strategy_id: str) -> bool:
        """激活策略"""
        return self._set_status(strategy_id, "active")

    def retire_strategy(self, strategy_id: str) -> bool:
        """退役策略"""
        return self._set_status(strategy_id, "retired")

    def set_testing(self, strategy_id: str) -> bool:
        """设为测试中"""
        return self._set_status(strategy_id, "testing")

    def record_backtest(self, strategy_id: str, result: BacktestResult):
        """记录回测结果 (IO 失败时记日志不崩溃)"""
        self._backtest_results[strategy_id] = result
        # 保存
        result_file = self.strategies_dir / f"{strategy_id}_backtest.json"
        try:
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"回测结果保存失败 [{result_file}]: {e}")
        logger.info(f"回测结果已保存: {strategy_id} -> {result.grade}级")

    def get_backtest_result(self, strategy_id: str) -> Optional[BacktestResult]:
        """获取回测结果"""
        return self._backtest_results.get(strategy_id)

    def get_active_strategies(self) -> list[StrategyConfig]:
        """获取所有活跃策略"""
        return self.list_strategies(status="active")

    def get_summary(self) -> list[dict]:
        """策略摘要"""
        summaries = []
        for s in self._strategies.values():
            bt = self._backtest_results.get(s.strategy_id)
            summaries.append({
                "strategy_id": s.strategy_id,
                "name": s.name,
                "model_type": s.model_type,
                "status": s.status,
                "stock_pool_size": len(s.stock_pool),
                "factors_count": len(s.factors),
                "backtest_grade": bt.grade.value if bt else "未回测",
                "annual_return": bt.annual_return if bt else 0,
                "sharpe": bt.sharpe_ratio if bt else 0,
                "max_drawdown": bt.max_drawdown if bt else 0,
                "updated_at": s.updated_at.strftime("%Y-%m-%d %H:%M"),
            })
        return summaries

    # ─── 内部方法 ───

    def _set_status(self, strategy_id: str, status: str) -> bool:
        strategy = self._strategies.get(strategy_id)
        if not strategy:
            return False
        strategy.status = status
        strategy.updated_at = datetime.now()
        self._save(strategy)
        logger.info(f"策略状态变更: {strategy_id} -> {status}")
        return True

    def _save(self, strategy: StrategyConfig):
        """保存策略到 JSON (IO 失败时记日志不崩溃)"""
        filepath = self.strategies_dir / f"{strategy.strategy_id}.json"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(strategy.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"策略保存失败 [{filepath}]: {e}")

    def _load_all(self):
        """加载所有策略"""
        for filepath in self.strategies_dir.glob("*.json"):
            if filepath.name.endswith("_backtest.json"):
                # 加载回测结果
                strategy_id = filepath.stem.replace("_backtest", "")
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._backtest_results[strategy_id] = BacktestResult(**data)
                except Exception as e:
                    logger.error(f"加载回测结果失败 [{filepath}]: {e}")
            else:
                # 加载策略配置
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    strategy = StrategyConfig(**data)
                    self._strategies[strategy.strategy_id] = strategy
                except Exception as e:
                    logger.error(f"加载策略失败 [{filepath}]: {e}")

        logger.info(f"策略管理器: {len(self._strategies)} 个策略, "
                     f"{len(self._backtest_results)} 个回测结果")


# 全局实例
strategy_manager = StrategyManager()
