"""
预置策略模板库
=====================================================
T5.5: 提供开箱即用的策略模板，小白用户无需理解因子和模型细节

5 个预置模板:
  1. steady_growth   - 稳健成长 (大盘蓝筹, LightGBM, 低换手)
  2. momentum_rider  - 动量追踪 (动量因子为主, 中高频)
  3. value_hunter     - 价值挖掘 (基本面因子为主, 低频)
  4. tech_pioneer     - 科技先锋 (科技股池, LSTM)
  5. balanced_all     - 均衡配置 (多因子均衡, 中等仓位)

使用示例:
    from src.engine.strategy_templates import strategy_templates

    # 列出所有模板
    templates = strategy_templates.list_templates()

    # 从模板创建策略
    strategy = strategy_templates.create_from_template("steady_growth")
"""

from datetime import datetime
from typing import Optional

from loguru import logger

from src.engine.strategy_manager import strategy_manager
from src.models import StrategyConfig


class StrategyTemplate:
    """单个策略模板"""

    def __init__(
        self,
        template_id: str,
        name: str,
        description: str,
        model_type: str = "lightgbm",
        factors: Optional[list[str]] = None,
        stock_pool: Optional[list[str]] = None,
        max_position: float = 0.30,
        hyper_params: Optional[dict] = None,
        risk_level: str = "medium",       # low / medium / high
        rebalance_freq: str = "daily",    # daily / weekly
        tags: Optional[list[str]] = None,
    ):
        self.template_id = template_id
        self.name = name
        self.description = description
        self.model_type = model_type
        self.factors = factors or []
        self.stock_pool = stock_pool or []
        self.max_position = max_position
        self.hyper_params = hyper_params or {}
        self.risk_level = risk_level
        self.rebalance_freq = rebalance_freq
        self.tags = tags or []

    def to_dict(self) -> dict:
        """转字典 (返回副本, 避免外部修改影响内部状态)"""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type,
            "factors_count": len(self.factors),
            "stock_pool_size": len(self.stock_pool),
            "max_position": self.max_position,
            "risk_level": self.risk_level,
            "rebalance_freq": self.rebalance_freq,
            "tags": list(self.tags),  # 返回副本, 防止外部修改泄漏
        }


class StrategyTemplateLibrary:
    """预置策略模板库"""

    def __init__(self):
        self._templates: dict[str, StrategyTemplate] = {}
        self._init_builtin_templates()

    def _init_builtin_templates(self):
        """初始化内置模板"""

        # ─── 1. 稳健成长 ───
        self._templates["steady_growth"] = StrategyTemplate(
            template_id="steady_growth",
            name="稳健成长",
            description=(
                "大盘蓝筹股 + LightGBM 模型，追求稳定增长。"
                "适合风险偏好较低的投资者，持仓集中在大市值公司，换手率低。"
            ),
            model_type="lightgbm",
            factors=[
                "MA_5", "MA_20", "MA_60",          # 均线因子
                "RSI_14",                           # 相对强弱
                "STD_20",                           # 波动率
                "ROC_10", "ROC_20",                # 变化率
                "VOLUME_RATIO_5",                  # 量比
                "HIGH_LOW_RATIO",                  # 振幅
            ],
            stock_pool=[
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                "META", "JPM", "V", "JNJ", "PG",
            ],
            max_position=0.25,
            hyper_params={
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 31,
                "n_estimators": 200,
            },
            risk_level="low",
            rebalance_freq="weekly",
            tags=["蓝筹", "稳健", "低换手", "适合新手"],
        )

        # ─── 2. 动量追踪 ───
        self._templates["momentum_rider"] = StrategyTemplate(
            template_id="momentum_rider",
            name="动量追踪",
            description=(
                "以动量因子为核心，捕捉趋势行情。"
                "适合能承受一定波动的投资者，追求超额收益，换手率中等偏高。"
            ),
            model_type="lightgbm",
            factors=[
                "ROC_5", "ROC_10", "ROC_20",       # 多周期动量
                "MA_5", "MA_10",                   # 短期均线
                "RSI_14", "RSI_7",                # 相对强弱
                "VOLUME_RATIO_5", "VOLUME_RATIO_20",  # 量价配合
                "BETA_20",                         # Beta
                "MOMENTUM_20",                    # 20日动量
            ],
            stock_pool=[
                "AAPL", "NVDA", "AMD", "TSLA", "META",
                "NFLX", "AMZN", "MSFT", "GOOGL", "COIN",
            ],
            max_position=0.20,
            hyper_params={
                "learning_rate": 0.1,
                "max_depth": 8,
                "num_leaves": 63,
                "n_estimators": 300,
            },
            risk_level="high",
            rebalance_freq="daily",
            tags=["动量", "趋势", "高换手", "追求超额"],
        )

        # ─── 3. 价值挖掘 ───
        self._templates["value_hunter"] = StrategyTemplate(
            template_id="value_hunter",
            name="价值挖掘",
            description=(
                "以基本面因子为主，寻找被低估的优质公司。"
                "适合长期投资者，换手率低，持仓周期长。"
            ),
            model_type="linear",
            factors=[
                "PE_RATIO", "PB_RATIO",            # 估值因子
                "ROE", "ROA",                       # 盈利能力
                "REVENUE_GROWTH", "EPS_GROWTH",    # 成长性
                "DEBT_RATIO",                       # 财务健康
                "DIVIDEND_YIELD",                   # 股息率
                "MA_60", "MA_120",                  # 长期均线
            ],
            stock_pool=[
                "JPM", "V", "JNJ", "PG", "KO",
                "WMT", "HD", "DIS", "BAC", "C",
            ],
            max_position=0.30,
            hyper_params={
                "alpha": 0.01,
                "max_iter": 1000,
            },
            risk_level="low",
            rebalance_freq="weekly",
            tags=["价值", "基本面", "低换手", "长期持有"],
        )

        # ─── 4. 科技先锋 ───
        self._templates["tech_pioneer"] = StrategyTemplate(
            template_id="tech_pioneer",
            name="科技先锋",
            description=(
                "专注科技板块，使用 LSTM 深度学习模型捕捉非线性模式。"
                "适合看好科技行业、能承受较高波动的投资者。"
            ),
            model_type="lstm",
            factors=[
                "MA_5", "MA_10", "MA_20",          # 均线
                "RSI_14", "RSI_7",                # 相对强弱
                "ROC_5", "ROC_10",                # 动量
                "STD_10", "STD_20",              # 波动率
                "VOLUME_RATIO_5",                # 量比
                "HIGH_LOW_RATIO",               # 振幅
            ],
            stock_pool=[
                "AAPL", "MSFT", "GOOGL", "NVDA", "META",
                "AMD", "NFLX", "ADBE", "CRM", "INTC",
            ],
            max_position=0.20,
            hyper_params={
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.1,
                "n_epochs": 50,
                "lr": 1e-3,
            },
            risk_level="high",
            rebalance_freq="daily",
            tags=["科技", "LSTM", "深度学习", "高波动"],
        )

        # ─── 5. 均衡配置 ───
        self._templates["balanced_all"] = StrategyTemplate(
            template_id="balanced_all",
            name="均衡配置",
            description=(
                "多因子均衡策略，兼顾技术面和基本面。"
                "适合大多数投资者，风险和收益均衡，中等仓位。"
            ),
            model_type="lightgbm",
            factors=[
                "MA_5", "MA_20", "MA_60",          # 均线
                "RSI_14",                           # 相对强弱
                "ROC_10", "ROC_20",                # 动量
                "STD_20",                           # 波动率
                "PE_RATIO",                         # 估值
                "VOLUME_RATIO_5",                  # 量比
                "HIGH_LOW_RATIO",                  # 振幅
                "BETA_20",                          # Beta
            ],
            stock_pool=[
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                "JPM", "V", "JNJ", "PG", "HD",
            ],
            max_position=0.25,
            hyper_params={
                "learning_rate": 0.05,
                "max_depth": 7,
                "num_leaves": 50,
                "n_estimators": 250,
            },
            risk_level="medium",
            rebalance_freq="daily",
            tags=["均衡", "多因子", "中等风险", "适合大多数"],
        )

    def list_templates(self) -> list[dict]:
        """列出所有模板摘要"""
        return [t.to_dict() for t in self._templates.values()]

    def get_template(self, template_id: str) -> Optional[StrategyTemplate]:
        """获取模板详情"""
        return self._templates.get(template_id)

    def create_from_template(
        self,
        template_id: str,
        strategy_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Optional[StrategyConfig]:
        """
        从模板创建策略

        Args:
            template_id: 模板 ID
            strategy_id: 自定义策略 ID (默认用模板 ID + 时间戳)
            name: 自定义策略名称 (默认用模板名称)

        Returns:
            StrategyConfig

        Raises:
            ValueError: 模板 ID 不存在或为空时抛出
        """
        # 校验模板 ID 非空
        if not template_id or not template_id.strip():
            raise ValueError(f"模板 ID 不能为空: {template_id!r}")

        template = self._templates.get(template_id)
        if not template:
            raise ValueError(f"策略模板不存在: {template_id}")

        # 生成策略 ID 和名称
        if not strategy_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_id = f"{template_id}_{timestamp}"
        if not name:
            name = template.name

        # 创建策略
        strategy = strategy_manager.create_strategy(
            strategy_id=strategy_id,
            name=name,
            model_type=template.model_type,
            factors=template.factors.copy(),
            stock_pool=template.stock_pool.copy(),
            max_position=template.max_position,
            hyper_params=template.hyper_params.copy(),
        )

        # 设置调仓频率
        strategy.rebalance_freq = template.rebalance_freq
        strategy_manager._save(strategy)

        logger.info(f"从模板 '{template_id}' 创建策略: {strategy_id} ({name})")
        return strategy


# 全局实例
strategy_templates = StrategyTemplateLibrary()
