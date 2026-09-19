"""
核心数据模型 - 全系统统一数据结构
=====================================================
基于 Pydantic v2, 定义所有跨层数据传输对象
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════
# 枚举
# ═══════════════════════════════════════════════════════

class Direction(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class StrategyGrade(str, Enum):
    A = "A"   # 优秀: 年化>20%, 夏普>1.5, 回撤<10%
    B = "B"   # 良好: 年化>15%, 夏普>1.0, 回撤<15%
    C = "C"   # 一般: 年化>10%, 夏普>0.5, 回撤<20%
    D = "D"   # 差: 不达标


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class AssetType(str, Enum):
    """资产类型"""
    STOCK = "stock"
    CRYPTO = "crypto"


# ═══════════════════════════════════════════════════════
# 数据层模型
# ═══════════════════════════════════════════════════════

class StockCandidate(BaseModel):
    """调研发现的候选股票"""
    symbol: str
    name: str = ""
    sector: Optional[str] = None
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    mention_count: int = 0
    key_events: list[str] = Field(default_factory=list)
    plain_reason: str = ""        # 小白解读: 为什么推荐


class CryptoCandidate(BaseModel):
    """加密货币候选"""
    symbol: str                     # 如 "BTC/USD"
    name: str = ""                  # 如 "Bitcoin"
    market_cap: float = 0.0         # 市值 (USD)
    volume_24h: float = 0.0         # 24h 成交量 (USD)
    price_change_24h: float = 0.0   # 24h 涨跌幅 (%)
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    mention_count: int = 0
    key_events: list[str] = Field(default_factory=list)
    plain_reason: str = ""


class ResearchReport(BaseModel):
    """市场调研报告"""
    report_id: str
    date: datetime
    candidate_stocks: list[StockCandidate] = Field(default_factory=list)
    crypto_candidates: list[CryptoCandidate] = Field(default_factory=list)  # 加密货币候选
    market_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    key_events: list[dict] = Field(default_factory=list)
    asset_type: AssetType = AssetType.STOCK  # 资产类型: stock | crypto
    plain_summary: str = ""


class StockPick(BaseModel):
    """自然语言选股 - 单只入选股票"""
    symbol: str
    name: str = ""
    price: float = 0.0
    sector: Optional[str] = None
    composite_score: float = 0.0     # 综合评分 (0~100)
    tech_score: float = 0.0          # 技术面评分 (0~100)
    fundamental_score: float = 0.0   # 基本面评分 (0~100)
    sentiment_score: float = 0.0     # 情绪面评分 (0~100)
    momentum_score: float = 0.0      # 动量面评分 (0~100)
    llm_reason: str = ""             # LLM 推荐理由 (通俗中文)


class StockScreenResult(BaseModel):
    """自然语言选股 - 选股结果"""
    query: str = ""                       # 用户原始查询
    params: dict = Field(default_factory=dict)  # 筛选参数快照
    picks: list[StockPick] = Field(default_factory=list)  # 最终选中的股票
    total_scanned: int = 0               # 全美股扫描总数
    total_filtered: int = 0              # 价格筛选后数量
    plain_summary: str = ""              # 小白解读


class StrategyConfig(BaseModel):
    """策略配置"""
    strategy_id: str
    name: str
    model_type: str = "lightgbm"  # lightgbm | lstm | linear
    factors: list[str] = Field(default_factory=list)
    hyper_params: dict = Field(default_factory=dict)
    stock_pool: list[str] = Field(default_factory=list)
    rebalance_freq: str = "daily"
    max_position: float = 0.30
    status: str = "draft"  # draft | testing | active | retired
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class FactorResult(BaseModel):
    """因子搜索结果"""
    name: str
    ic: float = 0.0          # 信息系数 (Pearson)
    rank_ic: float = 0.0     # Rank IC (Spearman)
    ir: float = 0.0          # 信息比率 = IC均值 / IC标准差
    is_effective: bool = False


class ModelResult(BaseModel):
    """模型训练结果"""
    model_type: str
    train_ic: float = 0.0
    valid_ic: float = 0.0
    train_rank_ic: float = 0.0
    valid_rank_ic: float = 0.0
    model_path: str = ""
    params: dict = Field(default_factory=dict)


class BacktestResult(BaseModel):
    """回测结果"""
    strategy_id: str
    period: tuple[str, str] = ("", "")
    initial_capital: float = 100000.0
    final_value: float = 0.0
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    daily_returns: list[float] = Field(default_factory=list)
    equity_curve: list[float] = Field(default_factory=list)
    grade: StrategyGrade = StrategyGrade.D
    plain_summary: str = ""


# ═══════════════════════════════════════════════════════
# 交易层模型
# ═══════════════════════════════════════════════════════

class TradeSignal(BaseModel):
    """交易信号"""
    signal_id: str
    strategy_id: str = ""
    symbol: str
    direction: Direction
    target_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    asset_type: AssetType = AssetType.STOCK  # 资产类型: stock | crypto
    created_at: datetime = Field(default_factory=datetime.now)


class Order(BaseModel):
    """订单"""
    order_id: str = ""
    alpaca_order_id: Optional[str] = None
    symbol: str
    side: Direction
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    asset_type: AssetType = AssetType.STOCK  # 资产类型: stock | crypto
    filled_price: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None


class Position(BaseModel):
    """持仓"""
    symbol: str
    qty: float
    qty_available: float = 0.0       # 可用数量 (扣除挂单锁定)
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float                 # 占总资产比例
    asset_type: AssetType = AssetType.STOCK  # 资产类型: stock | crypto
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class Account(BaseModel):
    """账户信息"""
    equity: float                 # 总资产
    cash: float                   # 现金
    buying_power: float           # 购买力
    day_trade_count: int = 0
    pattern_day_trader: bool = False
    trading_mode: TradingMode = TradingMode.PAPER
    last_equity: float = 0.0      # 昨日总资产(用于算日亏损)


# ═══════════════════════════════════════════════════════
# 风控模型
# ═══════════════════════════════════════════════════════

class RiskCheckItem(BaseModel):
    """单项风控检查结果"""
    rule_name: str
    passed: bool
    reason: str = ""


class RiskCheckResult(BaseModel):
    """风控检查结果"""
    passed: bool
    failed_checks: list[RiskCheckItem] = Field(default_factory=list)
    reason: str = ""


class RiskStatus(BaseModel):
    """组合级风控状态"""
    max_single_position: float = 0.0
    max_total_drawdown: float = 0.0
    daily_loss_limit: float = 0.0
    current_drawdown: float = 0.0
    current_daily_loss: float = 0.0
    is_alert: bool = False
    alerts: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════
# 多策略组合模型 (Phase 6 T6.2)
# ═══════════════════════════════════════════════════════

class AllocationMethod(str, Enum):
    """权重分配方式"""
    EQUAL = "equal"              # 等权分配
    RISK_PARITY = "risk_parity"  # 风险平价 (按波动率反比)
    PERFORMANCE = "performance"  # 绩效加权 (按夏普/IC)


class PortfolioConfig(BaseModel):
    """多策略组合配置"""
    portfolio_id: str
    name: str
    strategy_allocations: dict[str, float] = Field(default_factory=dict)  # strategy_id -> weight
    allocation_method: AllocationMethod = AllocationMethod.EQUAL
    risk_budget: float = Field(default=0.10, ge=0.0, le=1.0)  # 组合风险预算
    rebalance_freq: str = "daily"  # daily | weekly | monthly
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: str = "active"  # active | paused


class AggregatedSignal(BaseModel):
    """多策略聚合信号"""
    signal_id: str = ""
    symbol: str
    direction: Direction
    target_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    source_strategies: list[str] = Field(default_factory=list)  # 信号来源策略列表
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class PortfolioStatus(BaseModel):
    """组合运行状态"""
    portfolio_id: str
    name: str
    strategy_count: int = 0
    total_weight: float = 0.0
    last_rebalance: Optional[datetime] = None
    signals_today: int = 0
    allocation_method: str = "equal"
    strategy_weights: dict[str, float] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════
# 实时行情模型 (Phase 7 T1)
# ═══════════════════════════════════════════════════════

class RealtimeQuote(BaseModel):
    """实时报价"""
    symbol: str
    price: float
    change: float = 0.0              # 涨跌额
    change_pct: float = 0.0          # 涨跌幅 (%)
    volume: int = 0                 # 成交量
    timestamp: datetime = Field(default_factory=datetime.now)
    source: str = "unknown"         # alpaca | finnhub | rest


class RealtimeBar(BaseModel):
    """1分钟K线 (由逐笔成交聚合)"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)  # K线开始时间


class PriceAlert(BaseModel):
    """价格异动告警"""
    symbol: str
    alert_type: str = "big_move"     # big_move | stop_loss | take_profit | custom
    price: float = 0.0               # 触发时价格
    threshold: float = 0.0           # 触发阈值
    change_pct: float = 0.0          # 触发时涨跌幅
    message: str = ""
    triggered_at: datetime = Field(default_factory=datetime.now)
