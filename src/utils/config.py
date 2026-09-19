"""
全局配置加载器 - 从 .env 和 settings.yaml 读取配置
=====================================================
T1.2: 配置系统 (settings.yaml + .env)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


# ─── 项目根目录 (模块级常量，兼容 PyInstaller 打包) ───
def _get_project_root() -> Path:
    """获取项目根目录，兼容开发环境和 PyInstaller 打包环境"""
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后: exe 所在目录
        return Path(sys.executable).parent
    else:
        # 开发环境: src/utils/config.py -> 上三级是项目根
        return Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT = _get_project_root()


class Config:
    """全局配置单例"""

    _instance = None

    # ─── Alpaca 交易 ───
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets/v2"

    # ─── Finnhub 数据 ───
    FINNHUB_API_KEY: str = ""

    # ─── FRED 宏观经济数据 ───
    FRED_API_KEY: str = ""

    # ─── Financial Modeling Prep 财务数据 ───
    FMP_API_KEY: str = ""

    # ─── Qlib ───
    QLIB_DATA_DIR: str = "./data/qlib_data"

    # ─── 系统配置 ───
    LOG_LEVEL: str = "INFO"
    TRADING_MODE: str = "paper"       # paper | live
    MAX_POSITION_PCT: float = 0.30    # 单票最大仓位
    MAX_DAILY_LOSS: float = 0.05      # 日最大亏损
    MAX_DRAWDOWN: float = 0.15        # 最大回撤

    # ─── 路径 ───
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CACHE_DIR: Path = PROJECT_ROOT / "data" / "cache"
    QLIB_DATA_PATH: Path = PROJECT_ROOT / "data" / "qlib_data"
    REPORTS_DIR: Path = PROJECT_ROOT / "data" / "reports"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # ─── 数据缓存 ───
    CACHE_TTL_HOURS: int = 12         # 行情数据缓存有效期(小时)
    NEWS_CACHE_TTL_MINUTES: int = 30  # 新闻数据缓存有效期(分钟)

    # ─── 风控 ───
    MAX_TOTAL_POSITIONS: int = 10
    SINGLE_STOP_LOSS: float = 0.05
    DAILY_TRADE_LIMIT: int = 20
    MIN_PAPER_DAYS: int = 30

    # ─── 实盘安全配置 (Phase 6 新增) ───
    LIVE_MAX_POSITION_PCT: float = 0.15       # 实盘单票仓位上限 (更严格)
    LIVE_DAILY_TRADE_LIMIT: int = 10           # 实盘日交易次数上限
    LIVE_PROGRESSIVE_WEEKS: int = 3            # 实盘渐进式仓位周数
    LIVE_MIN_GRADE: str = "B"                  # 实盘要求最低策略评级
    LIVE_MAX_DRAWDOWN: float = 0.10            # 实盘允许的最大回撤

    # ─── 实时行情配置 (Phase 7 T1 新增) ───
    REALTIME_ENABLED: bool = True              # 是否启用实时行情
    REALTIME_DATA_SOURCE: str = "both"          # alpaca | finnhub | both
    REALTIME_PRICE_ALERT_THRESHOLD: float = 0.03  # 价格异动告警阈值 (3%)
    REALTIME_RECONNECT_INTERVAL: int = 5         # WebSocket 重连间隔 (秒)
    REALTIME_HEARTBEAT_INTERVAL: int = 30        # 心跳间隔 (秒)

    # ─── 选股配置 ───
    MAX_CANDIDATES: int = 5           # 调研阶段最大候选股票数
    TOP_K_SIGNALS: int = 5            # 信号生成阶段选前K只做多
    WATCHLIST: list = None            # 关注股票列表 (None=使用默认)

    # ─── 加密货币配置 (Phase 8 新增) ───
    CRYPTO_ENABLED: bool = True                    # 是否启用加密货币模块
    CRYPTO_TRADING_ENABLED: bool = False            # 是否启用加密货币交易 (默认关闭，仅数据采集)
    CRYPTO_DATA_SOURCE: str = "alpaca"              # 数据源: alpaca | ccxt
    CRYPTO_TRADING_PAIRS: list = None               # 交易对列表 (None=使用默认)
    CRYPTO_MAX_POSITION_PCT: float = 0.20           # 单币最大仓位 (20%，比股票更保守)
    CRYPTO_MAX_DAILY_LOSS: float = 0.08             # 日最大亏损 (8%，加密货币波动大)
    CRYPTO_MAX_DRAWDOWN: float = 0.25               # 最大回撤 (25%)
    CRYPTO_SINGLE_STOP_LOSS: float = 0.10           # 单币止损 (10%)
    CRYPTO_DAILY_TRADE_LIMIT: int = 50              # 日交易次数 (24h 交易，上限更高)
    CRYPTO_MAX_TOTAL_POSITIONS: int = 8             # 最大持仓币种数
    CRYPTO_LEVERAGE: int = 1                        # 杠杆倍数 (1 = 无杠杆)
    CRYPTO_MIN_PAPER_DAYS: int = 45                 # Paper Trading 最少天数 (比股票更长)
    CRYPTO_SIGNAL_SCAN_INTERVAL_HOURS: int = 4      # 信号扫描间隔 (小时)
    CRYPTO_RISK_CHECK_INTERVAL_HOURS: int = 1       # 风控检查间隔 (小时)
    CRYPTO_STOPLOSS_CHECK_INTERVAL_MINUTES: int = 15  # 止损检查间隔 (分钟)

    # ─── 加密货币默认交易对 ───
    DEFAULT_CRYPTO_PAIRS: list = [
        "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD",
        "AVAX/USD", "DOT/USD", "POL/USD", "LINK/USD",
    ]

    # ─── 部署模式 (v0.5.0 分布式架构新增) ───
    DEPLOY_MODE: str = "local"              # local | cloud
    QLIB_ENABLED: bool = True               # 是否启用 Qlib (云端禁用)
    TORCH_ENABLED: bool = True              # 是否启用 PyTorch (云端禁用)
    MODEL_API_KEY: str = ""                 # 模型同步 API 密钥
    CLOUD_API_URL: str = ""                 # 云端 API 地址 (本地上传用)
    CLOUD_API_PORT: str = "8000"            # 云端 API 端口
    CLOUD_API_HOST: str = "0.0.0.0"         # API 监听地址

    # ─── 自然语言选股配置 ───
    SCREEN_UNIVERSE_SIZE: int = 50        # 预筛选进入详细分析的最大数量
    SCREEN_WEIGHT_TECH: float = 0.25      # 技术面权重
    SCREEN_WEIGHT_FUND: float = 0.30      # 基本面权重
    SCREEN_WEIGHT_SENT: float = 0.20      # 情绪面权重
    SCREEN_WEIGHT_MOM: float = 0.25       # 动量面权重

    # ─── 默认关注列表 (当 WATCHLIST 为 None 时使用) ───
    DEFAULT_WATCHLIST: list = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "AMD", "NFLX", "JPM",
        "V", "DIS", "BABA", "INTC", "CRM",
        "ORCL", "ADBE", "PYPL", "UBER", "COIN",
    ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """从 .env 加载配置"""
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"配置文件已加载: {env_path}")
        else:
            logger.warning(f".env 文件不存在: {env_path}")

        # 读取环境变量
        self.ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
        self.ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
        self.ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
        self.FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
        # 安全修复: 密钥不硬编码在源码中, 从 .env 读取 (FRED key 已迁移至 .env)
        self.FRED_API_KEY = os.getenv("FRED_API_KEY", "")
        self.FMP_API_KEY = os.getenv("FMP_API_KEY", "")
        self.QLIB_DATA_DIR = os.getenv("QLIB_DATA_DIR", "./data/qlib_data")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.TRADING_MODE = os.getenv("TRADING_MODE", "paper")
        self.MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.30"))
        self.MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "0.05"))
        self.MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.15"))

        # 实时行情配置 (Phase 7 T1 新增)
        self.REALTIME_ENABLED = os.getenv("REALTIME_ENABLED", "true").lower() == "true"
        self.REALTIME_DATA_SOURCE = os.getenv("REALTIME_DATA_SOURCE", "both")
        self.REALTIME_PRICE_ALERT_THRESHOLD = float(os.getenv("REALTIME_PRICE_ALERT_THRESHOLD", "0.03"))
        self.REALTIME_RECONNECT_INTERVAL = int(os.getenv("REALTIME_RECONNECT_INTERVAL", "5"))
        self.REALTIME_HEARTBEAT_INTERVAL = int(os.getenv("REALTIME_HEARTBEAT_INTERVAL", "30"))

        # 选股配置
        self.MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "5"))
        self.TOP_K_SIGNALS = int(os.getenv("TOP_K_SIGNALS", "5"))

        # 部署模式 (v0.5.0 分布式架构新增)
        self.DEPLOY_MODE = os.getenv("DEPLOY_MODE", "local")
        self.QLIB_ENABLED = os.getenv("QLIB_ENABLED", "true").lower() == "true"
        self.TORCH_ENABLED = os.getenv("TORCH_ENABLED", "true").lower() == "true"
        self.MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
        self.CLOUD_API_URL = os.getenv("CLOUD_API_URL", "")
        self.CLOUD_API_PORT = os.getenv("CLOUD_API_PORT", "8000")
        self.CLOUD_API_HOST = os.getenv("CLOUD_API_HOST", "0.0.0.0")

        # 自然语言选股配置
        self.SCREEN_UNIVERSE_SIZE = int(os.getenv("SCREEN_UNIVERSE_SIZE", "50"))
        self.SCREEN_WEIGHT_TECH = float(os.getenv("SCREEN_WEIGHT_TECH", "0.25"))
        self.SCREEN_WEIGHT_FUND = float(os.getenv("SCREEN_WEIGHT_FUND", "0.30"))
        self.SCREEN_WEIGHT_SENT = float(os.getenv("SCREEN_WEIGHT_SENT", "0.20"))
        self.SCREEN_WEIGHT_MOM = float(os.getenv("SCREEN_WEIGHT_MOM", "0.25"))

        # 加密货币配置 (Phase 8 新增)
        self.CRYPTO_ENABLED = os.getenv("CRYPTO_ENABLED", "true").lower() == "true"
        self.CRYPTO_TRADING_ENABLED = os.getenv("CRYPTO_TRADING_ENABLED", "false").lower() == "true"
        self.CRYPTO_DATA_SOURCE = os.getenv("CRYPTO_DATA_SOURCE", "alpaca")
        self.CRYPTO_MAX_POSITION_PCT = float(os.getenv("CRYPTO_MAX_POSITION_PCT", "0.20"))
        self.CRYPTO_MAX_DAILY_LOSS = float(os.getenv("CRYPTO_MAX_DAILY_LOSS", "0.08"))
        self.CRYPTO_MAX_DRAWDOWN = float(os.getenv("CRYPTO_MAX_DRAWDOWN", "0.25"))
        self.CRYPTO_SINGLE_STOP_LOSS = float(os.getenv("CRYPTO_SINGLE_STOP_LOSS", "0.10"))
        self.CRYPTO_DAILY_TRADE_LIMIT = int(os.getenv("CRYPTO_DAILY_TRADE_LIMIT", "50"))
        self.CRYPTO_MAX_TOTAL_POSITIONS = int(os.getenv("CRYPTO_MAX_TOTAL_POSITIONS", "8"))
        self.CRYPTO_LEVERAGE = int(os.getenv("CRYPTO_LEVERAGE", "1"))
        self.CRYPTO_MIN_PAPER_DAYS = int(os.getenv("CRYPTO_MIN_PAPER_DAYS", "45"))
        self.CRYPTO_SIGNAL_SCAN_INTERVAL_HOURS = int(os.getenv("CRYPTO_SIGNAL_SCAN_INTERVAL_HOURS", "4"))
        self.CRYPTO_RISK_CHECK_INTERVAL_HOURS = int(os.getenv("CRYPTO_RISK_CHECK_INTERVAL_HOURS", "1"))
        self.CRYPTO_STOPLOSS_CHECK_INTERVAL_MINUTES = int(os.getenv("CRYPTO_STOPLOSS_CHECK_INTERVAL_MINUTES", "15"))

        # 加密货币交易对 (从 .env 读取, 逗号分隔)
        crypto_pairs_str = os.getenv("CRYPTO_TRADING_PAIRS", "")
        if crypto_pairs_str:
            self.CRYPTO_TRADING_PAIRS = [s.strip() for s in crypto_pairs_str.split(",") if s.strip()]
        else:
            self.CRYPTO_TRADING_PAIRS = None  # 使用 DEFAULT_CRYPTO_PAIRS

        # 关注列表 (从 .env 读取, 逗号分隔)
        watchlist_str = os.getenv("WATCHLIST", "")
        if watchlist_str:
            self.WATCHLIST = [s.strip().upper() for s in watchlist_str.split(",") if s.strip()]
        else:
            self.WATCHLIST = None  # 使用 DEFAULT_WATCHLIST

        # 实盘安全配置 (Phase 6 新增)
        self.LIVE_MAX_POSITION_PCT = float(os.getenv("LIVE_MAX_POSITION_PCT", "0.15"))
        self.LIVE_DAILY_TRADE_LIMIT = int(os.getenv("LIVE_DAILY_TRADE_LIMIT", "10"))
        self.LIVE_PROGRESSIVE_WEEKS = int(os.getenv("LIVE_PROGRESSIVE_WEEKS", "3"))
        self.LIVE_MIN_GRADE = os.getenv("LIVE_MIN_GRADE", "B")
        self.LIVE_MAX_DRAWDOWN = float(os.getenv("LIVE_MAX_DRAWDOWN", "0.10"))

        # 更新路径
        self.QLIB_DATA_PATH = PROJECT_ROOT / self.QLIB_DATA_DIR.strip("./").strip(".\\")
        self.CACHE_DIR = self.DATA_DIR / "cache"
        self.REPORTS_DIR = self.DATA_DIR / "reports"

        # 创建必要目录
        for d in [self.DATA_DIR, self.CACHE_DIR, self.QLIB_DATA_PATH,
                  self.REPORTS_DIR, self.LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def is_paper_trading(self) -> bool:
        """是否为模拟盘模式"""
        return self.TRADING_MODE == "paper"

    def has_alpaca_keys(self) -> bool:
        return bool(self.ALPACA_API_KEY and self.ALPACA_SECRET_KEY
                     and not self.ALPACA_API_KEY.startswith("your_")
                     and not self.ALPACA_SECRET_KEY.startswith("your_"))

    def has_finnhub_key(self) -> bool:
        return bool(self.FINNHUB_API_KEY
                     and not self.FINNHUB_API_KEY.startswith("your_"))

    def has_fred_key(self) -> bool:
        return bool(self.FRED_API_KEY
                     and not self.FRED_API_KEY.startswith("your_"))

    def has_fmp_key(self) -> bool:
        return bool(self.FMP_API_KEY
                     and not self.FMP_API_KEY.startswith("your_"))

    def get_crypto_pairs(self) -> list:
        """获取加密货币交易对列表 (优先用户配置, 否则使用默认)"""
        if self.CRYPTO_TRADING_PAIRS:
            return self.CRYPTO_TRADING_PAIRS
        return self.DEFAULT_CRYPTO_PAIRS

    def has_crypto_enabled(self) -> bool:
        """加密货币模块是否启用"""
        return self.CRYPTO_ENABLED

    def has_crypto_trading_enabled(self) -> bool:
        """加密货币交易是否启用 (需要同时满足 ENABLED + TRADING_ENABLED)"""
        return self.CRYPTO_ENABLED and self.CRYPTO_TRADING_ENABLED

    def to_dict(self) -> dict:
        """返回所有配置(脱敏)"""
        return {
            "TRADING_MODE": self.TRADING_MODE,
            "ALPACA_API_KEY": f"***{self.ALPACA_API_KEY[-4:]}" if self.has_alpaca_keys() else "未配置",
            "FINNHUB_API_KEY": f"***{self.FINNHUB_API_KEY[-4:]}" if self.has_finnhub_key() else "未配置",
            "FRED_API_KEY": f"***{self.FRED_API_KEY[-4:]}" if self.has_fred_key() else "未配置",
            "FMP_API_KEY": f"***{self.FMP_API_KEY[-4:]}" if self.has_fmp_key() else "未配置",
            "QLIB_DATA_DIR": str(self.QLIB_DATA_PATH),
            "CACHE_DIR": str(self.CACHE_DIR),
            "MAX_POSITION_PCT": self.MAX_POSITION_PCT,
            "MAX_DAILY_LOSS": self.MAX_DAILY_LOSS,
            "MAX_DRAWDOWN": self.MAX_DRAWDOWN,
            "LOG_LEVEL": self.LOG_LEVEL,
            "MAX_CANDIDATES": self.MAX_CANDIDATES,
            "TOP_K_SIGNALS": self.TOP_K_SIGNALS,
            "SCREEN_UNIVERSE_SIZE": self.SCREEN_UNIVERSE_SIZE,
            "DEPLOY_MODE": self.DEPLOY_MODE,
            "QLIB_ENABLED": self.QLIB_ENABLED,
            "TORCH_ENABLED": self.TORCH_ENABLED,
            # 加密货币配置
            "CRYPTO_ENABLED": self.CRYPTO_ENABLED,
            "CRYPTO_TRADING_ENABLED": self.CRYPTO_TRADING_ENABLED,
            "CRYPTO_DATA_SOURCE": self.CRYPTO_DATA_SOURCE,
            "CRYPTO_TRADING_PAIRS": self.get_crypto_pairs(),
            "CRYPTO_MAX_POSITION_PCT": self.CRYPTO_MAX_POSITION_PCT,
            "CRYPTO_MAX_DAILY_LOSS": self.CRYPTO_MAX_DAILY_LOSS,
            "CRYPTO_MAX_DRAWDOWN": self.CRYPTO_MAX_DRAWDOWN,
            "CRYPTO_SINGLE_STOP_LOSS": self.CRYPTO_SINGLE_STOP_LOSS,
            "CRYPTO_DAILY_TRADE_LIMIT": self.CRYPTO_DAILY_TRADE_LIMIT,
            "CRYPTO_MAX_TOTAL_POSITIONS": self.CRYPTO_MAX_TOTAL_POSITIONS,
        }

    def save_to_env(self, updates: dict) -> str:
        """
        将配置变更写入 .env 文件
        :param updates: {KEY: value} 字典，value 为 None 表示不修改该字段
        :return: 保存结果消息
        """
        env_path = PROJECT_ROOT / ".env"

        # 读取现有 .env 内容
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = []

        # 构建已有 key -> 行号 映射
        existing_keys = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                existing_keys[key] = i

        # 更新或追加配置项
        changed_count = 0
        for key, value in updates.items():
            if value is None:
                continue  # 跳过不修改的字段

            # 布尔值转为小写字符串
            if isinstance(value, bool):
                value = "true" if value else "false"
            # 列表转为逗号分隔字符串
            elif isinstance(value, list):
                value = ",".join(str(v) for v in value)

            line_str = f"{key}={value}\n"

            if key in existing_keys:
                lines[existing_keys[key]] = line_str
            else:
                # 追加新行
                if lines and not lines[-1].endswith("\n"):
                    lines[-1] += "\n"
                lines.append(line_str)
            changed_count += 1

        # 写回 .env
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        logger.info(f"配置已保存到 .env, 更新了 {changed_count} 项")
        return f"已更新 {changed_count} 项配置到 .env"

    def __repr__(self):
        return f"Config({self.to_dict()})"


# ─── 全局单例 ───
config = Config()
