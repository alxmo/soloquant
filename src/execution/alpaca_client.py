"""
Alpaca 交易 API 客户端封装
=====================================================
T1.7: Alpaca API 客户端 (Paper Trading)
- 账户查询 / 订单管理 / 持仓查询
- 支持 Paper Trading 和 Live 模式
- 懒初始化 + 重试 + 断路器保护 (稳定性加固)
"""

from datetime import datetime
from typing import Optional

from loguru import logger

from src.models import (
    Account, Direction, Order, OrderType, OrderStatus,
    Position, TradingMode, AssetType,
)
from src.utils.config import config
from src.utils.retry import retry, get_breaker, CircuitBreakerOpenError


class AlpacaClient:
    """Alpaca 交易 API 封装"""

    # 断路器: 连续失败5次后熔断，30秒后探测恢复 (交易API恢复更快)
    _breaker = get_breaker("alpaca", threshold=5, recovery_timeout=30)

    def __init__(self):
        self._client = None
        self._initialized = False
        # 懒初始化: import 时不连接，首次使用时才初始化

    def _ensure_init(self):
        """懒初始化: 首次调用时才创建 TradingClient"""
        if self._initialized:
            return
        self._initialized = True  # 标记已尝试初始化
        if not config.has_alpaca_keys():
            logger.warning("ALPACA API Key 未配置, 交易功能不可用")
            return
        self._init_client()

    def _init_client(self):
        """初始化 Alpaca TradingClient"""
        try:
            from alpaca.trading.client import TradingClient
            paper = config.is_paper_trading()
            self._client = TradingClient(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
                paper=paper,
            )
            mode = "Paper" if paper else "Live"
            logger.info(f"Alpaca 客户端初始化成功 ({mode} mode, "
                        f"key: ***{config.ALPACA_API_KEY[-4:]})")
        except ImportError:
            logger.error("alpaca-py 包未安装, 请运行: pip install alpaca-py")
        except Exception as e:
            logger.error(f"Alpaca 客户端初始化失败: {e}")

    @property
    def available(self) -> bool:
        self._ensure_init()
        return self._client is not None

    @property
    def client(self):
        self._ensure_init()
        if not self.available:
            raise RuntimeError("Alpaca 客户端未初始化")
        return self._client

    # ─── 账户 ───

    def get_account(self) -> Optional[Account]:
        """获取账户信息"""
        if not self.available:
            return None

        try:
            @retry(max_retries=3, base_delay=0.5, max_delay=5)
            def _get():
                return self._breaker.call(self.client.get_account)

            acct = _get()
            # daytrade_count (注意: alpaca-py 属性名是 daytrade_count)
            dtc = getattr(acct, "daytrade_count", None) or getattr(acct, "day_trade_count", None) or 0
            return Account(
                equity=float(acct.equity),
                cash=float(acct.cash),
                buying_power=float(acct.buying_power),
                day_trade_count=int(dtc),
                pattern_day_trader=bool(acct.pattern_day_trader),
                trading_mode=TradingMode.PAPER if config.is_paper_trading() else TradingMode.LIVE,
                last_equity=float(acct.last_equity) if acct.last_equity else 0.0,
            )
        except CircuitBreakerOpenError:
            logger.warning("Alpaca 断路器熔断中, 跳过账户查询")
            return None
        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
            return None

    # ─── 订单 ───

    def submit_market_order(
        self, symbol: str, qty: float, side: str,
    ) -> Optional[Order]:
        """
        提交市价单

        Args:
            symbol: 股票代码
            qty: 数量
            side: "buy" | "sell"
        """
        if not self.available:
            return None

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )

            @retry(max_retries=2, base_delay=0.5, max_delay=3)
            def _submit():
                return self._breaker.call(self.client.submit_order, order_req)

            result = _submit()
            order = Order(
                order_id=str(result.id),
                alpaca_order_id=str(result.id),
                symbol=symbol,
                side=Direction.BUY if side == "buy" else Direction.SELL,
                qty=qty,
                order_type=OrderType.MARKET,
                status=self._map_order_status(str(result.status)),
                created_at=datetime.now(),
            )
            logger.info(f"市价单已提交: {side.upper()} {qty} {symbol} -> ID: {result.id}")
            return order
        except CircuitBreakerOpenError:
            logger.warning(f"Alpaca 断路器熔断中, 跳过下单 [{symbol}]")
            return None
        except Exception as e:
            logger.error(f"提交市价单失败 [{symbol}]: {e}")
            return None

    def submit_limit_order(
        self, symbol: str, qty: float, side: str, limit_price: float,
    ) -> Optional[Order]:
        """提交限价单"""
        if not self.available:
            return None

        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
            result = self.client.submit_order(order_req)

            order = Order(
                order_id=str(result.id),
                alpaca_order_id=str(result.id),
                symbol=symbol,
                side=Direction.BUY if side == "buy" else Direction.SELL,
                qty=qty,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
                status=self._map_order_status(str(result.status)),
                created_at=datetime.now(),
            )
            logger.info(f"限价单已提交: {side.upper()} {qty} {symbol} @ ${limit_price:.2f}")
            return order
        except Exception as e:
            logger.error(f"提交限价单失败 [{symbol}]: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        if not self.available:
            return False

        try:
            self.client.cancel_order_by_id(order_id)
            logger.info(f"订单已撤销: {order_id}")
            return True
        except Exception as e:
            logger.error(f"撤单失败 [{order_id}]: {e}")
            return False

    def get_orders(self, status: str = "all") -> list[Order]:
        """查询订单"""
        if not self.available:
            return []

        try:
            from alpaca.trading.enums import QueryOrderStatus
            status_map = {
                "all": QueryOrderStatus.ALL,
                "open": QueryOrderStatus.OPEN,
                "closed": QueryOrderStatus.CLOSED,
            }
            qstatus = status_map.get(status, QueryOrderStatus.ALL)
            orders = self.client.get_orders(status=qstatus)

            result = []
            for o in orders:
                filled_price = float(o.filled_avg_price) if o.filled_avg_price else None
                filled_at = o.filled_at if o.filled_at else None
                result.append(Order(
                    order_id=str(o.id),
                    alpaca_order_id=str(o.id),
                    symbol=o.symbol,
                    side=Direction.BUY if str(o.side) == "OrderSide.BUY" else Direction.SELL,
                    qty=float(o.qty),
                    order_type=OrderType.MARKET if "market" in str(o.order_type).lower() else OrderType.LIMIT,
                    limit_price=float(o.limit_price) if o.limit_price else None,
                    status=self._map_order_status(str(o.status)),
                    filled_price=filled_price,
                    created_at=o.created_at if o.created_at else datetime.now(),
                    filled_at=filled_at,
                ))
            return result
        except Exception as e:
            logger.error(f"查询订单失败: {e}")
            return []

    # ─── 持仓 ───

    def get_positions(self) -> list[Position]:
        """查询所有持仓"""
        if not self.available:
            return []

        try:
            @retry(max_retries=3, base_delay=0.5, max_delay=5)
            def _get():
                return self._breaker.call(self.client.get_all_positions)

            positions = _get()
            account = self.get_account()
            total_equity = account.equity if account else 100000.0

            result = []
            for p in positions:
                market_value = float(p.market_value)
                unrealized_pnl = float(p.unrealized_pl)
                avg_cost = float(p.avg_entry_price)
                qty = float(p.qty)
                # 可用数量 = 总持仓 - 挂单锁定量 (alpaca-py 无直接字段, 通过 qty_available 或 quantity_available 属性获取)
                qty_available = qty  # 默认全部可用
                for attr in ("qty_available", "quantity_available"):
                    val = getattr(p, attr, None)
                    if val is not None:
                        qty_available = float(val)
                        break
                else:
                    # 部分版本字段名不同, 尝试从 asset 中推断
                    qty_available = qty
                unrealized_pnl_pct = float(p.unrealized_plpc) if p.unrealized_plpc else 0.0

                result.append(Position(
                    symbol=p.symbol,
                    qty=qty,
                    qty_available=qty_available,
                    avg_cost=avg_cost,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    weight=market_value / total_equity if total_equity > 0 else 0.0,
                ))
            return result
        except Exception as e:
            logger.error(f"查询持仓失败: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        """查询单只股票持仓"""
        positions = self.get_positions()
        for p in positions:
            if p.symbol == symbol:
                return p
        return None

    def liquidate(self, symbol: str) -> bool:
        """清仓某只股票"""
        if not self.available:
            return False

        try:
            @retry(max_retries=2, base_delay=0.5, max_delay=3)
            def _close():
                return self._breaker.call(self.client.close_position, symbol)

            _close()
            logger.info(f"已清仓: {symbol}")
            return True
        except CircuitBreakerOpenError:
            logger.warning(f"Alpaca 断路器熔断中, 跳过清仓 [{symbol}]")
            return False
        except Exception as e:
            logger.error(f"清仓失败 [{symbol}]: {e}")
            return False

    # ─── 资产 ───

    def get_tradable_assets(self) -> list[dict]:
        """查询可交易资产列表"""
        if not self.available:
            return []

        try:
            from alpaca.trading.requests import GetAssetsRequest
            from alpaca.trading.enums import AssetClass

            req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status="active")
            assets = self.client.get_all_assets(req)
            tradable = [
                {"symbol": a.symbol, "name": a.name, "exchange": a.exchange}
                for a in assets if a.tradable
            ]
            logger.info(f"可交易美股: {len(tradable)} 只")
            return tradable
        except Exception as e:
            logger.error(f"查询可交易资产失败: {e}")
            return []

    def get_crypto_tradable_assets(self) -> list[dict]:
        """
        查询可交易加密货币列表 (Phase 8 新增)

        Returns:
            list[dict]: [{"symbol": "BTC/USD", "name": "Bitcoin"}, ...]
        """
        if not self.available:
            return []

        try:
            from alpaca.trading.requests import GetAssetsRequest
            from alpaca.trading.enums import AssetClass

            req = GetAssetsRequest(asset_class=AssetClass.CRYPTO, status="active")
            assets = self.client.get_all_assets(req)
            tradable = [
                {"symbol": a.symbol, "name": a.name}
                for a in assets if a.tradable
            ]
            logger.info(f"可交易加密货币: {len(tradable)} 个")
            return tradable
        except ImportError:
            logger.warning("alpaca-py 不支持 CRYPTO asset class")
            return []
        except Exception as e:
            logger.error(f"查询可交易加密货币失败: {e}")
            return []

    # Alpaca 最小订单金额 (美元), 低于此金额的订单会被拒绝
    MIN_ORDER_AMOUNT = 10.0

    def submit_crypto_order(
        self, symbol: str, qty: float, side: str, price: float = 0.0,
    ) -> Optional[Order]:
        """
        提交加密货币订单 (Phase 8 新增)

        Alpaca 对加密货币和股票使用相同的下单接口,
        只需确保 symbol 格式正确 (如 "BTC/USD")

        Args:
            symbol: 交易对, 如 "BTC/USD"
            qty: 数量
            side: "buy" | "sell"
            price: 最新价格, 用于提交前校验最小订单金额

        Returns:
            Order 对象或 None
        """
        if not self.available:
            return None

        # 安全兜底: 校验最小订单金额 (Alpaca 要求 cost basis >= $10)
        if price > 0 and qty * price < self.MIN_ORDER_AMOUNT:
            logger.warning(
                f"加密货币订单金额不足: {qty:.6f} × ${price:.2f} = ${qty*price:.2f} "
                f"< ${self.MIN_ORDER_AMOUNT} (最小限额), 跳过 [{symbol}]"
            )
            return None

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            # 加密货币订单使用 GTC (Good Till Cancelled)
            # 因为 24/7 交易, 不受盘中限制
            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
            )

            @retry(max_retries=2, base_delay=0.5, max_delay=3)
            def _submit():
                return self._breaker.call(self.client.submit_order, order_req)

            result = _submit()
            order = Order(
                order_id=str(result.id),
                alpaca_order_id=str(result.id),
                symbol=symbol,
                side=Direction.BUY if side == "buy" else Direction.SELL,
                qty=qty,
                order_type=OrderType.MARKET,
                status=self._map_order_status(str(result.status)),
                asset_type=AssetType.CRYPTO,  # 标记为加密货币订单
                created_at=datetime.now(),
            )
            logger.info(f"加密货币市价单已提交: {side.upper()} {qty} {symbol} -> ID: {result.id}")
            return order
        except CircuitBreakerOpenError:
            logger.warning(f"Alpaca 断路器熔断中, 跳过加密货币下单 [{symbol}]")
            return None
        except Exception as e:
            logger.error(f"提交加密货币订单失败 [{symbol}]: {e}")
            return None

    def submit_crypto_limit_order(
        self, symbol: str, qty: float, side: str, limit_price: float,
    ) -> Optional[Order]:
        """
        提交加密货币限价单 (Phase 8 新增)

        Args:
            symbol: 交易对, 如 "BTC/USD"
            qty: 数量
            side: "buy" | "sell"
            limit_price: 限价
        """
        if not self.available:
            return None

        # 安全兜底: 校验最小订单金额 (Alpaca 要求 cost basis >= )
        cost_basis = qty * limit_price
        if cost_basis < self.MIN_ORDER_AMOUNT:
            logger.warning(
                f"加密货币限价单金额不足: {qty:.6f} ×  =  "
                f"<  (最小限额), 跳过 [{symbol}]"
            )
            return None

        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                limit_price=limit_price,
            )
            result = self.client.submit_order(order_req)

            order = Order(
                order_id=str(result.id),
                alpaca_order_id=str(result.id),
                symbol=symbol,
                side=Direction.BUY if side == "buy" else Direction.SELL,
                qty=qty,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
                status=self._map_order_status(str(result.status)),
                asset_type=AssetType.CRYPTO,
                created_at=datetime.now(),
            )
            logger.info(f"加密货币限价单已提交: {side.upper()} {qty} {symbol} @ ${limit_price:.2f}")
            return order
        except Exception as e:
            logger.error(f"提交加密货币限价单失败 [{symbol}]: {e}")
            return None

    def get_crypto_positions(self) -> list[Position]:
        """
        查询加密货币持仓 (Phase 8 新增)

        Alpaca 的 get_all_positions 已包含加密货币持仓,
        此方法通过 symbol 格式 (包含 "/") 来过滤

        Returns:
            list[Position]: 加密货币持仓列表
        """
        positions = self.get_positions()
        # 加密货币的 symbol 通常包含 "/", 如 "BTC/USD"
        crypto_positions = [p for p in positions if "/" in p.symbol]
        # 更新 asset_type
        for p in crypto_positions:
            p.asset_type = AssetType.CRYPTO
        return crypto_positions

    def get_stock_positions(self) -> list[Position]:
        """
        查询股票持仓 (排除加密货币) (Phase 8 新增)

        Returns:
            list[Position]: 股票持仓列表
        """
        positions = self.get_positions()
        return [p for p in positions if "/" not in p.symbol]

    def get_crypto_account(self) -> Optional[Account]:
        """
        获取加密货币账户信息 (Phase 8 新增)

        Alpaca 的账户是统一的, 股票和加密货币共享同一个账户。
        此方法返回与 get_account() 相同的账户信息。

        Returns:
            Account 对象或 None
        """
        return self.get_account()

    # ─── 内部方法 ───

    def _map_order_status(self, raw_status: str) -> OrderStatus:
        """将 Alpaca 状态映射为内部 OrderStatus"""
        status_lower = raw_status.lower()
        # 去掉 OrderStatus. 前缀
        if "orderstatus." in status_lower:
            status_lower = status_lower.split(".")[-1]

        mapping = {
            "new": OrderStatus.PENDING,
            "pending_new": OrderStatus.PENDING,
            "accepted": OrderStatus.PENDING,
            "pending_review": OrderStatus.PENDING,
            "filled": OrderStatus.FILLED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "done_for_day": OrderStatus.PENDING,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.REJECTED,
        }
        return mapping.get(status_lower, OrderStatus.PENDING)

    def health_check(self) -> bool:
        """
        连接健康检查

        Returns:
            True 表示连接正常
        """
        if not self.available:
            return False
        try:
            acct = self._breaker.call(self.client.get_account)
            return acct is not None
        except Exception:
            return False

    def is_healthy(self) -> bool:
        """断路器是否正常 (未熔断)"""
        return self._breaker.state != "open"


# 全局实例
alpaca_client = AlpacaClient()
