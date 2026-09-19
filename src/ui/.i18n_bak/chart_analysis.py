"""
专业K线行情分析页面
=====================================================
v0.7.0 新增

功能:
- 全周期K线图 (1分/5分/15分/30分/60分/日/周/月)
- 蜡烛图 + 成交量副图
- 技术指标: MA均线系统、MACD、KDJ、BOLL、RSI
- 多股票快速切换 (关注列表 + 搜索)
- 时间范围快捷选择
- 十字光标、缩放、平移交互
- 股票基本面信息展示

数据源:
- 日K/周K/月K: AKShare (主) / yfinance (备)
- 日内分时: yfinance (1m/5m/15m/30m/60m)
- 实时报价: RealtimeQuoteFetcher (多源降级)
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ═══════════════════════════════════════════════════════
# 延迟导入
# ═══════════════════════════════════════════════════════

def _get_config():
    """延迟获取 config"""
    from src.utils.config import config
    return config


def _get_realtime_fetcher():
    """延迟获取 realtime_fetcher"""
    from src.data.realtime_fetcher import realtime_fetcher
    return realtime_fetcher


def _get_finnhub_client():
    """延迟获取 finnhub_client"""
    from src.data.finnhub_client import finnhub_client
    return finnhub_client


# ═══════════════════════════════════════════════════════
# 用户关注列表管理 (持久化到本地 JSON, 与智能选股页面共享)
# ═══════════════════════════════════════════════════════

def _get_watchlist_path():
    """获取用户关注列表文件路径"""
    try:
        cfg = _get_config()
        return Path(cfg.DATA_DIR) / "user_watchlist.json"
    except Exception:
        return Path("data") / "user_watchlist.json"


def _load_user_watchlist():
    """从本地 JSON 加载用户关注列表"""
    import json
    filepath = _get_watchlist_path()
    try:
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("watchlist", [])
    except Exception:
        pass
    return []


def _save_user_watchlist(watchlist):
    """保存用户关注列表到本地 JSON"""
    import json
    filepath = _get_watchlist_path()
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"watchlist": watchlist}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"关注列表保存失败: {e}")


# ═══════════════════════════════════════════════════════
# 数据获取层
# ═══════════════════════════════════════════════════════

# 周期配置: (显示名, yfinance interval, akshare适用, 描述)
PERIOD_CONFIG = {
    "1分": ("1m", False, "日内1分钟K线 (最近7天)"),
    "5分": ("5m", False, "日内5分钟K线 (最近60天)"),
    "15分": ("15m", False, "日内15分钟K线 (最近60天)"),
    "30分": ("30m", False, "日内30分钟K线 (最近60天)"),
    "60分": ("60m", False, "日内60分钟K线 (最近60天)"),
    "日K": ("1d", True, "日线K线 (前复权)"),
    "周K": ("1wk", True, "周线K线"),
    "月K": ("1mo", True, "月线K线"),
}

# 时间范围快捷选项
TIME_RANGES = {
    "1周": 7,
    "1月": 30,
    "3月": 90,
    "半年": 180,
    "1年": 365,
    "3年": 1095,
    "5年": 1825,
    "全部": None,  # None = 不限制
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_kline_data(symbol: str, period: str, days_back: int = None) -> pd.DataFrame:
    """
    获取K线数据 (带 Streamlit 缓存, 5分钟刷新)

    Args:
        symbol: 股票代码, 如 "AAPL"
        period: 周期名称, 如 "日K", "5分"
        days_back: 回溯天数 (None=全部)

    Returns:
        DataFrame: datetime, open, high, low, close, volume
    """
    interval, use_akshare, _desc = PERIOD_CONFIG.get(period, ("1d", True, ""))

    df = None

    # 日K/周K/月K: 优先 AKShare (前复权日线数据更完整)
    if use_akshare:
        df = _fetch_from_akshare(symbol, period, days_back)

    # 日内分时 或 AKShare失败: 使用 yfinance
    if df is None or len(df) == 0:
        df = _fetch_from_yfinance(symbol, interval, days_back)

    if df is None or len(df) == 0:
        return pd.DataFrame()

    # 确保列名统一
    required = ["datetime", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    # 数值类型转换
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 去除NaN
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    return df


def _fetch_from_akshare(symbol: str, period: str, days_back: int = None) -> pd.DataFrame:
    """从 AKShare 获取日/周/月K线数据"""
    try:
        from src.data.akshare_collector import akshare_collector

        start = None
        end = None
        if days_back is not None:
            start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        df = akshare_collector.get_us_stock_daily(
            symbol=symbol, start=start, end=end, adjust="qfq", use_cache=True
        )

        if df is None or len(df) == 0:
            return pd.DataFrame()

        # AKShare 返回列: date, open, high, low, close, volume, ...
        df = df.rename(columns={"date": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])

        # 周K/月K: 从日线重采样
        if period == "周K":
            df = _resample_kline(df, "W")
        elif period == "月K":
            df = _resample_kline(df, "ME")

        return df[["datetime", "open", "high", "low", "close", "volume"]]

    except Exception as e:
        st.warning(f"AKShare 数据获取失败: {e}")
        return pd.DataFrame()


def _fetch_from_yfinance(symbol: str, interval: str, days_back: int = None) -> pd.DataFrame:
    """从 yfinance 获取K线数据 (日内分时 + 日/周/月线)"""
    try:
        import yfinance as yf

        # 根据 interval 和 days_back 确定下载范围
        if days_back is None:
            period_str = "max"
        else:
            # yfinance 的 period 参数有上限, 超过则用 start/end
            max_period_map = {
                "1m": "7d", "5m": "60d", "15m": "60d",
                "30m": "60d", "60m": "730d",
                "1d": "max", "1wk": "max", "1mo": "max",
            }
            max_period = max_period_map.get(interval, "max")
            # 如果请求的天数超过 interval 的最大范围, 用 max
            period_limits = {
                "1m": 7, "5m": 60, "15m": 60, "30m": 60, "60m": 730,
            }
            limit = period_limits.get(interval, 999999)
            if days_back <= limit:
                period_str = f"{days_back}d"
            else:
                period_str = max_period

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period_str, interval=interval)

        if df is None or len(df) == 0:
            return pd.DataFrame()

        # yfinance 返回的列: Open, High, Low, Close, Volume, ...
        df = df.reset_index()
        # 列名标准化
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ("datetime", "date", "index"):
                col_map[col] = "datetime"
            elif col_lower == "open":
                col_map[col] = "open"
            elif col_lower == "high":
                col_map[col] = "high"
            elif col_lower == "low":
                col_map[col] = "low"
            elif col_lower == "close":
                col_map[col] = "close"
            elif col_lower == "volume":
                col_map[col] = "volume"

        df = df.rename(columns=col_map)

        # 确保 datetime 列存在
        if "datetime" not in df.columns:
            # yfinance 的 index 列名可能是 "Date" 或其他
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "datetime"})

        df["datetime"] = pd.to_datetime(df["datetime"])

        return df[["datetime", "open", "high", "low", "close", "volume"]]

    except Exception as e:
        st.warning(f"yfinance 数据获取失败: {e}")
        return pd.DataFrame()


def _resample_kline(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """将日线数据重采样为周线/月线"""
    df = df.set_index("datetime")
    resampled = df.resample(freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna().reset_index()
    return resampled


# ═══════════════════════════════════════════════════════
# 技术指标计算
# ═══════════════════════════════════════════════════════

def calc_ma(df: pd.DataFrame, periods: list = [5, 10, 20, 60, 120, 250]) -> pd.DataFrame:
    """计算移动平均线"""
    for p in periods:
        if len(df) >= p:
            df[f"MA{p}"] = df["close"].rolling(window=p).mean()
    return df


def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """计算 MACD 指标"""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["DEA"] = df["DIF"].ewm(span=signal, adjust=False).mean()
    df["MACD"] = (df["DIF"] - df["DEA"]) * 2  # 柱状图
    return df


def calc_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """计算 KDJ 指标"""
    low_n = df["low"].rolling(window=n).min()
    high_n = df["high"].rolling(window=n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)

    # K = SMA(RSV, m1), D = SMA(K, m2), J = 3K - 2D
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d

    df["K"] = k
    df["D"] = d
    df["J"] = j
    return df


def calc_boll(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """计算布林带"""
    mid = df["close"].rolling(window=n).mean()
    std = df["close"].rolling(window=n).std()
    df["BOLL_MID"] = mid
    df["BOLL_UP"] = mid + k * std
    df["BOLL_LOW"] = mid - k * std
    return df


def calc_rsi(df: pd.DataFrame, periods: list = [6, 12, 24]) -> pd.DataFrame:
    """计算 RSI 指标"""
    delta = df["close"].diff()
    for p in periods:
        gain = delta.where(delta > 0, 0).rolling(window=p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f"RSI{p}"] = (100 - (100 / (1 + rs))).fillna(50)
    return df


# ═══════════════════════════════════════════════════════
# 图表渲染层
# ═══════════════════════════════════════════════════════

# 美股标准配色: 红涨绿跌
COLOR_UP = "#ef5350"       # 涨-红色
COLOR_DOWN = "#26a69a"     # 跌-绿色
COLOR_VOLUME_UP = "#ef5350"
COLOR_VOLUME_DOWN = "#26a69a"

# 均线颜色
MA_COLORS = {
    "MA5": "#ffd700",    # 金色
    "MA10": "#ff9800",   # 橙色
    "MA20": "#2196f3",   # 蓝色
    "MA60": "#9c27b0",   # 紫色
    "MA120": "#795548",  # 棕色
    "MA250": "#607d8b",  # 灰蓝色
}


def render_kline_chart(
    df: pd.DataFrame,
    show_ma: list = None,
    show_volume: bool = True,
    show_macd: bool = False,
    show_kdj: bool = False,
    show_boll: bool = False,
    show_rsi: bool = False,
    symbol: str = "",
    period: str = "",
) -> go.Figure:
    """
    渲染专业K线图

    Args:
        df: K线数据 (已计算指标)
        show_ma: 要显示的均线列表, 如 ["MA5", "MA20"]
        show_volume: 是否显示成交量
        show_macd: 是否显示MACD副图
        show_kdj: 是否显示KDJ副图
        show_boll: 是否显示布林带
        show_rsi: 是否显示RSI副图
        symbol: 股票代码
        period: 周期

    Returns:
        plotly Figure
    """
    if df is None or len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="暂无数据", showarrow=False, font_size=20)
        return fig

    # 计算副图数量
    subplot_count = 1  # 主图
    if show_volume:
        subplot_count += 1
    if show_macd:
        subplot_count += 1
    if show_kdj:
        subplot_count += 1
    if show_rsi:
        subplot_count += 1

    # 主图高度占比大, 副图较小
    heights = [0.5]  # 主图
    if show_volume:
        heights.append(0.12)
    if show_macd:
        heights.append(0.13)
    if show_kdj:
        heights.append(0.12)
    if show_rsi:
        heights.append(0.13)

    # 归一化高度
    total_h = sum(heights)
    heights = [h / total_h for h in heights]

    # 创建子图
    subplot_titles = [f"📊 {symbol} · {period} K线"]
    if show_volume:
        subplot_titles.append("成交量")
    if show_macd:
        subplot_titles.append("MACD")
    if show_kdj:
        subplot_titles.append("KDJ")
    if show_rsi:
        subplot_titles.append("RSI")

    # 构建子图规格
    specs = [[{"secondary_y": False}]]
    for i in range(subplot_count - 1):
        specs.append([{"secondary_y": False}])

    fig = make_subplots(
        rows=subplot_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
        subplot_titles=subplot_titles,
    )

    # ─── 主图: 蜡烛图 ───
    # 美股配色: 涨红跌绿
    colors_candle = [COLOR_UP if c >= o else COLOR_DOWN for c, o in zip(df["close"], df["open"])]

    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing_line_color=COLOR_UP,
            decreasing_line_color=COLOR_DOWN,
            increasing_fillcolor=COLOR_UP,
            decreasing_fillcolor=COLOR_DOWN,
            whiskerwidth=0.5,
        ),
        row=1, col=1,
    )

    # ─── 均线 ───
    if show_ma:
        for ma_name in show_ma:
            if ma_name in df.columns:
                color = MA_COLORS.get(ma_name, "#888888")
                fig.add_trace(
                    go.Scatter(
                        x=df["datetime"],
                        y=df[ma_name],
                        mode="lines",
                        name=ma_name,
                        line=dict(color=color, width=1.2),
                        opacity=0.85,
                    ),
                    row=1, col=1,
                )

    # ─── 布林带 ───
    if show_boll and "BOLL_UP" in df.columns:
        # 上轨
        fig.add_trace(
            go.Scatter(
                x=df["datetime"], y=df["BOLL_UP"],
                mode="lines", name="BOLL上轨",
                line=dict(color="#ab47bc", width=1, dash="dash"),
                opacity=0.6,
            ),
            row=1, col=1,
        )
        # 中轨
        fig.add_trace(
            go.Scatter(
                x=df["datetime"], y=df["BOLL_MID"],
                mode="lines", name="BOLL中轨",
                line=dict(color="#ab47bc", width=1),
                opacity=0.6,
            ),
            row=1, col=1,
        )
        # 下轨 + 填充
        fig.add_trace(
            go.Scatter(
                x=df["datetime"], y=df["BOLL_LOW"],
                mode="lines", name="BOLL下轨",
                line=dict(color="#ab47bc", width=1, dash="dash"),
                opacity=0.6,
                fill="tonexty",
                fillcolor="rgba(171, 71, 188, 0.05)",
            ),
            row=1, col=1,
        )

    # ─── 成交量 ───
    current_row = 2
    if show_volume:
        vol_colors = [COLOR_VOLUME_UP if c >= o else COLOR_VOLUME_DOWN
                      for c, o in zip(df["close"], df["open"])]
        fig.add_trace(
            go.Bar(
                x=df["datetime"],
                y=df["volume"],
                name="成交量",
                marker_color=vol_colors,
                opacity=0.7,
            ),
            row=current_row, col=1,
        )
        current_row += 1

    # ─── MACD ───
    if show_macd and "DIF" in df.columns:
        # DIF 线
        fig.add_trace(
            go.Scatter(
                x=df["datetime"], y=df["DIF"],
                mode="lines", name="DIF",
                line=dict(color="#2196f3", width=1.2),
            ),
            row=current_row, col=1,
        )
        # DEA 线
        fig.add_trace(
            go.Scatter(
                x=df["datetime"], y=df["DEA"],
                mode="lines", name="DEA",
                line=dict(color="#ff9800", width=1.2),
            ),
            row=current_row, col=1,
        )
        # MACD 柱状图
        macd_colors = [COLOR_UP if v >= 0 else COLOR_DOWN for v in df["MACD"]]
        fig.add_trace(
            go.Bar(
                x=df["datetime"], y=df["MACD"],
                name="MACD柱",
                marker_color=macd_colors,
                opacity=0.6,
            ),
            row=current_row, col=1,
        )
        current_row += 1

    # ─── KDJ ───
    if show_kdj and "K" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["datetime"], y=df["K"],
                mode="lines", name="K",
                line=dict(color="#2196f3", width=1.2),
            ),
            row=current_row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["datetime"], y=df["D"],
                mode="lines", name="D",
                line=dict(color="#ff9800", width=1.2),
            ),
            row=current_row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["datetime"], y=df["J"],
                mode="lines", name="J",
                line=dict(color="#9c27b0", width=1.2),
            ),
            row=current_row, col=1,
        )
        current_row += 1

    # ─── RSI ───
    if show_rsi and "RSI6" in df.columns:
        for col_name, color in [("RSI6", "#2196f3"), ("RSI12", "#ff9800"), ("RSI24", "#9c27b0")]:
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["datetime"], y=df[col_name],
                        mode="lines", name=col_name,
                        line=dict(color=color, width=1.2),
                    ),
                    row=current_row, col=1,
                )
        # 超买超卖参考线
        fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.05,
                      row=current_row, col=1)
        fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.05,
                      row=current_row, col=1)
        current_row += 1

    # ─── 全局布局配置 ───
    fig.update_layout(
        template="plotly_dark",
        height=800,
        xaxis_rangeslider_visible=False,  # 关闭范围选择器 (专业K线不用)
        dragmode="pan",  # 默认拖拽模式: 平移
        margin=dict(l=50, r=30, t=40, b=30),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        font=dict(size=11),
    )

    # X轴: 隐藏非交易时间间隙 (仅对日内分时有效)
    for i in range(1, subplot_count + 1):
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.15)",
            rangeslider_visible=False,
            row=i, col=1,
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.15)",
            row=i, col=1,
        )

    # 十字光标配置
    fig.update_layout(hovermode="x unified")

    return fig


# ═══════════════════════════════════════════════════════
# 股票信息面板
# ═══════════════════════════════════════════════════════

@st.cache_data(ttl=60, show_spinner=False)
def fetch_realtime_quote(symbol: str) -> dict:
    """获取实时报价 (缓存60秒)"""
    try:
        rf = _get_realtime_fetcher()
        quote = rf.get_realtime_quote(symbol)
        if quote:
            return quote
    except Exception:
        pass
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_profile(symbol: str) -> dict:
    """获取公司概况 (缓存1小时)"""
    try:
        fc = _get_finnhub_client()
        if fc.available:
            return fc.get_company_profile(symbol)
    except Exception:
        pass
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_financials(symbol: str) -> dict:
    """获取基本面财务指标 (缓存1小时)"""
    try:
        fc = _get_finnhub_client()
        if fc.available:
            return fc.get_financials(symbol)
    except Exception:
        pass
    return {}


def render_stock_info(symbol: str, quote: dict, profile: dict, financials: dict):
    """渲染股票信息面板"""
    # ── 实时报价卡片 ──
    if quote:
        price = quote.get("price", 0)
        change = quote.get("change", 0)
        change_pct = quote.get("change_pct", 0)
        color = "🟢" if change >= 0 else "🔴"

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("💰 最新价", f"${price:.2f}", f"{change:+.2f}")
        col2.metric("📊 涨跌幅", f"{change_pct:+.2f}%")
        col3.metric("🔵 开盘", f"${quote.get('open', 0):.2f}")
        col4.metric("⬆️ 最高", f"${quote.get('high', 0):.2f}")
        col5.metric("⬇️ 最低", f"${quote.get('low', 0):.2f}")

        col6, col7, col8, col9, col10 = st.columns(5)
        col6.metric("📋 昨收", f"${quote.get('prev_close', 0):.2f}")
        col7.metric("📈 成交量", f"{quote.get('volume', 0):,}")
        col8.metric("📡 数据源", quote.get("source", "-"))
        col9.metric("🕐 时间", quote.get("timestamp", "-")[:19] if quote.get("timestamp") else "-")

    st.divider()

    # ── 公司概况 ──
    if profile:
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.markdown(f"### {profile.get('name', symbol)} ({symbol})")
            # 公司简介描述 (完整展示, 过长则在 expander 中展开)
            if profile.get('description'):
                desc = profile['description']
                if len(desc) > 200:
                    st.caption(desc[:200] + "...")
                    with st.expander("📖 查看完整公司简介", expanded=False):
                        st.markdown(desc)
                else:
                    st.caption(desc)
            # 行业、国家、交易所等信息 (一行展示, 更紧凑)
            info_parts = []
            if profile.get('finnhubIndustry'):
                info_parts.append(f"🏷️ 行业: {profile['finnhubIndustry']}")
            if profile.get('country'):
                info_parts.append(f"🌍 国家: {profile['country']}")
            if profile.get('exchange'):
                info_parts.append(f"🏛️ 交易所: {profile['exchange']}")
            if profile.get('weburl'):
                info_parts.append(f"🔗 [官网]({profile['weburl']})")
            if profile.get('employeeTotal'):
                info_parts.append(f"👥 员工: {profile['employeeTotal']:,}")
            if info_parts:
                st.markdown("  ·  ".join(info_parts))

        with col_r:
            if profile.get('marketCapitalization'):
                mcap = profile['marketCapitalization']
                if mcap >= 1e12:
                    mcap_str = f"${mcap/1e12:.2f}T"
                elif mcap >= 1e9:
                    mcap_str = f"${mcap/1e9:.2f}B"
                elif mcap >= 1e6:
                    mcap_str = f"${mcap/1e6:.2f}M"
                else:
                    mcap_str = f"${mcap:,.0f}"
                st.metric("🏦 市值", mcap_str)
            if profile.get('shareOutstanding'):
                shares = profile['shareOutstanding']
                if shares >= 1e9:
                    st.metric("📊 流通股", f"{shares/1e9:.2f}B")
                elif shares >= 1e6:
                    st.metric("📊 流通股", f"{shares/1e6:.2f}M")
                else:
                    st.metric("📊 流通股", f"{shares:,.0f}")
            if profile.get('ipo'):
                st.metric("📅 IPO日期", profile['ipo'])
    else:
        # 无公司概况数据时显示提示
        st.info(f"💡 暂无 {symbol} 的公司概况数据 (需配置 Finnhub API Key)")

    # ── 基本面指标 ──
    if financials and financials.get('metric'):
        metric = financials['metric']
        st.markdown("#### 📋 关键财务指标")

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("P/E (TTM)", f"{metric.get('peNormalizedAnnual', metric.get('peTTM', 'N/A'))}")
        col2.metric("P/B", f"{metric.get('pbAnnual', 'N/A')}")
        col3.metric("ROE", f"{metric.get('roeTTM', 'N/A')}")
        col4.metric("营收增长", f"{metric.get('revenueGrowth5Y', 'N/A')}")
        col5.metric("毛利率", f"{metric.get('grossMarginTTM', 'N/A')}")
        col6.metric("Beta", f"{metric.get('beta', 'N/A')}")


# ═══════════════════════════════════════════════════════
# 主页面
# ═══════════════════════════════════════════════════════

def page_chart_analysis():
    """行情分析页面 - 专业K线图"""
    st.title("📈 行情分析")
    st.caption("全周期K线图 · 技术指标 · 实时行情 · 基本面数据")

    config = _get_config()

    # ─── 左侧控制栏 + 右侧图表 ───
    col_ctrl, col_chart = st.columns([1, 4])

    with col_ctrl:
        st.markdown("#### 🎛️ 控制面板")

        # 股票选择 - 合并 config 默认关注列表 + 用户持久化关注列表
        config_watchlist = config.WATCHLIST or config.DEFAULT_WATCHLIST
        user_watchlist = _load_user_watchlist()
        # 合并去重 (用户关注列表优先显示)
        symbol_options = list(dict.fromkeys(user_watchlist + list(config_watchlist)))

        # 手动输入股票代码
        manual_symbol = st.text_input(
            "🔍 输入股票代码",
            value="",
            placeholder="如: AAPL, TSLA, NVDA",
            key="kline_symbol_input",
        )

        # 快捷选择
        if manual_symbol.strip():
            selected_symbol = manual_symbol.strip().upper()
        else:
            selected_symbol = st.selectbox(
                "⭐ 关注列表",
                options=symbol_options,
                index=0,
                key="kline_symbol_select",
            )

        st.divider()

        # 周期选择
        period = st.selectbox(
            "⏱️ K线周期",
            options=list(PERIOD_CONFIG.keys()),
            index=5,  # 默认日K
            key="kline_period",
        )
        # 显示周期描述
        _, _, desc = PERIOD_CONFIG.get(period, ("", "", ""))
        st.caption(f"💡 {desc}")

        st.divider()

        # 时间范围
        time_range = st.selectbox(
            "📅 时间范围",
            options=list(TIME_RANGES.keys()),
            index=4,  # 默认1年
            key="kline_time_range",
        )

        st.divider()

        # 技术指标选择
        st.markdown("#### 📊 技术指标")

        # 均线系统
        ma_options = ["MA5", "MA10", "MA20", "MA60", "MA120", "MA250"]
        default_ma = ["MA5", "MA10", "MA20", "MA60"]
        selected_ma = st.multiselect(
            "均线系统",
            options=ma_options,
            default=default_ma,
            key="kline_ma",
        )

        # 布林带
        show_boll = st.checkbox("布林带 BOLL", value=False, key="kline_boll")

        # 副图指标
        show_volume = st.checkbox("成交量", value=True, key="kline_volume")
        show_macd = st.checkbox("MACD", value=True, key="kline_macd")
        show_kdj = st.checkbox("KDJ", value=False, key="kline_kdj")
        show_rsi = st.checkbox("RSI", value=False, key="kline_rsi")

        st.divider()

        # 刷新按钮
        if st.button("🔄 刷新数据", key="kline_refresh", width='stretch'):
            st.cache_data.clear()
            st.rerun()

    # ─── 右侧主区域 ───
    with col_chart:
        if not selected_symbol:
            st.warning("请选择或输入股票代码")
            return

        # 获取实时报价和公司信息
        with st.spinner("获取实时行情..."):
            quote = fetch_realtime_quote(selected_symbol)
            profile = fetch_company_profile(selected_symbol)
            financials = fetch_company_financials(selected_symbol)

        # 渲染股票信息面板
        render_stock_info(selected_symbol, quote, profile, financials)

        # ─── 关注管理 + 策略生成 ───
        st.markdown("#### ⚡ 快捷操作")
        col_watch, col_strat = st.columns([1, 2])

        with col_watch:
            # 关注/取消关注当前股票 (持久化到 user_watchlist.json)
            user_wl = _load_user_watchlist()
            is_watched = selected_symbol in user_wl
            if is_watched:
                if st.button("💔 取消关注", key=f"btn_unwatch_chart_{selected_symbol}", use_container_width=True):
                    user_wl.remove(selected_symbol)
                    _save_user_watchlist(user_wl)
                    st.success(f"已取消关注 {selected_symbol}")
                    st.rerun()
            else:
                if st.button("⭐ 关注该股票", key=f"btn_watch_chart_{selected_symbol}", use_container_width=True):
                    if selected_symbol not in user_wl:
                        user_wl.append(selected_symbol)
                        _save_user_watchlist(user_wl)
                    st.success(f"已关注 {selected_symbol}")
                    st.rerun()
            # 显示关注列表数量
            st.caption(f"📋 已关注 {len(user_wl)} 只股票")

        with col_strat:
            # 为当前股票生成交易策略
            with st.expander("📈 为该股票生成交易策略", expanded=False):
                default_name = f"行情策略_{selected_symbol}_{datetime.now().strftime('%Y%m%d')}"
                strategy_name = st.text_input(
                    "策略名称", value=default_name, key=f"chart_strat_name_{selected_symbol}"
                )

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    model_type = st.selectbox(
                        "模型类型", ["lightgbm", "lstm"], key=f"chart_strat_model_{selected_symbol}"
                    )
                with col_s2:
                    max_position = st.slider(
                        "最大仓位", min_value=0.05, max_value=0.50, value=0.30, step=0.05,
                        format="%.0f%%", key=f"chart_strat_pos_{selected_symbol}",
                    )

                # 可选: 使用策略模板
                use_template = st.checkbox(
                    "使用策略模板 (自动填充因子和参数)", value=False,
                    key=f"chart_strat_tmpl_chk_{selected_symbol}",
                )

                selected_template_id = None
                if use_template:
                    try:
                        from src.engine.strategy_templates import strategy_templates
                        templates = strategy_templates.list_templates()
                        if templates:
                            tmpl_options = {t["template_id"]: f"{t['name']} ({t['risk_level']})" for t in templates}
                            selected_template_id = st.selectbox(
                                "选择模板", options=list(tmpl_options.keys()),
                                format_func=lambda x: tmpl_options[x],
                                key=f"chart_strat_tmpl_{selected_symbol}",
                            )
                        else:
                            st.warning("暂无可用模板")
                    except Exception as e:
                        st.warning(f"模板加载失败: {e}")

                if st.button("🚀 创建策略", type="primary",
                             key=f"chart_btn_create_strat_{selected_symbol}", use_container_width=True):
                    try:
                        from src.engine.strategy_manager import strategy_manager
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        strategy_id = f"chart_{selected_symbol.lower()}_{timestamp}"

                        if use_template and selected_template_id:
                            from src.engine.strategy_templates import strategy_templates as stm
                            template = stm._templates.get(selected_template_id)
                            if template:
                                strategy = strategy_manager.create_strategy(
                                    strategy_id=strategy_id,
                                    name=strategy_name or default_name,
                                    model_type=template.model_type,
                                    factors=template.factors.copy(),
                                    stock_pool=[selected_symbol],
                                    max_position=max_position,
                                    hyper_params=template.hyper_params.copy(),
                                )
                                strategy.rebalance_freq = template.rebalance_freq
                                strategy_manager._save(strategy)
                            else:
                                st.error("模板不存在")
                        else:
                            strategy = strategy_manager.create_strategy(
                                strategy_id=strategy_id,
                                name=strategy_name or default_name,
                                model_type=model_type,
                                factors=[],
                                stock_pool=[selected_symbol],
                                max_position=max_position,
                            )

                        st.success(f"✅ 策略 '{strategy.name}' 创建成功! (ID: {strategy_id})")
                        st.info(f"💡 股票池: {selected_symbol} | 模型: {strategy.model_type} | 仓位: {max_position:.0%}")
                        st.caption("👉 前往「📈 策略管理」查看详情和回测")
                    except Exception as e:
                        st.error(f"创建失败: {e}")

        st.divider()

        # 获取K线数据
        days_back = TIME_RANGES.get(time_range, 365)
        with st.spinner(f"加载 {selected_symbol} {period} K线数据..."):
            df = fetch_kline_data(selected_symbol, period, days_back)

        if df is None or len(df) == 0:
            st.error(f"❌ 无法获取 {selected_symbol} 的K线数据，请检查股票代码或网络连接")
            st.info("💡 提示: 日内分时数据需要安装 yfinance (`pip install yfinance`)")
            return

        # 计算技术指标
        df = calc_ma(df, periods=[5, 10, 20, 60, 120, 250])
        if show_macd:
            df = calc_macd(df)
        if show_kdj:
            df = calc_kdj(df)
        if show_boll:
            df = calc_boll(df)
        if show_rsi:
            df = calc_rsi(df)

        # 数据统计
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("📊 K线条数", f"{len(df)}")
        col_s2.metric("📅 起始", df["datetime"].iloc[0].strftime("%Y-%m-%d %H:%M") if len(df) > 0 else "-")
        col_s3.metric("📅 结束", df["datetime"].iloc[-1].strftime("%Y-%m-%d %H:%M") if len(df) > 0 else "-")
        if len(df) > 0:
            period_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
            col_s4.metric("📈 区间收益", f"{period_return:+.2f}%")

        # 渲染K线图
        fig = render_kline_chart(
            df=df,
            show_ma=selected_ma,
            show_volume=show_volume,
            show_macd=show_macd,
            show_kdj=show_kdj,
            show_boll=show_boll,
            show_rsi=show_rsi,
            symbol=selected_symbol,
            period=period,
        )

        st.plotly_chart(fig, width='stretch', config={
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
            "scrollZoom": True,
            "displaylogo": False,
        })

        # ─── K线数据表格 (可折叠) ───
        with st.expander(f"📋 {selected_symbol} {period} 原始数据 (最近20条)", expanded=False):
            display_df = df.tail(20).copy()
            display_df["datetime"] = display_df["datetime"].dt.strftime("%Y-%m-%d %H:%M")
            display_df = display_df[["datetime", "open", "high", "low", "close", "volume"]]
            display_df.columns = ["时间", "开盘", "最高", "最低", "收盘", "成交量"]
            st.dataframe(display_df, width='stretch', hide_index=True)
