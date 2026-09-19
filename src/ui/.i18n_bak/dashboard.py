"""
Streamlit 可视化仪表盘 - Soloquant
页面:
1. 📊 总览首页  - 账户资产、持仓分布、今日盈亏、告警状态
2. 📈 行情分析  - 全周期K线图、技术指标、实时行情、基本面
3. 🎯 智能选股  - 自然语言选股、多维度评分
4. 📈 策略管理  - 策略列表、回测结果图表、评级展示、组合管理
5. 💰 交易记录  - 历史订单、执行状态、风控检查
6. 🔍 市场调研  - 最新调研报告、候选股票、新闻情绪
7. 🌐 宏观指标  - 宏观经济仪表盘、美债收益率曲线、宏观风险评分
8. 📊 期权数据  - 期权链、PCR比率、隐含波动率、到期日选择
9. 🧠 模型训练  - 本地模型训练、多模型竞赛、因子搜索、训练历史
10. ⚙️ 系统设置  - 配置查看、工作流手动触发、紧急止损、实盘迁移

启动:
  streamlit run src/ui/dashboard.py
  python main.py dashboard
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ═══════════════════════════════════════════════════════
# 延迟导入: 项目模块 (避免顶层 import 触发完整依赖链)
# ═══════════════════════════════════════════════════════

_config = None
_notifier = None
_agent_registry = None
_workflow_engine = None
_strategy_manager = None
_position_manager = None
_portfolio_monitor = None
_realtime_manager = None
_data_collector = None


def _get_config():
    """延迟获取 config (首次调用时导入)"""
    global _config
    if _config is None:
        from src.utils.config import config
        _config = config
    return _config


def _get_notifier():
    """延迟获取 notifier"""
    global _notifier
    if _notifier is None:
        from src.utils.notifier import notifier
        _notifier = notifier
    return _notifier


def _get_agent_registry():
    """延迟获取 agent_registry"""
    global _agent_registry
    if _agent_registry is None:
        from src.agents.registry import agent_registry
        _agent_registry = agent_registry
    return _agent_registry


def _get_workflow_engine():
    """延迟获取 workflow_engine"""
    global _workflow_engine
    if _workflow_engine is None:
        from src.workflow.engine import workflow_engine
        _workflow_engine = workflow_engine
    return _workflow_engine


def _get_strategy_manager():
    """延迟获取 strategy_manager"""
    global _strategy_manager
    if _strategy_manager is None:
        from src.engine.strategy_manager import strategy_manager
        _strategy_manager = strategy_manager
    return _strategy_manager


def _get_position_manager():
    """延迟获取 position_manager"""
    global _position_manager
    if _position_manager is None:
        from src.execution.position_manager import position_manager
        _position_manager = position_manager
    return _position_manager


def _get_portfolio_monitor():
    """延迟获取 portfolio_monitor"""
    global _portfolio_monitor
    if _portfolio_monitor is None:
        from src.execution.portfolio_monitor import portfolio_monitor
        _portfolio_monitor = portfolio_monitor
    return _portfolio_monitor


def _get_realtime_manager():
    """延迟获取 realtime_manager"""
    global _realtime_manager
    if _realtime_manager is None:
        from src.data.realtime_manager import realtime_manager
        _realtime_manager = realtime_manager
    return _realtime_manager


def _get_data_collector():
    """延迟获取 data_collector (统一数据采集器)"""
    global _data_collector
    if _data_collector is None:
        from src.data.collector import data_collector
        _data_collector = data_collector
    return _data_collector


# ═══════════════════════════════════════════════════════
# 数据源切换 (本地直连 / 云端API)
# ═══════════════════════════════════════════════════════

def _get_cloud_client():
    """获取云端数据客户端 (单例)"""
    from src.data.cloud_client import cloud_client
    return cloud_client


def _init_data_source():
    """初始化数据源模式 (从 session_state 读取)"""
    if "data_source_mode" not in st.session_state:
        # 默认本地模式，可从环境变量覆盖
        env_mode = os.environ.get("DATA_SOURCE_MODE", "local")
        st.session_state["data_source_mode"] = env_mode
    if "cloud_api_url" not in st.session_state:
        # 从配置读取云端 API 地址
        try:
            cfg = _get_config()
            api_url = cfg.CLOUD_API_URL or ""
            api_port = cfg.CLOUD_API_PORT or "8001"
            if api_url:
                st.session_state["cloud_api_url"] = api_url
            else:
                st.session_state["cloud_api_url"] = ""
        except Exception:
            st.session_state["cloud_api_url"] = ""

    # 应用到 cloud_client
    cc = _get_cloud_client()
    cc.set_mode(
        st.session_state["data_source_mode"],
        st.session_state.get("cloud_api_url", "")
    )


def _is_cloud_mode() -> bool:
    """当前是否为云端API模式"""
    return st.session_state.get("data_source_mode") == "cloud"


# ─── 页面配置 ───
st.set_page_config(
    page_title="Soloquant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════
# 通用辅助函数
# ═══════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def get_account_info():
    """获取账户信息 (缓存30秒) - 支持本地/云端切换"""
    try:
        cc = _get_cloud_client()
        data = cc.get_overview()
        if data and data.get("account"):
            return data["account"]
        # 记录具体原因，方便排查
        if data is None:
            st.session_state["_account_error"] = "云端API返回空数据 (检查API地址配置)"
        elif not data.get("account"):
            err = data.get("positions", {})
            if isinstance(err, dict) and "error" in err:
                st.session_state["_account_error"] = f"账户获取失败: {err['error']}"
            else:
                st.session_state["_account_error"] = "账户数据为空"
    except Exception as e:
        st.session_state["_account_error"] = str(e)
    return None


@st.cache_data(ttl=30)
def get_positions_list():
    """获取持仓列表 (缓存30秒) - 支持本地/云端切换"""
    try:
        cc = _get_cloud_client()
        data = cc.get_overview()
        if data and data.get("positions"):
            return data["positions"]
        return {"error": "无法获取持仓"}
    except Exception:
        return {"error": "无法获取持仓"}


@st.cache_data(ttl=60)
def get_strategies():
    """获取策略列表 (缓存60秒) - 支持本地/云端切换"""
    try:
        cc = _get_cloud_client()
        return cc.get_strategies()
    except Exception:
        return []


@st.cache_data(ttl=30)
def get_notifications(limit=20):
    """获取通知历史 - 支持本地/云端切换"""
    try:
        cc = _get_cloud_client()
        return cc.get_notifications(limit)
    except Exception:
        return []


def get_workflow_names():
    """获取可用工作流 - 支持本地/云端切换"""
    try:
        cc = _get_cloud_client()
        return cc.get_workflows()
    except Exception:
        return []


# ─── 宏观指标 & 期权数据缓存函数 ───

@st.cache_data(ttl=3600 * 6)
def get_macro_dashboard_data():
    """获取宏观经济仪表盘数据 (缓存6小时) - 支持本地/云端切换"""
    try:
        cc = _get_cloud_client()
        return cc.get_macro()
    except Exception as e:
        st.error(f"获取宏观数据失败: {e}")
        return {}


@st.cache_data(ttl=3600 * 12)
def get_yield_curve_data():
    """获取美债收益率曲线数据 (缓存12小时) - 支持本地/云端切换"""
    try:
        cc = _get_cloud_client()
        return cc.get_yield_curve()
    except Exception as e:
        st.error(f"获取收益率曲线失败: {e}")
        return {}


@st.cache_data(ttl=300)
def get_options_chain_data(symbol: str, expiration_date: str = ""):
    """获取期权链数据 (缓存5分钟) - 支持本地/云端切换"""
    try:
        cc = _get_cloud_client()
        return cc.get_options_chain(symbol, expiration_date)
    except Exception as e:
        st.error(f"获取期权链失败: {e}")
        return {}



@st.cache_data(ttl=60)
def get_crypto_prices_with_change():
    """获取加密货币价格及24h涨跌幅 (缓存60秒) - 支持本地/云端切换"""
    try:
        cc = _get_cloud_client()
        return cc.get_crypto_prices()
    except Exception:
        return []

# ═══════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════

def _inject_sidebar_css():
    """注入侧边栏自定义样式 (自适应浅色/深色主题)"""
    st.markdown("""
    <style>
    /* ══════════════════════════════════════════
       侧边栏样式 - 双主题自适应
       通过 [data-theme="light"] / [data-theme="dark"]
       自动匹配 Streamlit 当前主题
    ══════════════════════════════════════════ */

    /* ── 侧边栏整体: 深色主题 ── */
    [data-theme="dark"] section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1929 0%, #1a2332 100%);
    }
    [data-theme="dark"] section[data-testid="stSidebar"] .stMarkdown {
        color: #c8d3e0;
    }

    /* ── 侧边栏整体: 浅色主题 ── */
    [data-theme="light"] section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
    }
    [data-theme="light"] section[data-testid="stSidebar"] .stMarkdown {
        color: #374151;
    }

    /* ── 侧边栏内部容器 ── */
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* ── 标题区域 ── */
    .sidebar-title {
        padding: 6px 4px 8px;
        margin-bottom: 2px;
    }
    .sidebar-title h1 {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.3;
    }
    [data-theme="dark"] .sidebar-title h1 { color: #ffffff; }
    [data-theme="light"] .sidebar-title h1 { color: #1a2332; }

    .sidebar-title .version {
        font-size: 0.75rem;
        margin-top: 4px;
    }
    [data-theme="dark"] .sidebar-title .version { color: #7a9cc6; }
    [data-theme="light"] .sidebar-title .version { color: #6b7c93; }

    /* ── 状态徽章 ── */
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px 0;
    }
    .badge-paper {
        background: rgba(76, 175, 80, 0.15);
        color: #4caf50;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    .badge-live {
        background: rgba(244, 67, 54, 0.15);
        color: #f44336;
        border: 1px solid rgba(244, 67, 54, 0.3);
    }

    /* ── 信息卡片 ── */
    .info-card {
        border-radius: 8px;
        padding: 10px 12px;
        margin: 6px 0;
    }
    [data-theme="dark"] .info-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-theme="light"] .info-card {
        background: rgba(0, 0, 0, 0.03);
        border: 1px solid rgba(0, 0, 0, 0.06);
    }

    .info-card-title {
        font-size: 0.88rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    [data-theme="dark"] .info-card-title { color: #7a9cc6; }
    [data-theme="light"] .info-card-title { color: #5b7088; }

    .info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.92rem;
        padding: 2px 0;
    }
    [data-theme="dark"] .info-row { color: #b0bec5; }
    [data-theme="light"] .info-row { color: #4b5563; }

    .info-row .value {
        font-weight: 600;
    }
    [data-theme="dark"] .info-row .value { color: #e0e0e0; }
    [data-theme="light"] .info-row .value { color: #1f2937; }

    .info-row .value-green { color: #4caf50; }
    .info-row .value-red { color: #ef5350; }
    [data-theme="light"] .info-row .value-green { color: #2e7d32; }
    [data-theme="light"] .info-row .value-red { color: #c62828; }
    .info-row .value-blue { color: #42a5f5; }
    [data-theme="light"] .info-row .value-blue { color: #1565c0; }

    /* ── 数据源指示器 ── */
    .source-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .dot-on { background: #4caf50; box-shadow: 0 0 6px rgba(76, 175, 80, 0.5); }
    .dot-off { background: #ef5350; }

    /* ── 页面导航 ── */
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 2px;
    }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        transition: all 0.2s;
        border: 1px solid transparent;
    }

    /* 导航 hover: 深色 */
    [data-theme="dark"] section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(74, 158, 255, 0.08);
        border-color: rgba(74, 158, 255, 0.15);
    }
    [data-theme="dark"] section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background: rgba(74, 158, 255, 0.12);
        border-color: rgba(74, 158, 255, 0.3);
    }

    /* 导航 hover: 浅色 */
    [data-theme="light"] section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(33, 150, 243, 0.06);
        border-color: rgba(33, 150, 243, 0.15);
    }
    [data-theme="light"] section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background: rgba(33, 150, 243, 0.08);
        border-color: rgba(33, 150, 243, 0.25);
    }

    /* ── 分隔线 ── */
    [data-theme="dark"] section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.06);
    }
    [data-theme="light"] section[data-testid="stSidebar"] hr {
        border-color: rgba(0, 0, 0, 0.06);
    }
    section[data-testid="stSidebar"] hr {
        margin: 12px 0;
    }

    /* ── 隐藏 radio 默认圆点 (用 emoji 替代) ── */
    section[data-testid="stSidebar"] .stRadio input[type="radio"] {
        display: none;
    }

    /* ── Metric组件数值字体大小覆盖 (将默认2.25rem改为1.8rem)
       适配总资产、现金、购买力等metric卡片的大数字
       使用通用选择器，适配Streamlit emotion类名动态变化 ── */
    /* 1. 通过data-testid定位metric数值 (官方稳定选择器) */
    [data-testid="stMetricValue"],
    /* 2. 匹配所有emotion类名下的metric数值 */
    div[class*="st-emotion-cache-"] [data-testid="stMetricValue"],
    /* 3. metric容器内的大数字元素 */
    [data-testid="stMetric"] > div > div > div:nth-child(2) {
        font-size: 1.8rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """渲染侧边栏"""
    config = _get_config()

    # 初始化数据源模式
    _init_data_source()

    # 注入自定义样式
    _inject_sidebar_css()

    # ── 标题区域 ──
    is_paper = config.is_paper_trading()
    mode_class = "badge-paper" if is_paper else "badge-live"
    mode_text = "📝 模拟盘" if is_paper else "🔴 实盘"

    st.sidebar.markdown(f"""
    <div class="sidebar-title">
        <h1>Soloquant</h1>
        <div class="version">v0.8.0 · {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
    </div>
    <div style="text-align: center; margin: 8px 0 4px;">
        <span class="status-badge {mode_class}">{mode_text}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── 数据源切换 ──
    with st.sidebar.expander("🔌 数据源切换", expanded=False):
        current_mode = st.session_state.get("data_source_mode", "local")
        mode_options = ["本地直连", "云端API"]
        mode_idx = 1 if current_mode == "cloud" else 0
        selected_mode = st.radio(
            "数据来源",
            mode_options,
            index=mode_idx,
            horizontal=True,
            key="data_source_radio",
            label_visibility="collapsed",
        )

        new_mode = "cloud" if selected_mode == "云端API" else "local"

        # 云端模式下显示 API 地址输入
        if new_mode == "cloud":
            api_url = st.text_input(
                "云端 API 地址",
                value=st.session_state.get("cloud_api_url", ""),
                placeholder="http://101.96.194.75:8001",
                key="cloud_api_url_input",
            )
            if api_url != st.session_state.get("cloud_api_url", ""):
                st.session_state["cloud_api_url"] = api_url
                st.cache_data.clear()
                st.rerun()

        # 模式切换
        if new_mode != current_mode:
            st.session_state["data_source_mode"] = new_mode
            cc = _get_cloud_client()
            cc.set_mode(new_mode, st.session_state.get("cloud_api_url", ""))
            st.cache_data.clear()
            st.rerun()

        # 显示当前模式状态
        if new_mode == "cloud":
            api_base = st.session_state.get("cloud_api_url", "")
            if api_base:
                st.caption(f"📡 连接: `{api_base}`")
                # 快速健康检查
                try:
                    cc = _get_cloud_client()
                    health = cc.health_check()
                    if health and health.get("status") == "ok":
                        st.caption("✅ 云端连接正常")
                    elif health:
                        st.caption(f"⚠️ {health.get('status', '未知')}")
                except Exception:
                    st.caption("⚠️ 无法连接云端")
            else:
                st.caption("⚠️ 请输入云端 API 地址")
        else:
            st.caption("💻 本地直连模式")

    # ── 数据源状态 ──
    alpaca_on = config.has_alpaca_keys()
    finnhub_on = config.has_finnhub_key()
    alpaca_dot = "dot-on" if alpaca_on else "dot-off"
    finnhub_dot = "dot-on" if finnhub_on else "dot-off"
    alpaca_text = "已连接" if alpaca_on else "未配置"
    finnhub_text = "已连接" if finnhub_on else "未配置"

    st.sidebar.markdown(f"""
    <div class="info-card">
        <div class="info-card-title">📡 数据源</div>
        <div class="info-row">
            <span><span class="source-dot {alpaca_dot}"></span>Alpaca</span>
            <span class="value {'value-green' if alpaca_on else 'value-red'}">{alpaca_text}</span>
        </div>
        <div class="info-row">
            <span><span class="source-dot {finnhub_dot}"></span>Finnhub</span>
            <span class="value {'value-green' if finnhub_on else 'value-red'}">{finnhub_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 风控参数 ──
    st.sidebar.markdown(f"""
    <div class="info-card">
        <div class="info-card-title">🛡️ 风控</div>
        <div class="info-row">
            <span>单票仓位</span>
            <span class="value value-blue">{config.MAX_POSITION_PCT:.0%}</span>
        </div>
        <div class="info-row">
            <span>日最大亏损</span>
            <span class="value value-red">{config.MAX_DAILY_LOSS:.0%}</span>
        </div>
        <div class="info-row">
            <span>最大回撤</span>
            <span class="value value-red">{config.MAX_DRAWDOWN:.0%}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 选股参数 ──
    st.sidebar.markdown(f"""
    <div class="info-card">
        <div class="info-card-title">🎯 选股</div>
        <div class="info-row">
            <span>候选上限</span>
            <span class="value">{config.MAX_CANDIDATES}</span>
        </div>
        <div class="info-row">
            <span>信号 Top-K</span>
            <span class="value">{config.TOP_K_SIGNALS}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 页面导航 ──
    st.sidebar.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
    page = st.sidebar.radio(
        "导航",
        ["📊 总览首页", "📈 行情分析", "🎯 智能选股", "📈 策略管理", "💰 交易记录", "🔍 市场调研", "🌐 宏观指标", "📊 期权数据", "🪙 加密货币", "🧠 模型训练", "⚙️ 系统设置"],
        index=0,
        label_visibility="collapsed",
    )
    return page


# ═══════════════════════════════════════════════════════
# 页面1: 总览首页
# ═══════════════════════════════════════════════════════

def _render_realtime_panel():
    """渲染实时行情面板 (Phase 7 T1)"""
    try:
        rm = _get_realtime_manager()
        status = rm.get_status()

        # 启动/停止按钮 + 状态指示
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 1])

        with ctrl_col1:
            if not status["running"]:
                if st.button("▶️ 启动实时行情", type="primary", key="btn_start_realtime"):
                    _start_realtime_stream()
                    st.rerun()
            else:
                if st.button("⏹️ 停止实时行情", type="secondary", key="btn_stop_realtime"):
                    _stop_realtime_stream()
                    st.rerun()

        with ctrl_col2:
            st.metric("实时连接", "🟢 运行中" if status["running"] else "⚪ 未启动")
        with ctrl_col3:
            st.metric("缓存价格", f"{status['cached_prices']} 只")
        with ctrl_col4:
            st.metric("告警数", f"{status['alerts_count']} 条")

        # 数据源状态
        src_col1, src_col2, src_col3 = st.columns(3)
        src_col1.caption(f"📡 Alpaca: {'✅' if status['alpaca_available'] else '❌'}")
        src_col2.caption(f"📡 Finnhub: {'✅' if status['finnhub_available'] else '❌'}")
        src_col3.caption(f"🔧 模式: `{status['data_source']}`")

        # 实时价格表
        prices = rm.get_all_prices()
        if prices:
            price_rows = []
            for sym, quote in prices.items():
                pnl_color = "🟢" if quote.change >= 0 else "🔴"
                price_rows.append({
                    "代码": sym,
                    "价格": f"",
                    "涨跌": f"{pnl_color} " if quote.change else "—",
                    "涨跌幅": f"{quote.change_pct:+.2f}%" if quote.change_pct else "—",
                    "来源": quote.source,
                    "时间": quote.timestamp.strftime("%H:%M:%S"),
                })
            st.dataframe(pd.DataFrame(price_rows), width='stretch', hide_index=True)
        else:
            st.info("💡 暂无实时价格数据。点击「启动实时行情」开始接收盘中数据。")

        # 最近告警
        alerts = rm.get_alerts(limit=5)
        if alerts:
            st.markdown("**🚨 最近价格异动告警:**")
            for a in reversed(alerts):
                st.text(f"  ⚠️ [{a.triggered_at.strftime('%H:%M:%S')}] {a.message}")

    except Exception as e:
        st.info(f"💡 实时行情模块加载中... ({e})")


# ─── 实时行情启动/停止 (后台线程 + asyncio) ───

_realtime_loop = None
_realtime_thread = None


def _start_realtime_stream():
    """在后台线程中启动实时行情 WebSocket"""
    import asyncio
    import threading

    global _realtime_loop, _realtime_thread

    config = _get_config()

    # 确定订阅列表: 持仓 + 关注列表
    symbols = set()
    try:
        pm = _get_position_manager()
        account = pm.get_account()
        if account:
            positions = pm.get_position_summary()
            if positions and "positions" in positions:
                symbols.update(p["symbol"] for p in positions["positions"])
    except Exception:
        pass

    watchlist = config.WATCHLIST or config.DEFAULT_WATCHLIST
    symbols.update(watchlist)
    symbols = list(symbols)

    if not symbols:
        st.warning("无订阅股票 (无持仓且关注列表为空)")
        return

    def _background_run():
        global _realtime_loop
        _realtime_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_realtime_loop)

        async def _start():
            rm = _get_realtime_manager()
            await rm.start(symbols)

        _realtime_loop.run_until_complete(_start())
        _realtime_loop.run_forever()

    _realtime_thread = threading.Thread(target=_background_run, daemon=True)
    _realtime_thread.start()
    st.success(f"⚡ 实时行情已启动! 订阅 {len(symbols)} 只股票")


def _stop_realtime_stream():
    """停止后台实时行情"""
    import asyncio

    global _realtime_loop, _realtime_thread

    rm = _get_realtime_manager()

    if _realtime_loop and _realtime_loop.is_running():
        asyncio.run_coroutine_threadsafe(rm.stop(), _realtime_loop)
        _realtime_loop.call_soon_threadsafe(_realtime_loop.stop)

    _realtime_thread = None
    _realtime_loop = None
    st.info("⏹️ 实时行情已停止")


# ═══════════════════════════════════════════════════════
# 页面1: 总览首页
# ═══════════════════════════════════════════════════════

def page_overview():
    """总览首页"""
    st.title("📊 总览首页")
    st.caption("账户资产、持仓分布、今日盈亏、告警状态")

    # 账户概览卡片
    account = get_account_info()
    if account:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💼 总资产", f"${account['equity']:,.2f}")
        col2.metric("💵 现金", f"${account['cash']:,.2f}")
        col3.metric("💪 贴买力", f"${account['buying_power']:,.2f}")
        col4.metric("📋 交易模式", account["trading_mode"])
    else:
        _err = st.session_state.get("_account_error", "")
        if _err:
            st.warning(f"⚠️ 账户数据获取失败: {_err}")
        else:
            st.info("💡 Alpaca 未配置或无法连接，显示模拟数据")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💼 总资产", "$100,000.00")
        col2.metric("💵 现金", "$100,000.00")
        col3.metric("💪 购买力", "$200,000.00")
        col4.metric("📋 交易模式", "📝 模拟盘")

    st.divider()

    # 持仓分布
    st.subheader("📈 持仓分布")
    pos_data = get_positions_list()

    if pos_data and "positions" in pos_data and pos_data["positions"]:
        positions = pos_data["positions"]
        df_pos = pd.DataFrame(positions)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            # 持仓饼图
            fig_pie = px.pie(
                df_pos,
                values="market_value",
                names="symbol",
                title="持仓占比分布",
                hole=0.4,
            )
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, width='stretch')

        with col_right:
            # 持仓明细表
            st.dataframe(
                df_pos[["symbol", "qty", "avg_cost", "market_value",
                        "unrealized_pnl", "unrealized_pnl_pct", "weight"]]
                .style.format({
                    "qty": "{:.2f}",
                    "avg_cost": "${:.2f}",
                    "market_value": "${:.2f}",
                    "unrealized_pnl": "${:.2f}",
                    "unrealized_pnl_pct": "{:.1%}",
                    "weight": "{:.1%}",
                }),
                width='stretch',
                height=350,
            )

        # 浮动盈亏柱状图
        st.subheader("📊 各持仓浮动盈亏")
        fig_bar = go.Figure()
        colors = ["green" if p["unrealized_pnl"] >= 0 else "red" for p in positions]
        fig_bar.add_trace(go.Bar(
            x=[p["symbol"] for p in positions],
            y=[p["unrealized_pnl"] for p in positions],
            marker_color=colors,
            text=[f"${p['unrealized_pnl']:.2f}" for p in positions],
            textposition="auto",
        ))
        fig_bar.update_layout(
            title="浮动盈亏 ($)",
            xaxis_title="股票",
            yaxis_title="盈亏 ($)",
            height=300,
        )
        st.plotly_chart(fig_bar, width='stretch')

    else:
        st.info("📭 当前无持仓，运行工作流后将显示持仓信息")

    st.divider()

    # 实时行情面板 (Phase 7 T1)
    st.subheader("⚡ 实时行情")
    _render_realtime_panel()

    st.divider()

    # 最近通知
    st.subheader("🔔 最近通知")
    notifications = get_notifications(10)
    if notifications:
        icons = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌",
                 "CRITICAL": "🚨", "SUCCESS": "✅"}
        rows = []
        for n in reversed(notifications):
            level = n.get("level", "INFO")
            rows.append({
                "级别": f"{icons.get(level, 'ℹ️')} {level}",
                "时间": n.get("timestamp", ""),
                "标题": n.get("title", ""),
                "内容": n.get("message", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("暂无通知记录")

# ═══════════════════════════════════════════════════════
# 用户关注列表管理 (持久化到本地 JSON)
# ═══════════════════════════════════════════════════════

def _get_watchlist_path():
    """获取用户关注列表文件路径"""
    try:
        cfg = _get_config()
        return cfg.DATA_DIR / "user_watchlist.json"
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
# 智能选股结果缓存 (持久化到本地 JSON, 下次启动自动恢复)
# ═══════════════════════════════════════════════════════

def _get_screen_cache_path():
    """获取选股结果缓存文件路径"""
    try:
        cfg = _get_config()
        return Path(cfg.DATA_DIR) / "last_screen_result.json"
    except Exception:
        return Path("data") / "last_screen_result.json"


def _save_screen_cache(result):
    """将选股结果序列化保存到本地 JSON"""
    import json
    filepath = _get_screen_cache_path()
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # 将 StockScreenResult 序列化为 dict
        cache_data = {
            "query": result.query,
            "params": result.params,
            "total_scanned": result.total_scanned,
            "total_filtered": result.total_filtered,
            "plain_summary": result.plain_summary,
            "picks": [
                {
                    "symbol": p.symbol,
                    "name": p.name,
                    "price": p.price,
                    "sector": p.sector,
                    "composite_score": p.composite_score,
                    "tech_score": p.tech_score,
                    "fundamental_score": p.fundamental_score,
                    "sentiment_score": p.sentiment_score,
                    "momentum_score": p.momentum_score,
                    "llm_reason": p.llm_reason,
                }
                for p in result.picks
            ],
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"选股结果缓存保存失败: {e}")


def _load_screen_cache():
    """从本地 JSON 恢复上一次选股结果, 返回 StockScreenResult 或 None"""
    import json
    filepath = _get_screen_cache_path()
    try:
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 重建 StockScreenResult 对象
        from src.models import StockScreenResult, StockPick
        picks = [StockPick(**p) for p in data.get("picks", [])]
        result = StockScreenResult(
            query=data.get("query", ""),
            params=data.get("params", {}),
            picks=picks,
            total_scanned=data.get("total_scanned", 0),
            total_filtered=data.get("total_filtered", 0),
            plain_summary=data.get("plain_summary", ""),
        )
        return result
    except Exception:
        return None


# ═══════════════════════════════════════════════════════
# 页面2: 智能选股
# ═══════════════════════════════════════════════════════

def page_stock_screen():
    """智能选股页面"""
    st.title("🎯 智能选股")
    st.caption("用自然语言选股，AI 自动扫描全美股并多维度评分")

    # ─── 自然语言输入区 ───
    st.subheader("💬 自然语言选股")
    st.caption('试试输入: "帮我选5支10美元的潜力股" 或 "选3只20-50美元的科技股"')

    nl_input = st.text_input(
        "输入选股需求",
        placeholder="例如: 帮我选5支10美元的潜力股",
        key="nl_screen_input",
    )

    # ─── 参数微调区 ───
    with st.expander("⚙️ 手动调整参数 (可选)", expanded=False, key="expander_manual_screen_params"):
        col1, col2, col3 = st.columns(3)
        with col1:
            screen_count = st.number_input("选股数量", 1, 20, 5, key="screen_count")
            screen_price_min = st.number_input("最低价格 ($)", 0.0, 10000.0, 0.0, step=1.0, key="screen_price_min")
        with col2:
            screen_price_max = st.number_input("最高价格 ($)", 0.0, 10000.0, 10.0, step=1.0, key="screen_price_max", help="0 = 不限")
            screen_sector = st.selectbox(
                "行业偏好",
                ["", "科技", "金融", "能源", "医疗", "消费"],
                key="screen_sector",
            )
        with col3:
            screen_keywords = st.text_input("关键词", "潜力", key="screen_keywords", help="如: 潜力/成长/低价")
            st.caption("💡 参数会自动从自然语言中提取，也可手动调整")

    # ─── 执行选股 ───
    if st.button("🔍 开始选股", type="primary", key="btn_screen"):
        # 确定参数: 如果有自然语言输入，用 NLRouter 解析; 否则用手动参数
        params = None

        if nl_input.strip():
            # 用 NLRouter 的意图识别提取参数
            try:
                from src.llm.intent_recognizer import intent_recognizer
                llm_intent = intent_recognizer.recognize(nl_input)
                if llm_intent.intent == "stock_screen":
                    params = llm_intent.params
                    st.info(f"🧠 AI 解析结果: 数量={params.get('count', 5)}, "
                            f"价格={params.get('price_min', 0)}~{params.get('price_max', '不限')}, "
                            f"行业={params.get('sector', '不限')}, 关键词={params.get('keywords', '无')}")
                else:
                    st.warning(f"未识别为选股意图 (识别为: {llm_intent.intent})，使用手动参数")
            except Exception as e:
                st.warning(f"AI 解析失败 ({e})，使用手动参数")

        if params is None:
            params = {
                "count": int(screen_count),
                "price_min": float(screen_price_min),
                "price_max": float(screen_price_max),
                "sector": screen_sector,
                "keywords": screen_keywords,
            }

        # 执行选股
        with st.spinner("🔍 正在扫描全美股并评分... (可能需要 30-60 秒)"):
            try:
                from src.engine.stock_screener import stock_screener
                result = stock_screener.screen(params, query=nl_input or "手动选股")
                st.session_state["screen_result"] = result
                # 持久化选股结果到本地, 下次启动自动恢复
                _save_screen_cache(result)
            except Exception as e:
                st.error(f"选股失败: {e}")
                return

    # ─── 展示选股结果 ───
    result = st.session_state.get("screen_result")
    if result:
        if not result.picks:
            st.warning(result.plain_summary or "未找到符合条件的股票")
            return

        st.divider()
        st.subheader(f"📊 选股结果 (扫描 {result.total_scanned} 只 -> 入选 {len(result.picks)} 只)")

        # 概览指标
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("扫描总数", str(result.total_scanned))
        col2.metric("价格筛选后", str(result.total_filtered))
        col3.metric("入选数量", str(len(result.picks)))
        avg_score = sum(p.composite_score for p in result.picks) / len(result.picks)
        col4.metric("平均综合分", f"{avg_score:.1f}")

        # 候选股票表格
        st.subheader("📋 入选股票详情")

        # ── 获取实时行情 + 财务指标 (与行情分析页面一致) ──
        with st.spinner("⏳ 正在加载实时行情与财务指标..."):
            try:
                from src.ui.chart_analysis import fetch_realtime_quote, fetch_company_financials
            except Exception:
                fetch_realtime_quote = None
                fetch_company_financials = None

            quotes = {}
            financials_map = {}
            for p in result.picks:
                try:
                    if fetch_realtime_quote:
                        quotes[p.symbol] = fetch_realtime_quote(p.symbol) or {}
                    if fetch_company_financials:
                        financials_map[p.symbol] = fetch_company_financials(p.symbol) or {}
                except Exception:
                    quotes[p.symbol] = {}
                    financials_map[p.symbol] = {}

        # 格式化辅助函数
        def _fmt_price(v, fallback=None):
            try:
                v = float(v)
                return f"${v:,.2f}"
            except (TypeError, ValueError):
                return "-" if fallback is None else fallback

        def _fmt_pct(v):
            try:
                v = float(v)
                return f"{v:+.2f}%"
            except (TypeError, ValueError):
                return "-"

        def _fmt_volume(v):
            try:
                v = float(v)
                return f"{v:,.0f}"
            except (TypeError, ValueError):
                return "-"

        def _fmt_metric(v, fmt="{:.2f}"):
            """格式化财务指标，统一返回字符串，避免 pyarrow 类型混合错误"""
            try:
                return fmt.format(float(v))
            except (TypeError, ValueError):
                return "-"

        picks_data = []
        for p in result.picks:
            q = quotes.get(p.symbol, {})
            fin = financials_map.get(p.symbol, {})
            metric = fin.get("metric", {}) if fin else {}

            picks_data.append({
                "代码": p.symbol,
                "名称": p.name,
                "行业": p.sector or "-",
                # ── 实时行情 (与行情分析一致) ──
                "最新价 ($)": _fmt_price(q.get("price"), fallback=_fmt_price(p.price)),
                "涨跌幅": _fmt_pct(q.get("change_pct")),
                "开盘 ($)": _fmt_price(q.get("open")),
                "最高 ($)": _fmt_price(q.get("high")),
                "最低 ($)": _fmt_price(q.get("low")),
                "昨收 ($)": _fmt_price(q.get("prev_close")),
                "成交量": _fmt_volume(q.get("volume")),
                # ── 关键财务指标 (与行情分析一致) ── 统一用 _fmt_metric 格式化，避免 Arrow 类型混合错误
                "P/E (TTM)": _fmt_metric(metric.get("peNormalizedAnnual", metric.get("peTTM"))),
                "P/B": _fmt_metric(metric.get("pbAnnual")),
                "ROE": _fmt_metric(metric.get("roeTTM"), "{:.2f}%"),
                "营收增长": _fmt_metric(metric.get("revenueGrowth5Y"), "{:.2f}%"),
                "毛利率": _fmt_metric(metric.get("grossMarginTTM"), "{:.2f}%"),
                "Beta": _fmt_metric(metric.get("beta")),
                # ── 原有评分列 ──
                "综合评分": p.composite_score,
                "技术面": p.tech_score,
                "基本面": p.fundamental_score,
                "情绪面": p.sentiment_score,
                "动量": p.momentum_score,
                "AI 推荐理由": p.llm_reason or "-",
            })
        df_picks = pd.DataFrame(picks_data)
        st.dataframe(df_picks, width='stretch', hide_index=True)

        # 综合评分柱状图
        st.subheader("📊 综合评分对比")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=[f"{p.symbol}" for p in result.picks],
            y=[p.composite_score for p in result.picks],
            marker_color=[
                "green" if p.composite_score >= 60 else
                "orange" if p.composite_score >= 50 else "red"
                for p in result.picks
            ],
            text=[f"{p.composite_score:.1f}" for p in result.picks],
            textposition="auto",
        ))
        fig_bar.update_layout(
            title="综合评分 (0-100, 越高越好)",
            xaxis_title="股票代码",
            yaxis_title="综合评分",
            yaxis=dict(range=[0, 100]),
            height=350,
        )
        st.plotly_chart(fig_bar, width='stretch')

        # 四维评分雷达图 (选第一只展示)
        if len(result.picks) > 0:
            st.subheader("🎯 四维评分详情")
            selected_pick = st.selectbox(
                "选择股票查看四维评分",
                [f"{p.symbol} ({p.name})" for p in result.picks],
                key="radar_pick_select",
            )
            idx = [f"{p.symbol} ({p.name})" for p in result.picks].index(selected_pick)
            pick = result.picks[idx]

            categories = ["技术面", "基本面", "情绪面", "动量"]
            values = [pick.tech_score, pick.fundamental_score, pick.sentiment_score, pick.momentum_score]

            # ─── 左右两栏: 左侧雷达图, 右侧操作面板 ───
            col_radar, col_action = st.columns([3, 2])

            with col_radar:
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + [values[0]],  # 闭合
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=pick.symbol,
                    line=dict(color="royalblue"),
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 100])),
                    title=f"{pick.symbol} 四维评分雷达图",
                    height=350,
                )
                st.plotly_chart(fig_radar, width='stretch')

                if pick.llm_reason:
                    st.info(f"💡 AI 推荐理由: {pick.llm_reason}")

            # ─── 公司简介 ───
            with st.spinner(f"正在加载 {pick.symbol} 公司信息..."):
                try:
                    from src.data.collector import data_collector
                    profile = None
                    # 优先用 FMP 获取详细公司概况 (含 description)
                    try:
                        profile = data_collector.get_company_profile_fmp(pick.symbol)
                    except Exception:
                        pass
                    # FMP 不可用时回退到 Finnhub
                    if not profile or not profile.get("description"):
                        profile = data_collector.get_company_profile(pick.symbol) or {}

                    if profile and profile.get("description"):
                        st.markdown("#### 🏢 公司简介")
                        # 公司基本信息卡片
                        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                        with col_info1:
                            st.metric("行业", profile.get("sector") or profile.get("finnhubIndustry") or "-")
                        with col_info2:
                            st.metric("CEO", profile.get("ceo", "-") or "-")
                        with col_info3:
                            employees = profile.get("employees", 0) or profile.get("fullTimeEmployees", 0) or 0
                            try:
                                employees = int(employees)
                                st.metric("员工数", f"{employees:,}" if employees else "-")
                            except (ValueError, TypeError):
                                st.metric("员工数", str(employees) if employees else "-")
                        with col_info4:
                            st.metric("网站", profile.get("website", "-") or "-")

                        # 公司简介正文
                        description = profile.get("description", "")
                        if description and len(description) > 20:
                            with st.expander("📖 查看公司简介详情", expanded=True):
                                st.write(description)
                    else:
                        st.caption("📭 暂无该公司简介信息")
                except Exception as e:
                    st.caption(f"📭 公司信息加载失败: {e}")

            with col_action:
                # ════════════════════════════════════════
                # 功能1: 关注/取消关注
                # ════════════════════════════════════════
                st.markdown("#### ⭐ 关注管理")

                # 加载用户关注列表 (持久化到本地 JSON)
                user_watchlist = _load_user_watchlist()
                is_watched = pick.symbol in user_watchlist

                col_watch1, col_watch2 = st.columns(2)
                with col_watch1:
                    if is_watched:
                        if st.button("💔 取消关注", key=f"btn_unwatch_{pick.symbol}", use_container_width=True):
                            user_watchlist.remove(pick.symbol)
                            _save_user_watchlist(user_watchlist)
                            st.success(f"已取消关注 {pick.symbol}")
                            st.rerun()
                    else:
                        if st.button("⭐ 关注该股票", key=f"btn_watch_{pick.symbol}", use_container_width=True):
                            if pick.symbol not in user_watchlist:
                                user_watchlist.append(pick.symbol)
                                _save_user_watchlist(user_watchlist)
                            st.success(f"已关注 {pick.symbol}")
                            st.rerun()
                with col_watch2:
                    watch_count = len(user_watchlist)
                    st.metric("已关注股票数", f"{watch_count}")

                # 展示当前关注列表
                if user_watchlist:
                    watch_chips = " ".join(f"`{s}`" for s in user_watchlist[:15])
                    if len(user_watchlist) > 15:
                        watch_chips += f" ...等 {len(user_watchlist)} 只"
                    st.caption(f"📋 关注列表: {watch_chips}")

                st.divider()

                # ════════════════════════════════════════
                # 功能2: 创建交易策略
                # ════════════════════════════════════════
                st.markdown("#### 📈 创建交易策略")

                with st.expander("为该股票创建策略", expanded=False, key=f"expander_strategy_{pick.symbol}"):
                    # 策略名称 (默认值包含股票代码和日期)
                    default_name = f"选股策略_{pick.symbol}_{datetime.now().strftime('%Y%m%d')}"
                    strategy_name = st.text_input(
                        "策略名称",
                        value=default_name,
                        key=f"strat_name_{pick.symbol}",
                    )

                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        model_type = st.selectbox(
                            "模型类型",
                            ["lightgbm", "lstm"],
                            key=f"strat_model_{pick.symbol}",
                        )
                    with col_s2:
                        max_position = st.slider(
                            "最大仓位",
                            min_value=0.05, max_value=0.50, value=0.30, step=0.05,
                            format="%.0f%%",
                            key=f"strat_pos_{pick.symbol}",
                        )

                    # 可选: 使用策略模板
                    use_template = st.checkbox(
                        "使用策略模板 (自动填充因子和参数)",
                        value=False,
                        key=f"strat_use_tmpl_{pick.symbol}",
                    )

                    selected_template_id = None
                    if use_template:
                        try:
                            from src.engine.strategy_templates import strategy_templates
                            templates = strategy_templates.list_templates()
                            if templates:
                                tmpl_options = {t["template_id"]: f"{t['name']} ({t['risk_level']})" for t in templates}
                                selected_template_id = st.selectbox(
                                    "选择模板",
                                    options=list(tmpl_options.keys()),
                                    format_func=lambda x: tmpl_options[x],
                                    key=f"strat_tmpl_{pick.symbol}",
                                )
                            else:
                                st.warning("暂无可用模板")
                        except Exception as e:
                            st.warning(f"模板加载失败: {e}")

                    if st.button("🚀 创建策略", type="primary", key=f"btn_create_strat_{pick.symbol}", use_container_width=True):
                        try:
                            sm = _get_strategy_manager()
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            strategy_id = f"screen_{pick.symbol.lower()}_{timestamp}"

                            if use_template and selected_template_id:
                                # 从模板创建, 但替换股票池为当前选中的股票
                                from src.engine.strategy_templates import strategy_templates as stm
                                template = stm._templates.get(selected_template_id)
                                if template:
                                    strategy = sm.create_strategy(
                                        strategy_id=strategy_id,
                                        name=strategy_name or default_name,
                                        model_type=template.model_type,
                                        factors=template.factors.copy(),
                                        stock_pool=[pick.symbol],
                                        max_position=max_position,
                                        hyper_params=template.hyper_params.copy(),
                                    )
                                    strategy.rebalance_freq = template.rebalance_freq
                                    sm._save(strategy)
                                else:
                                    st.error("模板不存在")
                                    return
                            else:
                                # 自定义创建
                                strategy = sm.create_strategy(
                                    strategy_id=strategy_id,
                                    name=strategy_name or default_name,
                                    model_type=model_type,
                                    factors=[],
                                    stock_pool=[pick.symbol],
                                    max_position=max_position,
                                )

                            st.success(f"✅ 策略 '{strategy.name}' 创建成功! (ID: {strategy_id})")
                            st.info(f"💡 股票池: {pick.symbol} | 模型: {strategy.model_type} | 仓位: {max_position:.0%}")
                            st.caption("👉 前往「📈 策略管理」查看详情和回测")
                        except Exception as e:
                            st.error(f"创建失败: {e}")

            # ════════════════════════════════════════
            # 功能3: K线图展示
            # ════════════════════════════════════════
            st.divider()
            with st.expander(f"📊 {pick.symbol} K线图", expanded=True, key=f"expander_kline_{pick.symbol}"):
                # K线参数选择
                col_k1, col_k2, col_k3 = st.columns(3)
                with col_k1:
                    kline_period = st.selectbox(
                        "周期",
                        ["日K", "周K", "月K"],
                        key=f"kline_period_{pick.symbol}",
                    )
                with col_k2:
                    kline_range = st.selectbox(
                        "时间范围",
                        ["3月", "半年", "1年", "全部"],
                        index=1,
                        key=f"kline_range_{pick.symbol}",
                    )
                with col_k3:
                    # 均线多选
                    ma_options = st.multiselect(
                        "均线",
                        ["MA5", "MA10", "MA20", "MA60"],
                        default=["MA5", "MA20", "MA60"],
                        key=f"kline_ma_{pick.symbol}",
                    )

                range_map = {"3月": 90, "半年": 180, "1年": 365, "全部": None}
                days_back = range_map.get(kline_range, 180)

                with st.spinner(f"正在加载 {pick.symbol} K线数据..."):
                    try:
                        from src.ui.chart_analysis import fetch_kline_data, render_kline_chart, calc_ma
                        kline_df = fetch_kline_data(pick.symbol, kline_period, days_back)

                        if kline_df is not None and len(kline_df) > 0:
                            # 计算均线
                            ma_periods = []
                            if ma_options:
                                ma_periods = [int(ma.replace("MA", "")) for ma in ma_options]
                            if ma_periods:
                                kline_df = calc_ma(kline_df, ma_periods)

                            # 渲染K线图
                            fig_kline = render_kline_chart(
                                kline_df,
                                show_ma=ma_options if ma_options else None,
                                show_volume=True,
                                show_macd=False,
                                show_kdj=False,
                                show_boll=False,
                                show_rsi=False,
                                symbol=pick.symbol,
                                period=kline_period,
                            )
                            fig_kline.update_layout(height=500)
                            st.plotly_chart(fig_kline, width='stretch')

                            # 显示最近几日K线数据摘要
                            st.caption(f"📅 数据范围: {kline_df['datetime'].iloc[0].strftime('%Y-%m-%d')} ~ {kline_df['datetime'].iloc[-1].strftime('%Y-%m-%d')} | 共 {len(kline_df)} 根K线")
                        else:
                            st.warning(f"暂无 {pick.symbol} 的K线数据，请检查数据源")
                    except Exception as e:
                        st.warning(f"K线图加载失败: {e}")

        # 小白解读
        st.divider()
        st.subheader("📝 选股总结")

        # 筛选概览表
        _params = result.params or {}
        _price_min = _params.get("price_min", 0)
        _price_max = _params.get("price_max", 0)
        _count = _params.get("count", 5)
        _cond = f"${_price_min}~${_price_max}" if _price_max else f">${_price_min}"

        overview_data = {
            "项目": ["筛选条件", "扫描总数", "价格筛选后", "入选数量", "平均综合分"],
            "详情": [
                f"{_cond}, 选 {_count} 只",
                str(result.total_scanned),
                str(result.total_filtered),
                str(len(result.picks)),
                f"{avg_score:.1f}",
            ],
        }
        st.table(pd.DataFrame(overview_data))

        # 入选股票总结表
        st.markdown("**入选股票一览**")
        summary_picks = []
        for i, p in enumerate(result.picks, 1):
            summary_picks.append({
                "排名": i,
                "代码": p.symbol,
                "名称": p.name,
                "行业": p.sector or "-",
                "价格 ($)": f"${p.price}",
                "综合评分": p.composite_score,
                "AI 推荐理由": p.llm_reason or "-",
            })
        st.table(pd.DataFrame(summary_picks))

        # 风险提示
        st.caption("⚠️ 以上分析仅供参考，不构成投资建议")

        # 后续操作提示
        st.divider()
        st.subheader("💡 后续操作")
        st.markdown(
            "  • 在上方 **🎯 四维评分详情** 中查看K线图、关注股票、创建交易策略\n"
            "  • 前往 **📈 策略管理** 查看和管理已创建的策略\n"
            "  • 或在对话中输入: \"为 AAPL 生成交易策略\"\n"
            "  • ⚠️ 以上分析仅供参考，不构成投资建议"
        )
    else:
        # 尝试从本地缓存恢复上一次选股结果 (仅首次进入页面时)
        if "screen_result" not in st.session_state:
            cached = _load_screen_cache()
            if cached:
                st.session_state["screen_result"] = cached
                st.rerun()
        st.info("💡 输入选股需求或调整参数后，点击「开始选股」")


# ═══════════════════════════════════════════════════════
# 页面3: 策略管理
# ═══════════════════════════════════════════════════════

def page_strategies():
    """策略管理"""
    st.title("📈 策略管理")
    st.caption("策略列表、回测结果、评级展示、模板创建、组合管理")

    # ─── 从模板创建策略 ───
    st.subheader("📋 从模板创建策略")
    st.caption("无需理解因子和模型，选择一个模板即可快速创建策略")

    try:
        from src.engine.strategy_templates import strategy_templates
        templates = strategy_templates.list_templates()
    except Exception:
        templates = []

    if templates:
        # 模板卡片展示
        template_cols = st.columns(min(len(templates), 3))
        for i, tmpl in enumerate(templates):
            col = template_cols[i % len(template_cols)]
            risk_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
            risk_icon = risk_colors.get(tmpl["risk_level"], "⚪")
            with col:
                st.markdown(f"**{tmpl['name']}** {risk_icon}")
                st.caption(tmpl["description"][:80] + "..." if len(tmpl["description"]) > 80 else tmpl["description"])
                st.text(f"模型: {tmpl['model_type']} | 因子: {tmpl['factors_count']}个")
                st.text(f"股票池: {tmpl['stock_pool_size']}只 | 仓位: {tmpl['max_position']:.0%}")
                tags_str = " ".join(f"#{t}" for t in tmpl["tags"][:3])
                st.caption(tags_str)
                if st.button(f"创建", key=f"btn_tmpl_{tmpl['template_id']}"):
                    try:
                        strategy = strategy_templates.create_from_template(tmpl["template_id"])
                        if strategy:
                            st.success(f"✅ 策略 '{strategy.name}' 创建成功! (ID: {strategy.strategy_id})")
                            st.rerun()
                    except Exception as e:
                        st.error(f"创建失败: {e}")

    st.divider()

    # ─── 策略列表 ───
    strategies_raw = get_strategies()

    if not strategies_raw:
        st.info("📭 暂无策略，请从上方模板创建，或运行工作流自动生成")
        return

    # 云端模式返回 dict 列表，本地模式返回对象列表，统一处理
    strategies = strategies_raw

    st.subheader(f"📋 策略列表 ({len(strategies)} 个)")
    # 将策略数据转为扁平 dict (兼容对象和 dict 两种格式)
    strat_rows = []
    for s in strategies:
        if isinstance(s, dict):
            # 云端模式: 已经是 dict
            strat_rows.append({
                "策略ID": s.get("strategy_id", ""),
                "名称": s.get("name", ""),
                "模型": s.get("model_type", ""),
                "状态": s.get("status", ""),
                "股票池": ", ".join(s.get("stock_pool", [])) if s.get("stock_pool") else "-",
                "因子数": len(s.get("factors", [])) if s.get("factors") else 0,
                "最大仓位": f"{s.get('max_position', 0):.0%}" if s.get("max_position") else "-",
                "调仓频率": s.get("rebalance_freq", "-"),
                "创建时间": s.get("created_at", ""),
            })
        else:
            # 本地模式: StrategyConfig 对象
            strat_rows.append({
                "策略ID": s.strategy_id,
                "名称": s.name,
                "模型": s.model_type,
                "状态": s.status,
                "股票池": ", ".join(s.stock_pool) if s.stock_pool else "-",
                "因子数": len(s.factors),
                "最大仓位": f"{s.max_position:.0%}",
                "调仓频率": s.rebalance_freq,
                "创建时间": s.created_at.strftime("%Y-%m-%d %H:%M"),
            })
    df_strat = pd.DataFrame(strat_rows)
    st.dataframe(df_strat, width='stretch', hide_index=True)

    st.divider()

    # ─── 多策略组合管理 ───
    st.subheader("📊 多策略组合管理")
    st.caption("创建多策略组合，支持等权/风险平价/绩效加权分配")

    try:
        from src.engine.portfolio_manager import portfolio_manager
        from src.models import AllocationMethod

        # 创建组合
        with st.expander("➕ 创建新组合", expanded=False):
            # 策略多选
            all_strategy_ids = [s.strategy_id for s in strategies] if strategies else []
            selected_sids = st.multiselect(
                "选择策略 (至少2个)",
                all_strategy_ids,
                key="portfolio_strategy_select",
            )

            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                portfolio_name = st.text_input("组合名称", "我的组合", key="portfolio_name")
            with col_p2:
                method = st.selectbox(
                    "权重分配方式",
                    ["等权分配", "风险平价", "绩效加权"],
                    key="portfolio_method",
                )
            with col_p3:
                risk_budget = st.number_input(
                    "风险预算", 0.01, 0.50, 0.10, 0.01,
                    key="portfolio_risk_budget",
                )

            if st.button("创建组合", key="btn_create_portfolio"):
                if len(selected_sids) < 2:
                    st.warning("请至少选择 2 个策略")
                else:
                    method_map = {
                        "等权分配": AllocationMethod.EQUAL,
                        "风险平价": AllocationMethod.RISK_PARITY,
                        "绩效加权": AllocationMethod.PERFORMANCE,
                    }
                    try:
                        p = portfolio_manager.create_portfolio(
                            name=portfolio_name,
                            strategy_ids=selected_sids,
                            method=method_map[method],
                            risk_budget=risk_budget,
                        )
                        st.success(f"✅ 组合 '{p.name}' 创建成功! (ID: {p.portfolio_id})")
                        st.rerun()
                    except Exception as e:
                        st.error(f"创建失败: {e}")

        # 列出现有组合
        portfolios = portfolio_manager.list_portfolios()
        if portfolios:
            for p in portfolios:
                status = portfolio_manager.get_portfolio_status(p)
                with st.expander(f"📊 {p.name} ({p.portfolio_id}) - {status.allocation_method}"):
                    # 组合状态指标
                    col_s1, col_s2, col_s3 = st.columns(3)
                    col_s1.metric("策略数", str(status.strategy_count))
                    col_s2.metric("权重总和", f"{status.total_weight:.0%}")
                    col_s3.metric("上次再平衡", status.last_rebalance.strftime("%Y-%m-%d %H:%M") if status.last_rebalance else "未执行")

                    # 权重分布
                    if status.strategy_weights:
                        weight_df = pd.DataFrame([
                            {"策略ID": sid, "权重": w}
                            for sid, w in status.strategy_weights.items()
                        ])
                        fig_w = px.pie(
                            weight_df, values="权重", names="策略ID",
                            title=f"{p.name} 策略权重分布", hole=0.4,
                        )
                        fig_w.update_layout(height=300)
                        st.plotly_chart(fig_w, width='stretch')

                    # 再平衡操作
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        if st.button("🔄 重新分配权重", key=f"btn_realloc_{p.portfolio_id}"):
                            try:
                                new_weights = portfolio_manager.allocate_weights(p)
                                st.success(f"✅ 权重已更新: {new_weights}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"分配失败: {e}")
                    with col_r2:
                        if st.button("🗑️ 删除组合", key=f"btn_del_port_{p.portfolio_id}"):
                            if portfolio_manager.delete_portfolio(p.portfolio_id):
                                st.success("✅ 组合已删除")
                                st.rerun()
                            else:
                                st.error("删除失败")
        else:
            st.info("📭 暂无组合，请在上方创建")

    except Exception as e:
        st.info(f"组合管理模块未就绪: {e}")

    st.divider()

    # 策略详情选择
    st.subheader("🔍 策略详情")
    strat_names = []
    for s in strategies:
        if isinstance(s, dict):
            strat_names.append(s.get("name") or s.get("strategy_id", ""))
        else:
            strat_names.append(s.name or s.strategy_id)
    selected = st.selectbox("选择策略", strat_names)

    if selected:
        # 找到对应策略
        selected_strat = None
        for s in strategies:
            name = s.get("name") if isinstance(s, dict) else s.name
            sid = s.get("strategy_id") if isinstance(s, dict) else s.strategy_id
            if name == selected or sid == selected:
                selected_strat = s
                break

        if selected_strat:
            # 兼容 dict 和对象
            if isinstance(selected_strat, dict):
                _sid = selected_strat.get("strategy_id", "")
                _name = selected_strat.get("name", "")
                _model = selected_strat.get("model_type", "")
                _status = selected_strat.get("status", "")
                _pool = selected_strat.get("stock_pool", [])
                _factors = selected_strat.get("factors", [])
                _max_pos = selected_strat.get("max_position", 0)
            else:
                _sid = selected_strat.strategy_id
                _name = selected_strat.name
                _model = selected_strat.model_type
                _status = selected_strat.status
                _pool = selected_strat.stock_pool
                _factors = selected_strat.factors
                _max_pos = selected_strat.max_position

            col1, col2 = st.columns(2)
            with col1:
                st.metric("策略ID", _sid)
                st.metric("模型类型", _model)
                st.metric("状态", _status)
            with col2:
                st.metric("股票池", ", ".join(_pool) if _pool else "-")
                st.metric("因子数", str(len(_factors)))

                # 获取回测结果 (通过 cloud_client)
                cc = _get_cloud_client()
                bt = cc.get_backtest(_sid)
                grade = bt.get("grade", "未回测") if bt else "未回测"
                grade_icon = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🔴"}.get(grade, "⚪")
                st.metric("评级", f"{grade_icon} {grade}")

            # 回测结果图表
            if bt:
                st.divider()
                st.subheader("📊 回测绩效")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("年化收益", f"{bt.get('annual_return', 0):.1%}")
                col2.metric("夏普比率", f"{bt.get('sharpe_ratio', 0):.2f}")
                col3.metric("最大回撤", f"{bt.get('max_drawdown', 0):.1%}")
                col4.metric("胜率", f"{bt.get('win_rate', 0):.1%}")

                # 净值曲线
                equity_curve = bt.get("equity_curve", [])
                if equity_curve:
                    st.subheader("📈 净值曲线")
                    fig_equity = go.Figure()
                    fig_equity.add_trace(go.Scatter(
                        y=equity_curve,
                        mode="lines",
                        name="净值",
                        line=dict(color="blue", width=2),
                    ))
                    fig_equity.update_layout(
                        title="回测净值曲线",
                        xaxis_title="交易日",
                        yaxis_title="净值 ($)",
                        height=400,
                    )
                    st.plotly_chart(fig_equity, width='stretch')

                # 日收益率分布
                daily_returns = bt.get("daily_returns", [])
                if daily_returns:
                    st.subheader("📊 日收益率分布")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=daily_returns,
                        nbinsx=50,
                        name="日收益率",
                        marker_color="skyblue",
                    ))
                    fig_hist.update_layout(
                        title="日收益率分布",
                        xaxis_title="日收益率",
                        yaxis_title="频次",
                        height=300,
                    )
                    st.plotly_chart(fig_hist, width='stretch')
            else:
                st.info("该策略暂无回测结果")


# ═══════════════════════════════════════════════════════
# 页面4: 交易记录
# ═══════════════════════════════════════════════════════

def page_trades():
    """交易记录"""
    st.title("💰 交易记录")
    st.caption("历史订单、执行状态、风控检查记录")

    # 持仓快照
    st.subheader("📋 当前持仓")
    pos_data = get_positions_list()

    if pos_data and "positions" in pos_data and pos_data["positions"]:
        df = pd.DataFrame(pos_data["positions"])
        st.dataframe(df, width='stretch')

        # 汇总
        col1, col2, col3 = st.columns(3)
        col1.metric("持仓数量", str(pos_data.get("position_count", 0)))
        col2.metric("持仓市值", f"${pos_data.get('total_market_value', 0):,.2f}")
        col3.metric("浮动盈亏", f"${pos_data.get('total_unrealized_pnl', 0):,.2f}")
    else:
        st.info("📭 当前无持仓")

    st.divider()

    # 通知历史 (交易相关)
    st.subheader("📜 交易通知历史")
    notifications = get_notifications(30)
    if notifications:
        # 过滤交易相关通知
        trade_keywords = ["交易", "执行", "清仓", "止损", "买入", "卖出", "订单"]
        trade_notifs = [
            n for n in notifications
            if any(kw in n.get("title", "") or kw in n.get("message", "")
                   for kw in trade_keywords)
        ]
        if trade_notifs:
            icons = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌",
                     "CRITICAL": "🚨", "SUCCESS": "✅"}
            rows = []
            for n in reversed(trade_notifs):
                level = n.get("level", "INFO")
                rows.append({
                    "级别": f"{icons.get(level, 'ℹ️')} {level}",
                    "时间": n.get("timestamp", ""),
                    "标题": n.get("title", ""),
                    "内容": n.get("message", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("暂无交易相关通知")
    else:
        st.info("暂无通知记录")


# ═══════════════════════════════════════════════════════
# 页面5: 市场调研
# ═══════════════════════════════════════════════════════

def page_research():
    """市场调研"""
    st.title("🔍 市场调研")
    st.caption("最新调研报告、候选股票、新闻情绪、关注股票深度调研")

    config = _get_config()

    # ════════════════════════════════════════
    # 区域1: 全市场调研 (带主题输入)
    # ════════════════════════════════════════
    st.subheader("🌐 全市场调研")
    research_topic = st.text_input(
        "关注主题 (可选)",
        placeholder="如: AI芯片, 新能源, 半导体",
        key="research_topic_input",
        help="输入行业/主题关键词，调研时优先匹配相关股票",
    )
    if st.button("🔄 执行市场调研", type="primary", key="btn_research_execute"):
        try:
            registry = _get_agent_registry()
            research_agent = registry.get("research")
            if research_agent:
                with st.spinner("正在调研市场..."):
                    report = research_agent.run(
                        user_interest=research_topic.strip(),
                        max_candidates=config.MAX_CANDIDATES,
                    )
                    st.session_state["research_report"] = report
                st.success("✅ 市场调研完成!")
            else:
                st.error("Research Agent 未就绪")
        except Exception as e:
            st.error(f"调研失败: {e}")

    st.caption(f"候选上限: {config.MAX_CANDIDATES}")

    st.divider()

    # ════════════════════════════════════════
    # 区域2: 单股/多股深度调研
    # ════════════════════════════════════════
    st.subheader("🔎 单股深度调研")
    stock_input = st.text_input(
        "输入股票代码",
        placeholder="如: AAPL  或  AAPL,NVDA,TSLA  (逗号分隔多个)",
        key="research_stock_input",
    )
    if st.button("🔍 调研指定股票", type="secondary", key="btn_research_single"):
        if not stock_input.strip():
            st.warning("请输入股票代码")
        else:
            symbols = [s.strip().upper() for s in stock_input.split(",") if s.strip()]
            try:
                registry = _get_agent_registry()
                research_agent = registry.get("research")
                if research_agent:
                    with st.spinner(f"正在深度调研 {len(symbols)} 只股票..."):
                        single_results = []
                        for sym in symbols:
                            result = research_agent.research_single_stock(sym)
                            single_results.append(result)
                        st.session_state["research_single_results"] = single_results
                    st.success(f"✅ {len(single_results)} 只股票调研完成!")
                else:
                    st.error("Research Agent 未就绪")
            except Exception as e:
                st.error(f"调研失败: {e}")

    st.divider()

    # ════════════════════════════════════════
    # 区域3: 关注股票专属调研
    # ════════════════════════════════════════
    st.subheader("⭐ 我的关注股票调研")
    watchlist = config.WATCHLIST if config.WATCHLIST else config.DEFAULT_WATCHLIST
    watchlist_display = ", ".join(watchlist)
    st.caption(f"📋 关注列表 ({len(watchlist)} 只): {watchlist_display}")

    if st.button("⭐ 调研全部关注股票", type="secondary", key="btn_research_watchlist"):
        try:
            registry = _get_agent_registry()
            research_agent = registry.get("research")
            if research_agent:
                with st.spinner(f"正在调研 {len(watchlist)} 只关注股票... (可能需要30-60秒)"):
                    results = research_agent.research_watchlist()
                    st.session_state["research_watchlist_results"] = results
                st.success(f"✅ {len(results)} 只关注股票调研完成!")
            else:
                st.error("Research Agent 未就绪")
        except Exception as e:
            st.error(f"关注股票调研失败: {e}")

    st.divider()

    # ════════════════════════════════════════
    # 结果展示区
    # ════════════════════════════════════════

    # ── 3a: 单股深度调研结果 ──
    single_results = st.session_state.get("research_single_results")
    if single_results:
        st.subheader(f"🔎 单股调研结果 ({len(single_results)} 只)")
        for r in single_results:
            _render_single_stock_research(r)
        st.divider()

    # ── 3b: 关注股票调研结果 ──
    watchlist_results = st.session_state.get("research_watchlist_results")
    if watchlist_results:
        st.subheader(f"⭐ 关注股票调研结果 ({len(watchlist_results)} 只)")
        _render_watchlist_research(watchlist_results)
        st.divider()

    # ── 3c: 全市场调研结果 ──
    report = st.session_state.get("research_report")
    if report:
        # 市场情绪
        if hasattr(report, "market_sentiment"):
            sentiment = report.market_sentiment
            if sentiment > 0.3:
                sentiment_label = "🟢 偏乐观"
            elif sentiment > 0:
                sentiment_label = "🟡 微乐观"
            elif sentiment > -0.3:
                sentiment_label = "🟠 偏悲观"
            else:
                sentiment_label = "🔴 悲观"

            col1, col2 = st.columns(2)
            col1.metric("市场情绪", f"{sentiment_label} ({sentiment:+.2f})")
            col2.metric("候选股票数", str(len(report.candidate_stocks)))

        # 候选股票
        if hasattr(report, "candidate_stocks") and report.candidate_stocks:
            st.subheader("🎯 候选股票")
            candidates = []
            for c in report.candidate_stocks:
                candidates.append({
                    "代码": c.symbol,
                    "名称": c.name,
                    "行业": c.sector or "-",
                    "情绪分": f"{c.sentiment_score:+.2f}",
                    "提及次数": c.mention_count,
                    "推荐理由": c.plain_reason or "-",
                })
            st.dataframe(pd.DataFrame(candidates), width='stretch')

        # 关键事件
        if hasattr(report, "key_events") and report.key_events:
            st.subheader("📰 近期关键事件")
            for event in report.key_events[:5]:
                title = event.get("title", "") if isinstance(event, dict) else str(event)
                st.text(f"  • {title}")

        # 通俗总结
        if hasattr(report, "plain_summary") and report.plain_summary:
            st.divider()
            st.subheader("📝 调研总结")
            # 解析 plain_summary 文本，按行拆分为结构化表格
            summary_rows = []
            for line in report.plain_summary.strip().split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                # 尝试按 "标题: 内容" 格式拆分
                if ":" in stripped:
                    parts = stripped.split(":", 1)
                    summary_rows.append({"项目": parts[0].strip(), "内容": parts[1].strip()})
                else:
                    summary_rows.append({"项目": "", "内容": stripped})
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            else:
                st.text(report.plain_summary)

    # 如果没有任何结果，显示提示
    if not report and not single_results and not watchlist_results:
        st.info("💡 点击上方按钮执行市场调研、单股调研或关注股票调研")


def _render_single_stock_research(r: dict):
    """渲染单只股票的深度调研结果卡片"""
    sentiment = r.get("sentiment_score", 0)
    if sentiment > 0.3:
        sent_icon = "🟢"
    elif sentiment > 0:
        sent_icon = "🟡"
    elif sentiment > -0.3:
        sent_icon = "🟠"
    else:
        sent_icon = "🔴"

    with st.expander(
        f"{r['symbol']} - {r['name']} | {sent_icon} 情绪: {r.get('sentiment_label', '中性')} ({sentiment:+.2f})",
        expanded=True,
    ):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("代码", r["symbol"])
        col2.metric("名称", r["name"])
        col3.metric("行业", r.get("sector") or "-")
        col4.metric("情绪分", f"{sentiment:+.2f}")

        # 推荐理由
        if r.get("plain_reason"):
            st.info(f"💡 {r['plain_reason']}")

        # 公司概况
        profile = r.get("profile")
        if profile:
            with st.expander("🏢 公司详情", expanded=False):
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    st.metric("市值", f"${profile.get('marketCapitalization', 0):,.0f}" if profile.get('marketCapitalization') else "-")
                with col_p2:
                    st.metric("PE", f"{profile.get('pe', 0):.1f}" if profile.get('pe') else "-")
                with col_p3:
                    st.metric("股息率", f"{profile.get('dividend', 0):.2f}%" if profile.get('dividend') else "-")
                if profile.get("description"):
                    st.caption(profile["description"][:500] + ("..." if len(profile.get("description", "")) > 500 else ""))

        # 相关新闻
        news = r.get("news", [])
        if news:
            st.markdown("**📰 相关新闻:**")
            for n in news[:5]:
                headline = n.get("headline", "")
                source = n.get("source", "")
                url = n.get("url", "")
                if url:
                    st.markdown(f"  • [{headline}]({url}) _({source})_")
                else:
                    st.text(f"  • {headline} ({source})")
        else:
            st.caption("📭 暂无相关新闻")


def _render_watchlist_research(results: list[dict]):
    """渲染关注列表调研结果：表格汇总 + 可展开详情"""
    if not results:
        st.info("📭 暂无关注股票调研结果")
        return

    # 汇总表格
    rows = []
    for r in results:
        sentiment = r.get("sentiment_score", 0)
        if sentiment > 0.3:
            sent_icon = "🟢"
        elif sentiment > 0:
            sent_icon = "🟡"
        elif sentiment > -0.3:
            sent_icon = "🟠"
        else:
            sent_icon = "🔴"

        news_count = len(r.get("news", []))
        rows.append({
            "代码": r["symbol"],
            "名称": r.get("name", "-"),
            "行业": r.get("sector") or "-",
            "情绪": f"{sent_icon} {sentiment:+.2f}",
            "新闻数": news_count,
            "推荐理由": r.get("plain_reason", "-")[:80],
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # 可展开详情
    st.markdown("**📋 点击查看个股详情:**")
    for r in results:
        _render_single_stock_research(r)


# ═══════════════════════════════════════════════════════
# 页面7: 宏观指标
# ═══════════════════════════════════════════════════════

def page_macro():
    """宏观经济指标页面"""
    st.title("🌐 宏观经济指标")
    st.caption("美联储宏观数据 (FRED) · 美债收益率曲线 · 宏观风险评分")

    dc = _get_data_collector()

    # 云端模式下跳过本地 FRED 检查 (数据通过 cloud_client 获取)
    if not _is_cloud_mode():
        if not dc.fred.available:
            st.warning("⚠️ FRED API 未配置，无法获取宏观数据。请在 .env 中设置 FRED_API_KEY")
            return

    # ─── 宏观风险评分卡片 ───
    st.subheader("📊 宏观风险评分")
    with st.spinner("正在获取宏观数据..."):
        macro_data = get_macro_dashboard_data()

    if not macro_data:
        st.error("获取宏观数据失败，请检查 FRED API 配置")
        return

    risk_score = macro_data.get("risk_score", {})
    if risk_score:
        score = risk_score.get("score", 0)
        max_score = risk_score.get("max_score", 100)
        level = risk_score.get("level", "未知")
        color = risk_score.get("color", "gray")
        recommendation = risk_score.get("recommendation", "")
        signals = risk_score.get("signals", [])

        color_map = {
            "red": "#ef5350", "orange": "#ff9800",
            "yellow": "#fdd835", "green": "#4caf50",
        }
        bar_color = color_map.get(color, "#9e9e9e")

        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
        with col_r1:
            st.metric("风险评分", f"{score}/{max_score}")
        with col_r2:
            # 评分进度条
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, max_score]},
                    "bar": {"color": bar_color},
                    "steps": [
                        {"range": [0, 30], "color": "#e8f5e9"},
                        {"range": [30, 50], "color": "#fff9c4"},
                        {"range": [50, 70], "color": "#ffe0b2"},
                        {"range": [70, 100], "color": "#ffcdd2"},
                    ],
                },
                title={"text": f"风险等级: {level}"},
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, width='stretch')
        with col_r3:
            st.metric("风险等级", level)

        st.info(f"💡 建议: {recommendation}")

        if signals:
            st.markdown("**风险信号:**")
            for sig in signals:
                st.text(f"  {sig}")

    st.divider()

    # ─── 核心宏观指标表格 ───
    st.subheader("📋 核心宏观指标")
    category_names = {
        "通胀": "📈 通胀指标",
        "就业": "👷 就业指标",
        "增长": "📊 经济增长",
        "利率": "🏦 利率指标",
        "收益率": "📉 美债收益率",
    }

    for category, label in category_names.items():
        items = macro_data.get(category, [])
        if not items:
            continue

        with st.expander(f"{label} ({len(items)} 个指标)", expanded=True):
            rows = []
            for item in items:
                value = item.get("value", 0)
                unit = item.get("unit", "")
                mom = item.get("mom_change")
                yoy = item.get("yoy_pct")

                # 格式化数值
                if unit == "%":
                    val_str = f"{value:.2f}%"
                elif value > 1000:
                    val_str = f"{value:,.1f}"
                else:
                    val_str = f"{value:.3f}"

                # 环比变化
                if mom is not None:
                    mom_icon = "🔺" if mom > 0 else "🔻" if mom < 0 else "➖"
                    mom_str = f"{mom_icon} {mom:+.3f}"
                else:
                    mom_str = "-"

                # 同比变化
                if yoy is not None:
                    yoy_icon = "🔺" if yoy > 0 else "🔻" if yoy < 0 else "➖"
                    yoy_str = f"{yoy_icon} {yoy:+.2f}%"
                else:
                    yoy_str = "-"

                rows.append({
                    "指标": item.get("name", ""),
                    "最新值": val_str,
                    "单位": unit,
                    "环比": mom_str,
                    "同比": yoy_str,
                    "更新日期": item.get("date", ""),
                    "频率": item.get("frequency", ""),
                })

            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.divider()

    # ─── 美债收益率曲线 ───
    st.subheader("📉 美债收益率曲线")
    yield_curve = macro_data.get("yield_curve", {})
    if yield_curve:
        curve_data = yield_curve.get("curve", [])
        is_inverted = yield_curve.get("is_inverted", False)
        inversion_signal = yield_curve.get("inversion_signal", "")
        t10y2y = yield_curve.get("t10y2y_spread")
        t10y3m = yield_curve.get("t10y3m_spread")

        # 倒挂警告
        if is_inverted:
            st.error(f"🚨 {inversion_signal} - 收益率倒挂是经济衰退的领先指标!")
        else:
            st.success(f"✅ {inversion_signal}")

        col_y1, col_y2 = st.columns(2)
        with col_y1:
            st.metric("10Y-2Y 利差", f"{t10y2y:.3f}%" if t10y2y is not None else "N/A")
        with col_y2:
            st.metric("10Y-3M 利差", f"{t10y3m:.3f}%" if t10y3m is not None else "N/A")

        # 收益率曲线图
        if curve_data:
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=[d["tenor"] for d in curve_data],
                y=[d["yield"] for d in curve_data],
                mode="lines+markers",
                name="收益率",
                line=dict(color="blue", width=3),
                marker=dict(size=10),
                text=[f"{d['yield']:.3f}%" for d in curve_data],
                textposition="top center",
            ))
            fig_curve.update_layout(
                title="美债收益率曲线 (当前)",
                xaxis_title="期限",
                yaxis_title="收益率 (%)",
                height=400,
                yaxis=dict(range=[0, max(d["yield"] for d in curve_data) + 1]),
            )
            st.plotly_chart(fig_curve, width='stretch')

    st.divider()

    # ─── VIX 恐慌指数 ───
    st.subheader("😱 VIX 恐慌指数")
    try:
        cc = _get_cloud_client()
        vix_data = cc.get_vix()
        if vix_data:
            col_v1, col_v2, col_v3, col_v4 = st.columns(4)
            col_v1.metric("VIX 值", f"{vix_data['value']:.2f}")
            col_v2.metric("涨跌", f"{vix_data['change']:+.2f}")
            col_v3.metric("涨跌幅", f"{vix_data['change_pct']:+.2f}%")
            col_v4.metric("风险等级", vix_data["risk_level"])

            color_map = {"green": "🟢", "lightgreen": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴"}
            st.info(f"{color_map.get(vix_data['risk_color'], '⚪')} VIX 风险等级: {vix_data['risk_level']} | "
                    f"数据时间: {vix_data.get('timestamp', 'N/A')}")
        else:
            st.info("获取 VIX 数据失败")
    except Exception as e:
        st.warning(f"VIX 数据获取失败: {e}")

    st.divider()

    # ─── 数据源信息 ───
    st.subheader("📡 数据源信息")
    st.text("数据来源: FRED (Federal Reserve Economic Data)")
    config = _get_config()
    fred_key_display = f"***{config.FRED_API_KEY[-4:]}" if config.has_fred_key() else "未配置"
    st.text(f"API Key: {fred_key_display}")
    st.text("缓存策略: 宏观数据6小时, 收益率曲线12小时, VIX 5分钟")
    st.caption("⚠️ 宏观数据有发布延迟，CPI/就业等月度指标通常滞后1-2个月")


# ═══════════════════════════════════════════════════════
# 页面8: 期权数据
# ═══════════════════════════════════════════════════════

def page_options():
    """期权数据页面"""
    st.title("📊 期权数据")
    st.caption("期权链 · PCR比率 · 隐含波动率 · 到期日选择 (Yahoo Finance)")

    dc = _get_data_collector()

    # ─── 标的选择 ───
    st.subheader("🔍 选择标的")
    col_s1, col_s2 = st.columns([1, 2])

    with col_s1:
        symbol_input = st.text_input(
            "股票代码",
            value="AAPL",
            key="options_symbol_input",
            help="输入美股代码，如 AAPL, TSLA, NVDA"
        ).upper().strip()

    # 获取期权链数据
    if not symbol_input:
        st.warning("请输入股票代码")
        return

    with st.spinner(f"正在获取 {symbol_input} 期权链数据..."):
        # 先获取一次以拿到到期日列表
        opt_data = get_options_chain_data(symbol_input, "")

    if not opt_data:
        st.error(f"获取 {symbol_input} 期权链失败，可能该股票没有期权数据")
        return

    expiration_dates = opt_data.get("expiration_dates", [])
    selected_exp = opt_data.get("selected_expiration", "")

    with col_s2:
        if expiration_dates:
            selected_exp = st.selectbox(
                "选择到期日",
                expiration_dates,
                index=0,
                key="options_exp_select",
                format_func=lambda x: x,
            )

    # 如果用户选了不同的到期日，重新获取
    if selected_exp and selected_exp != opt_data.get("selected_expiration"):
        with st.spinner(f"正在获取 {selected_exp} 到期日期权数据..."):
            opt_data = get_options_chain_data(symbol_input, selected_exp)

    if not opt_data or "calls" not in opt_data:
        st.error("期权数据不可用")
        return

    # ─── 期权概况卡片 ───
    st.divider()
    st.subheader("📋 期权概况")

    underlying_price = opt_data.get("underlying_price", 0)
    pcr_volume = opt_data.get("pcr_volume", 0)
    pcr_oi = opt_data.get("pcr_oi", 0)
    atm_iv = opt_data.get("atm_iv", 0)
    summary = opt_data.get("summary", {})

    col_o1, col_o2, col_o3, col_o4, col_o5 = st.columns(5)
    col_o1.metric("标的现价", f"")
    col_o2.metric("PCR (成交量)", f"{pcr_volume:.3f}", help="Put-Call Ratio < 1 偏多, > 1 偏空")
    col_o3.metric("PCR (持仓量)", f"{pcr_oi:.3f}", help="基于未平仓合约量的 PCR")
    col_o4.metric("ATM 隐含波动率", f"{atm_iv:.2f}%", help="平值期权隐含波动率")
    col_o5.metric("到期日", selected_exp)

    # PCR 解读
    if pcr_volume < 0.5:
        pcr_label = "🟢 极度看多"
    elif pcr_volume < 0.8:
        pcr_label = "🟢 偏多"
    elif pcr_volume < 1.0:
        pcr_label = "🟡 中性偏多"
    elif pcr_volume < 1.2:
        pcr_label = "🟡 中性偏空"
    elif pcr_volume < 1.5:
        pcr_label = "🟠 偏空"
    else:
        pcr_label = "🔴 极度看空"

    st.info(f"📊 PCR (成交量) 解读: {pcr_label} | "
            f"看涨 {summary.get('call_volume', 0):,} vs 看跌 {summary.get('put_volume', 0):,}")

    st.divider()

    # ─── 看涨期权表格 ───
    calls_df = opt_data.get("calls")
    puts_df = opt_data.get("puts")

    # 将 DataFrame 转换为可显示的格式
    if isinstance(calls_df, pd.DataFrame) and not calls_df.empty:
        # 选择关键列
        call_cols_map = {
            "strike": "行权价", "lastPrice": "最新价", "bid": "买价", "ask": "卖价",
            "volume": "成交量", "openInterest": "未平仓", "impliedVolatility": "隐含波动率(%)",
            "change": "涨跌", "percentChange": "涨跌幅(%)", "inTheMoney": "实值",
        }
        available_call_cols = [c for c in call_cols_map if c in calls_df.columns]
        if available_call_cols:
            display_calls = calls_df[available_call_cols].rename(columns=call_cols_map).copy()
            st.subheader(f"📈 看涨期权 - {selected_exp} ({len(display_calls)} 个合约)")
            st.dataframe(display_calls, width='stretch', hide_index=True)

            # 看涨期权成交量分布图 (Top 20)
            if "volume" in calls_df.columns and "strike" in calls_df.columns:
                top_calls = calls_df.nlargest(20, "volume")[["strike", "volume", "openInterest"]].copy()
                if not top_calls.empty:
                    fig_call_vol = go.Figure()
                    fig_call_vol.add_trace(go.Bar(
                        x=top_calls["strike"],
                        y=top_calls["volume"],
                        name="成交量",
                        marker_color="green",
                    ))
                    fig_call_vol.add_trace(go.Bar(
                        x=top_calls["strike"],
                        y=top_calls["openInterest"],
                        name="未平仓",
                        marker_color="lightgreen",
                    ))
                    fig_call_vol.update_layout(
                        title="看涨期权 Top 20 成交量 & 未平仓 (按行权价)",
                        xaxis_title="行权价",
                        yaxis_title="数量",
                        barmode="group",
                        height=350,
                    )
                    st.plotly_chart(fig_call_vol, width='stretch')
    else:
        st.warning("无看涨期权数据")

    st.divider()

    # ─── 看跌期权表格 ───
    if isinstance(puts_df, pd.DataFrame) and not puts_df.empty:
        put_cols_map = {
            "strike": "行权价", "lastPrice": "最新价", "bid": "买价", "ask": "卖价",
            "volume": "成交量", "openInterest": "未平仓", "impliedVolatility": "隐含波动率(%)",
            "change": "涨跌", "percentChange": "涨跌幅(%)", "inTheMoney": "实值",
        }
        available_put_cols = [c for c in put_cols_map if c in puts_df.columns]
        if available_put_cols:
            display_puts = puts_df[available_put_cols].rename(columns=put_cols_map).copy()
            st.subheader(f"📉 看跌期权 - {selected_exp} ({len(display_puts)} 个合约)")
            st.dataframe(display_puts, width='stretch', hide_index=True)

            # 看跌期权成交量分布图 (Top 20)
            if "volume" in puts_df.columns and "strike" in puts_df.columns:
                top_puts = puts_df.nlargest(20, "volume")[["strike", "volume", "openInterest"]].copy()
                if not top_puts.empty:
                    fig_put_vol = go.Figure()
                    fig_put_vol.add_trace(go.Bar(
                        x=top_puts["strike"],
                        y=top_puts["volume"],
                        name="成交量",
                        marker_color="red",
                    ))
                    fig_put_vol.add_trace(go.Bar(
                        x=top_puts["strike"],
                        y=top_puts["openInterest"],
                        name="未平仓",
                        marker_color="lightcoral",
                    ))
                    fig_put_vol.update_layout(
                        title="看跌期权 Top 20 成交量 & 未平仓 (按行权价)",
                        xaxis_title="行权价",
                        yaxis_title="数量",
                        barmode="group",
                        height=350,
                    )
                    st.plotly_chart(fig_put_vol, width='stretch')
    else:
        st.warning("无看跌期权数据")

    st.divider()

    # ─── 隐含波动率微笑曲线 ───
    st.subheader("😊 隐含波动率微笑曲线")
    if (isinstance(calls_df, pd.DataFrame) and not calls_df.empty and
        isinstance(puts_df, pd.DataFrame) and not puts_df.empty and
        "strike" in calls_df.columns and "impliedVolatility" in calls_df.columns and
        "strike" in puts_df.columns and "impliedVolatility" in puts_df.columns):

        fig_smile = go.Figure()

        # 看涨 IV
        call_iv = calls_df[["strike", "impliedVolatility"]].dropna()
        if not call_iv.empty:
            fig_smile.add_trace(go.Scatter(
                x=call_iv["strike"],
                y=call_iv["impliedVolatility"],
                mode="lines+markers",
                name="看涨 IV",
                line=dict(color="green", width=2),
            ))

        # 看跌 IV
        put_iv = puts_df[["strike", "impliedVolatility"]].dropna()
        if not put_iv.empty:
            fig_smile.add_trace(go.Scatter(
                x=put_iv["strike"],
                y=put_iv["impliedVolatility"],
                mode="lines+markers",
                name="看跌 IV",
                line=dict(color="red", width=2),
            ))

        # 标记当前价格
        if underlying_price > 0:
            fig_smile.add_vline(
                x=underlying_price,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"现价 ",
            )

        fig_smile.update_layout(
            title=f"{symbol_input} 隐含波动率曲线 - {selected_exp}",
            xaxis_title="行权价",
            yaxis_title="隐含波动率 (%)",
            height=400,
        )
        st.plotly_chart(fig_smile, width='stretch')
    else:
        st.info("隐含波动率数据不完整")

    st.divider()

    # ─── 期权希腊字母说明 ───
    with st.expander("📖 期权数据解读指南", expanded=False):
        st.markdown("""
        **PCR (Put-Call Ratio) 看跌看涨比率:**
        - PCR < 0.5: 极度看多 (市场情绪非常乐观)
        - PCR 0.5-0.8: 偏多
        - PCR 0.8-1.0: 中性偏多
        - PCR 1.0-1.2: 中性偏空
        - PCR 1.2-1.5: 偏空
        - PCR > 1.5: 极度看空 (可能为过度恐慌，反而视为反向指标)

        **隐含波动率 (IV):**
        - IV 越高表示市场预期未来波动越大
        - IV 低于 20% 为低波动环境
        - IV 高于 50% 为高波动环境
        - 隐含波动率微笑曲线显示不同行权价的 IV 差异

        **成交量 vs 未平仓量:**
        - 成交量: 当日新交易的合约数
        - 未平仓量: 尚未结算的合约总数
        - 高未平仓量的行权价通常是关键支撑/阻力位

        ⚠️ 期权数据仅供参考，不构成投资建议。
        """)


# ═══════════════════════════════════════════════════════
# 页面9: 加密货币 (Phase 8 新增)
# ═══════════════════════════════════════════════════════

def page_crypto():
    """加密货币交易面板"""
    st.title("🪙 加密货币")
    st.caption("加密货币持仓、风控状态、价格看板 (24/7 交易)")

    config = _get_config()

    # ── 模块状态检查 ──
    if not config.has_crypto_enabled():
        st.warning(
            "⚠️ 加密货币模块未启用\n\n"
            "请在 `.env` 文件中设置 `CRYPTO_ENABLED=true` 并重启系统。"
        )
        return

    # ── 加密货币配置概览 ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔄 数据源", config.CRYPTO_DATA_SOURCE.upper())
    col2.metric("📊 交易对数", len(config.get_crypto_pairs()))
    col3.metric("🛡️ 单币仓位上限", f"{config.CRYPTO_MAX_POSITION_PCT:.0%}")
    col4.metric("🚫 止损线", f"-{config.CRYPTO_SINGLE_STOP_LOSS:.0%}")

    st.divider()

    # ── 账户概览 ──
    st.subheader("💰 账户概览")
    try:
        from src.execution.alpaca_client import alpaca_client
        if alpaca_client.available:
            account = alpaca_client.get_account()
            if account:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("💼 总资产", f"")
                col2.metric("💵 现金", f"")
                col3.metric("💪 购买力", f"")
                mode_label = "📝 模拟盘" if account.trading_mode.value == "paper" else "🔴 实盘"
                col4.metric("📋 交易模式", mode_label)
            else:
                st.info("💡 无法获取账户信息")
        else:
            st.info("💡 Alpaca 未配置, 无法获取账户数据")
    except Exception as e:
        st.warning(f"获取账户信息异常: {e}")

    st.divider()

    # ── 加密货币持仓 ──
    st.subheader("📈 加密货币持仓")
    try:
        cc = _get_cloud_client()
        crypto_data = cc.get_crypto_positions()
        crypto_positions = crypto_data.get("positions", [])
        account_data = crypto_data.get("account")

        if account_data:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💼 总资产", f"")
            col2.metric("💵 现金", f"")
            col3.metric("💪 购买力", f"")
            mode_label = account_data.get("trading_mode", "📝 模拟盘")
            col4.metric("📋 交易模式", mode_label)

        if crypto_positions:
            pos_data = crypto_positions
            df_pos = pd.DataFrame(pos_data)

            col_left, col_right = st.columns([1, 1])

            with col_left:
                # 持仓饼图
                fig_pie = px.pie(
                    df_pos,
                    values="market_value",
                    names="symbol",
                    title="加密货币持仓分布",
                    hole=0.4,
                )
                fig_pie.update_layout(height=350)
                st.plotly_chart(fig_pie, width='stretch')

            with col_right:
                # 持仓明细表
                st.dataframe(
                    df_pos[["symbol", "qty", "avg_cost", "market_value",
                            "unrealized_pnl", "unrealized_pnl_pct", "weight"]]
                    .style.format({
                        "qty": "{:.6f}",
                        "avg_cost": "",
                        "market_value": "",
                        "unrealized_pnl": "",
                        "unrealized_pnl_pct": "{:.1%}",
                        "weight": "{:.1%}",
                    }),
                    width='stretch',
                    height=350,
                )

            # 浮动盈亏柱状图
            st.subheader("📊 各币种浮动盈亏")
            fig_bar = go.Figure()
            colors = ["green" if p["unrealized_pnl"] >= 0 else "red" for p in pos_data]
            fig_bar.add_trace(go.Bar(
                x=[p["symbol"] for p in pos_data],
                y=[p["unrealized_pnl"] for p in pos_data],
                marker_color=colors,
                text=[f"" for p in pos_data],
                textposition="auto",
            ))
            fig_bar.update_layout(
                title="加密货币浮动盈亏 ($)",
                xaxis_title="币种",
                yaxis_title="盈亏 ($)",
                height=300,
            )
            st.plotly_chart(fig_bar, width='stretch')
        else:
            if not account_data:
                st.info("💡 Alpaca 未配置, 无法获取账户数据")
            else:
                st.info(
                    "📭 当前无加密货币持仓\n\n"
                    "💡 你可以在下方「🛒 手动交易」面板中买入加密货币，或运行加密货币交易循环自动建仓。"
                )
    except Exception as e:
        st.warning(f"获取持仓信息异常: {e}")

    st.divider()

    # ── 价格看板 (增强版: 24h涨跌幅 + 成交量 + 缓存) ──
    st.subheader("💲 价格看板")
    price_results = get_crypto_prices_with_change()
    if price_results:
        # 统计可用数量
        available_count = sum(1 for r in price_results if r["available"])
        unavailable_count = len(price_results) - available_count

        col_info1, col_info2 = st.columns(2)
        col_info1.caption(f"✅ {available_count} 个交易对可获取价格")
        if unavailable_count > 0:
            col_info2.caption(f"❌ {unavailable_count} 个交易对不可用")

        # 构建展示数据
        price_data = []
        for r in price_results:
            if r["available"]:
                change_str = f"{r['change_pct']:+.2f}%"
                vol_str = f"${r['volume_24h']:,.0f}" if r["volume_24h"] > 0 else "-"
                price_data.append({
                    "币种": r["symbol"],
                    "最新价格 ($)": r["price"],
                    "24h涨跌幅": change_str,
                    "24h成交量": vol_str,
                    "状态": "🟢 正常",
                })
            else:
                price_data.append({
                    "币种": r["symbol"],
                    "最新价格 ($)": None,
                    "24h涨跌幅": "-",
                    "24h成交量": "-",
                    "状态": "❌ 不可用",
                })

        df_prices = pd.DataFrame(price_data)

        # 24h涨跌幅红绿色标注
        def _color_change(val):
            if isinstance(val, str) and val.startswith("+"):
                return "color: #4caf50; font-weight: bold"
            elif isinstance(val, str) and val.startswith("-"):
                return "color: #ef5350; font-weight: bold"
            return "color: gray"

        styled = df_prices.style.format(
            {"最新价格 ($)": lambda v: f"${v:,.2f}" if v is not None else "N/A"},
            na_rep="N/A",
        )
        styled = styled.map(_color_change, subset=["24h涨跌幅"])
        st.dataframe(styled, width='stretch', hide_index=True, height=320)

        # 24h涨跌幅柱状图
        available_prices = [r for r in price_results if r["available"]]
        if available_prices:
            fig_change = go.Figure()
            colors = ["#4caf50" if r["change_pct"] >= 0 else "#ef5350" for r in available_prices]
            fig_change.add_trace(go.Bar(
                x=[r["symbol"] for r in available_prices],
                y=[r["change_pct"] for r in available_prices],
                marker_color=colors,
                text=[f"{r['change_pct']:+.2f}%" for r in available_prices],
                textposition="auto",
            ))
            fig_change.update_layout(
                title="24h 涨跌幅 (%)",
                xaxis_title="币种",
                yaxis_title="涨跌幅 (%)",
                height=300,
            )
            st.plotly_chart(fig_change, width='stretch')
    else:
        st.info("💡 暂无可用价格数据，请检查 Alpaca API 配置")

    st.divider()

    # ── 手动交易面板 ──
    st.subheader("🛒 手动交易")
    st.caption("手动买入/卖出加密货币 (Paper Trading 模拟盘)")

    try:
        from src.execution.alpaca_client import alpaca_client

        if not alpaca_client.available:
            st.info("💡 Alpaca 未配置，无法进行交易")
        else:
            col_trade1, col_trade2, col_trade3, col_trade4 = st.columns([2, 2, 1, 1])

            with col_trade1:
                # 交易对选择 (优先显示可用的)
                trade_pairs = [r["symbol"] for r in price_results if r["available"]] if price_results else config.get_crypto_pairs()
                selected_pair = st.selectbox("交易对", trade_pairs, key="crypto_trade_pair")

            with col_trade2:
                trade_qty = st.number_input(
                    "数量",
                    min_value=0.0001,
                    max_value=10000.0,
                    value=0.001,
                    step=0.001,
                    format="%.4f",
                    key="crypto_trade_qty",
                )

            with col_trade3:
                # 显示当前价格
                current_price = None
                for r in price_results:
                    if r["symbol"] == selected_pair and r["available"]:
                        current_price = r["price"]
                        break
                if current_price:
                    st.metric("当前价格", f"${current_price:,.2f}")
                    est_value = current_price * trade_qty
                    st.caption(f"预估金额: ${est_value:,.2f}")
                    # 最小订单金额校验提示
                    if est_value < 10.0:
                        st.warning(f"⚠️ 预估金额 ${est_value:.2f} < $10, Alpaca 最小订单金额为 $10, 请增加数量")
                else:
                    st.metric("当前价格", "N/A")

            with col_trade4:
                st.write("")  # 占位对齐
                st.write("")

            # 买入/卖出按钮
            col_buy, col_sell, col_info = st.columns([1, 1, 2])

            with col_buy:
                if st.button("📈 买入", type="primary", key="btn_crypto_buy"):
                    if not selected_pair or trade_qty <= 0:
                        st.warning("请选择交易对并输入有效数量")
                    else:
                        # 前置校验: 最小订单金额
                        if current_price and current_price * trade_qty < 10.0:
                            st.error(f"❌ 订单金额 ${current_price * trade_qty:.2f} < $10 (Alpaca 最小限额), 请增加数量")
                        else:
                            with st.spinner(f"正在买入 {trade_qty} {selected_pair}..."):
                                try:
                                    order = alpaca_client.submit_crypto_order(
                                        symbol=selected_pair,
                                        qty=trade_qty,
                                        side="buy",
                                        price=current_price or 0,
                                    )
                                    if order:
                                        st.success(f"✅ 买入订单已提交! ID: {order.order_id}")
                                        st.info(f"📊 {selected_pair} × {trade_qty:.4f} @ 市价")
                                    else:
                                        st.error("订单提交失败，请检查 API 配置")
                                except Exception as e:
                                    st.error(f"买入失败: {e}")

            with col_sell:
                if st.button("📉 卖出", type="secondary", key="btn_crypto_sell"):
                    if not selected_pair or trade_qty <= 0:
                        st.warning("请选择交易对并输入有效数量")
                    else:
                        # 前置校验: 最小订单金额
                        if current_price and current_price * trade_qty < 10.0:
                            st.error(f"❌ 订单金额 ${current_price * trade_qty:.2f} < $10 (Alpaca 最小限额), 请增加数量")
                        else:
                            with st.spinner(f"正在卖出 {trade_qty} {selected_pair}..."):
                                try:
                                    order = alpaca_client.submit_crypto_order(
                                        symbol=selected_pair,
                                        qty=trade_qty,
                                        side="sell",
                                        price=current_price or 0,
                                    )
                                    if order:
                                        st.success(f"✅ 卖出订单已提交! ID: {order.order_id}")
                                        st.info(f"📊 {selected_pair} × {trade_qty:.4f} @ 市价")
                                    else:
                                        st.error("订单提交失败，请检查 API 配置")
                                except Exception as e:
                                    st.error(f"卖出失败: {e}")

            with col_info:
                st.caption("💡 交易在 Alpaca Paper Trading 模拟盘执行，不涉及真实资金")
                st.caption("⚠️ 加密货币 24/7 交易，订单使用 GTC (撤销前有效)")

    except Exception as e:
        st.warning(f"手动交易面板加载异常: {e}")
        st.warning(f"获取价格数据异常: {e}")

    st.divider()

    # ── 风控状态 ──
    st.subheader("🛡️ 加密货币风控")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("单币仓位上限", f"{config.CRYPTO_MAX_POSITION_PCT:.0%}")
    col2.metric("日最大亏损", f"{config.CRYPTO_MAX_DAILY_LOSS:.0%}")
    col3.metric("最大回撤", f"{config.CRYPTO_MAX_DRAWDOWN:.0%}")
    col4.metric("日交易次数上限", f"{config.CRYPTO_DAILY_TRADE_LIMIT}")

    col1, col2, col3 = st.columns(3)
    col1.metric("止损线", f"-{config.CRYPTO_SINGLE_STOP_LOSS:.0%}")
    col2.metric("最大持仓数", f"{config.CRYPTO_MAX_TOTAL_POSITIONS}")
    col3.metric("Paper 最少天数", f"{config.CRYPTO_MIN_PAPER_DAYS} 天")

    # 风控检查
    try:
        from src.execution.risk_guard import risk_guard
        from src.execution.alpaca_client import alpaca_client
        from src.models import AssetType

        if alpaca_client.available:
            account = alpaca_client.get_account()
            all_positions = alpaca_client.get_positions()
            crypto_positions = [
                p for p in all_positions
                if "/" in p.symbol or p.asset_type == AssetType.CRYPTO
            ]
            if account:
                risk_status = risk_guard.check_crypto_portfolio(account, crypto_positions)
                if risk_status.is_alert:
                    st.error("⚠️ 风控告警!")
                    for alert in risk_status.alerts:
                        st.warning(f"• {alert}")
                else:
                    st.success("✅ 风控正常, 无告警")
    except Exception as e:
        st.info(f"风控检查不可用: {e}")

    st.divider()

    # ── 交易对列表 ──
    st.subheader("📋 交易对列表")
    pairs = config.get_crypto_pairs()
    cols = st.columns(min(len(pairs), 4))
    for i, pair in enumerate(pairs):
        cols[i % len(cols)].code(pair)

    st.divider()

    # ── 快速操作 ──
    st.subheader("🚀 快速操作")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 运行加密货币交易循环", help="执行: 调研→信号→风控→执行→监控"):
            with st.spinner("正在执行加密货币交易循环..."):
                try:
                    from src.workflow.engine import workflow_engine
                    result = workflow_engine.run_crypto_loop()
                    if result.get("status") == "success":
                        st.success("✅ 交易循环执行完成!")
                    else:
                        st.warning(f"⚠️ 执行完成 (有异常): {result.get('status')}")
                    # 显示执行摘要
                    for step_name, step_status in result.get("steps", {}).items():
                        st.text(f"  {step_name}: {step_status}")
                except Exception as e:
                    st.error(f"执行失败: {e}")

    with col2:
        if st.button("📊 刷新加密货币快照", help="获取最新的持仓和风控状态"):
            st.rerun()


# ═══════════════════════════════════════════════════════
# 页面10: 系统设置
# ═══════════════════════════════════════════════════════

def page_settings():
    """系统设置"""
    st.title("⚙️ 系统设置")
    st.caption("配置查看、工作流手动触发、紧急止损、实盘迁移")

    config = _get_config()

  

    # 选股配置（可编辑）
    st.subheader("🎯 选股配置（运行时修改）")
    col1, col2 = st.columns(2)
    with col1:
        new_max_candidates = st.number_input(
            "候选股票上限",
            min_value=1, max_value=50,
            value=config.MAX_CANDIDATES,
            step=1,
            help="调研阶段最多发现多少只候选股票",
        )
    with col2:
        new_top_k = st.number_input(
            "信号Top-K",
            min_value=1, max_value=50,
            value=config.TOP_K_SIGNALS,
            step=1,
            help="信号生成阶段选前K只做多",
        )

    if st.button("💾 保存选股配置"):
        config.MAX_CANDIDATES = int(new_max_candidates)
        config.TOP_K_SIGNALS = int(new_top_k)
        st.success(f"✅ 已更新: 候选上限={config.MAX_CANDIDATES}, Top-K={config.TOP_K_SIGNALS}")

    st.divider()

    # 工作流手动触发
    st.subheader("🔄 工作流管理")
    workflows = get_workflow_names()
    if workflows:
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_wf = st.selectbox("选择工作流", workflows)
        with col2:
            st.write("")  # 占位
            st.write("")  # 占位

        user_input = st.text_input("输入参数 (可选)", "")

        if st.button("▶️ 运行工作流", type="primary"):
            with st.spinner(f"正在运行 {selected_wf}..."):
                try:
                    we = _get_workflow_engine()
                    kwargs = {}
                    if user_input:
                        kwargs["user_interest"] = user_input
                    result = we.run(selected_wf, **kwargs)
                    st.success(f"✅ 工作流完成: {result.get('status', 'unknown')}")

                    # 显示步骤结果
                    steps = result.get("steps", {})
                    if steps:
                        st.subheader("步骤执行结果")
                        for step_name, step_result in steps.items():
                            if step_result is not None:
                                st.text(f"  ✅ {step_name}: 已完成")
                            else:
                                st.text(f"  ❌ {step_name}: 失败或跳过")
                except Exception as e:
                    st.error(f"工作流执行失败: {e}")
    else:
        st.info("工作流引擎未初始化")

    st.divider()

    # 紧急止损
    st.subheader("🚨 紧急操作")
    st.warning("⚠️ 以下操作不可撤销，请谨慎执行!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚨 紧急清仓", type="secondary"):
            st.session_state["confirm_emergency"] = True

    with col2:
        if st.button("📊 运行监控快照"):
            with st.spinner("正在检查持仓..."):
                try:
                    registry = _get_agent_registry()
                    monitor = registry.get("monitor")
                    if monitor:
                        result = monitor.run(mode="snapshot")
                        st.success("✅ 监控完成")
                        if isinstance(result, dict):
                            alerts = result.get("alerts", [])
                            if alerts:
                                st.warning(f"检测到 {len(alerts)} 条告警:")
                                for a in alerts:
                                    st.text(f"  ⚠️ {a}")
                            else:
                                st.info("无告警，组合运行正常")
                    else:
                        st.error("Monitor Agent 未就绪")
                except Exception as e:
                    st.error(f"监控失败: {e}")

    # 紧急清仓确认
    if st.session_state.get("confirm_emergency"):
        st.error("确认要清仓所有持仓吗？此操作不可撤销!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 确认清仓"):
                with st.spinner("正在执行紧急清仓..."):
                    try:
                        we = _get_workflow_engine()
                        result = we.run_emergency_stop()
                        st.success("✅ 紧急清仓完成")
                        st.json(result)
                    except Exception as e:
                        st.error(f"紧急清仓失败: {e}")
                st.session_state["confirm_emergency"] = False
        with col2:
            if st.button("❌ 取消"):
                st.session_state["confirm_emergency"] = False
                st.rerun()

    st.divider()

    # 实盘迁移面板
    st.subheader("🔴 实盘迁移管理")
    st.caption("Paper Trading -> Live Trading 安全迁移 (6 项预检 + 渐进式仓位)")

    try:
        from src.execution.live_migration import live_migration_manager

        # 当前迁移状态
        mig_status = live_migration_manager.get_migration_status()
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("当前模式", "📝 模拟盘" if mig_status.get("mode") == "paper" else "🔴 实盘")
        col_m2.metric("仓位系数", f"{mig_status.get('current_scale_factor', 1.0):.0%}")
        col_m3.metric("迁移阶段", mig_status.get("phase", "未迁移"))

        # 迁移就绪检查
        with st.expander("🔍 迁移就绪检查", expanded=not mig_status.get("migrated", False)):
            if st.button("📋 执行预检", key="btn_migration_check"):
                with st.spinner("正在执行 6 项预检..."):
                    try:
                        from datetime import timedelta
                        # 模拟预检数据 (实际环境中应从系统获取)
                        paper_start = datetime.now() - timedelta(days=45) if config.is_paper_trading() else None
                        status = live_migration_manager.check_readiness(
                            paper_start_date=paper_start,
                            strategies=get_strategies(),
                            risk_alerts=[],
                            current_drawdown=0.05,
                            daily_trade_count=3,
                        )
                        st.session_state["migration_check"] = status
                    except Exception as e:
                        st.error(f"预检失败: {e}")

            check_result = st.session_state.get("migration_check")
            if check_result:
                checks = check_result.to_dict()["checks"] if hasattr(check_result, "to_dict") else check_result.get("checks", [])
                for c in checks:
                    icon = "✅" if c["passed"] else "❌"
                    st.text(f"  {icon} {c['name']}: {c['detail']}")

                if check_result.all_passed if hasattr(check_result, "all_passed") else check_result.get("all_passed"):
                    st.success("✅ 所有预检通过，可以安全迁移!")
                else:
                    st.warning("⚠️ 部分预检未通过，请先解决以上问题")

        # 迁移操作
        if not mig_status.get("migrated", False):
            col_mig1, col_mig2 = st.columns(2)
            with col_mig1:
                if st.button("🔴 迁移到实盘", type="secondary", key="btn_migrate"):
                    st.session_state["confirm_migrate"] = True
            with col_mig2:
                st.caption("⚠️ 迁移需通过 6 项预检")

            if st.session_state.get("confirm_migrate"):
                st.error("⚠️ 确认要迁移到实盘交易吗？此操作将切换到真实资金交易!")
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    if st.button("✅ 确认迁移", key="btn_confirm_migrate"):
                        with st.spinner("正在执行迁移..."):
                            try:
                                result = live_migration_manager.migrate(confirm=True)
                                if result.get("success"):
                                    st.success("✅ 迁移成功! 进入渐进式仓位阶段")
                                    st.rerun()
                                else:
                                    st.error(f"迁移失败: {result.get('reason', '未知')}")
                            except Exception as e:
                                st.error(f"迁移失败: {e}")
                        st.session_state["confirm_migrate"] = False
                with col_c2:
                    if st.button("❌ 取消", key="btn_cancel_migrate"):
                        st.session_state["confirm_migrate"] = False
                        st.rerun()
        else:
            st.info(f"📌 已迁移到实盘 | 迁移时间: {mig_status.get('migration_date', 'N/A')[:19]}")
            if st.button("📋 回滚到模拟盘", type="secondary", key="btn_rollback"):
                with st.spinner("正在回滚..."):
                    try:
                        result = live_migration_manager.rollback()
                        if result.get("success"):
                            st.success("✅ 已回滚到模拟盘模式")
                            st.rerun()
                    except Exception as e:
                        st.error(f"回滚失败: {e}")

    except Exception as e:
        st.info(f"实盘迁移模块未就绪: {e}")

    st.divider()

    # 调度器状态
    st.subheader("⏰ 定时任务调度器")
    try:
        cc = _get_cloud_client()
        sched_status = cc.get_scheduler_status()
        if sched_status["running"]:
            st.success(f"调度器运行中 | {sched_status['task_count']} 个任务")
        else:
            st.info("调度器未启动")

        tasks = sched_status.get("tasks", [])
        if tasks:
            st.dataframe(pd.DataFrame(tasks), width='stretch')

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ 启动调度器") and not sched_status["running"]:
                try:
                    trading_scheduler.register_daily_loop()
                    trading_scheduler.register_monitor_snapshot(5)
                    trading_scheduler.start()
                    st.success("调度器已启动")
                    st.rerun()
                except Exception as e:
                    st.error(f"启动失败: {e}")
        with col2:
            if st.button("⏹️ 停止调度器") and sched_status["running"]:
                trading_scheduler.stop()
                st.success("调度器已停止")
                st.rerun()
    except Exception as e:
        st.info(f"调度器模块未就绪: {e}")
    
    st.divider()
    
    # 系统配置（可编辑）
    st.subheader("📋 系统配置")
    st.caption("修改后点击底部保存按钮，配置将写入 .env 文件")

    # 直接使用本地 config 对象（而非 cloud_client 的脱敏字典）
    try:
        from src.utils.config import config as _config_instance
    except Exception:
        _config_instance = config

    # 收集用户修改
    env_updates = {}

    # ── 🔑 API 密钥 ──
    with st.expander("🔑 API 密钥", expanded=False):
        st.caption("留空表示不修改，显示为脱敏值。输入新值将覆盖原配置。")

        # Alpaca
        alpaca_key_display = f"***{_config_instance.ALPACA_API_KEY[-4:]}" if _config_instance.has_alpaca_keys() else ""
        new_alpaca_key = st.text_input(
            "ALPACA_API_KEY",
            value="",
            placeholder=f"当前: {alpaca_key_display}" if alpaca_key_display else "未配置",
            help="Alpaca 交易 API Key",
        )
        if new_alpaca_key.strip():
            env_updates["ALPACA_API_KEY"] = new_alpaca_key.strip()

        alpaca_secret_display = f"***{_config_instance.ALPACA_SECRET_KEY[-4:]}" if _config_instance.has_alpaca_keys() else ""
        new_alpaca_secret = st.text_input(
            "ALPACA_SECRET_KEY",
            value="",
            placeholder=f"当前: {alpaca_secret_display}" if alpaca_secret_display else "未配置",
            help="Alpaca 交易 API Secret Key",
        )
        if new_alpaca_secret.strip():
            env_updates["ALPACA_SECRET_KEY"] = new_alpaca_secret.strip()

        # Alpaca Base URL
        new_alpaca_url = st.text_input(
            "ALPACA_BASE_URL",
            value=_config_instance.ALPACA_BASE_URL,
            help="Alpaca API 基础 URL",
        )
        if new_alpaca_url != _config_instance.ALPACA_BASE_URL:
            env_updates["ALPACA_BASE_URL"] = new_alpaca_url

        # Finnhub
        finnhub_display = f"***{_config_instance.FINNHUB_API_KEY[-4:]}" if _config_instance.has_finnhub_key() else ""
        new_finnhub_key = st.text_input(
            "FINNHUB_API_KEY",
            value="",
            placeholder=f"当前: {finnhub_display}" if finnhub_display else "未配置",
            help="Finnhub 行情数据 API Key",
        )
        if new_finnhub_key.strip():
            env_updates["FINNHUB_API_KEY"] = new_finnhub_key.strip()

        # FRED
        fred_display = f"***{_config_instance.FRED_API_KEY[-4:]}" if _config_instance.has_fred_key() else ""
        new_fred_key = st.text_input(
            "FRED_API_KEY",
            value="",
            placeholder=f"当前: {fred_display}" if fred_display else "未配置",
            help="FRED 宏观经济数据 API Key",
        )
        if new_fred_key.strip():
            env_updates["FRED_API_KEY"] = new_fred_key.strip()

        # FMP
        fmp_display = f"***{_config_instance.FMP_API_KEY[-4:]}" if _config_instance.has_fmp_key() else ""
        new_fmp_key = st.text_input(
            "FMP_API_KEY",
            value="",
            placeholder=f"当前: {fmp_display}" if fmp_display else "未配置",
            help="Financial Modeling Prep 财务数据 API Key",
        )
        if new_fmp_key.strip():
            env_updates["FMP_API_KEY"] = new_fmp_key.strip()

    # ── 🧠 LLM 智能助手配置 ──
    with st.expander("🧠 LLM 智能助手配置", expanded=False):
        try:
            from src.llm.llm_config import llm_config
            llm_config.load()  # 确保加载最新配置
            llm_status = llm_config.status()

            # 显示当前 LLM 状态
            if llm_status["enabled"]:
                st.success(f"✅ LLM 已启用 | 默认模型: {llm_status['default_model']} | "
                           f"可用: {', '.join(llm_status['available_models'])}")
            elif llm_status["total_models"] > 0:
                st.warning(f"⚠️ 已配置 {llm_status['total_models']} 个模型但无可用项 (Key 可能未填)。"
                           f" 模型: {', '.join(llm_status['all_models'])}")
            else:
                st.info("📝 未配置 LLM 模型，智能助手使用规则引擎模式。配置后可启用 AI 智能对话。")

            st.caption("配置一个 OpenAI 兼容 API。支持 DeepSeek、豆包、通义千问、OpenAI 等。")

            existing_models = llm_status.get("all_models", [])

            # 模型选择或新建
            col_name, col_default = st.columns([3, 1])
            with col_name:
                if existing_models:
                    model_options = existing_models + ["➕ 新建模型"]
                    model_choice = st.selectbox("选择模型编辑", model_options, index=0)
                    if model_choice == "➕ 新建模型":
                        new_model_name = st.text_input(
                            "新模型名称",
                            value="",
                            placeholder="如: mymodel1 (英文字母数字)",
                            help="模型标识名，用于区分不同模型配置",
                        )
                        editing_model = new_model_name.strip().lower() if new_model_name.strip() else ""
                    else:
                        editing_model = model_choice
                else:
                    editing_model = st.text_input(
                        "模型名称",
                        value="",
                        placeholder="如: mymodel1",
                        help="模型标识名，自定义命名",
                    ).strip().lower()

            with col_default:
                is_default = st.checkbox(
                    "设为默认模型",
                    value=(editing_model == llm_status.get("default_model", "")) if editing_model else False,
                    help="勾选后此模型将作为智能助手的默认 LLM",
                )

            # 预填已有模型配置
            current_entry = None
            if editing_model and editing_model in existing_models:
                current_entry = llm_config.get_model(editing_model)

            # Endpoint
            llm_endpoint = st.text_input(
                "API Endpoint",
                value=current_entry.endpoint if current_entry else "",
                placeholder="如: https://api.deepseek.com/v1 或 https://ark.cn-beijing.volces.com/api/v3",
                help="OpenAI 兼容 API 的地址",
            )

            # API Key (密码模式，脱敏)
            if current_entry and current_entry.is_available():
                key_placeholder = f"当前: ***{current_entry.api_key[-4:]} (留空不修改)"
            else:
                key_placeholder = "输入 API Key"
            llm_api_key = st.text_input(
                "API Key",
                value="",
                type="password",
                placeholder=key_placeholder,
                help="留空表示不修改现有 Key",
            )

            # Model ID
            llm_model_id = st.text_input(
                "模型 ID",
                value=current_entry.model_id if current_entry else "",
                placeholder="如: deepseek-chat, gpt-4o-mini, doubao-seed-2-0-mini-260428",
                help="API 提供商指定的模型标识",
            )

            # Temperature 和 Max Tokens
            col_t, col_m = st.columns(2)
            with col_t:
                llm_temp = st.number_input(
                    "Temperature (温度)",
                    min_value=0.0, max_value=2.0,
                    value=float(current_entry.temperature) if current_entry else 0.7,
                    step=0.1,
                    format="%.1f",
                    help="值越高回复越随机，值越低越确定保守",
                )
            with col_m:
                llm_max_tokens = st.number_input(
                    "Max Tokens (最大 Token 数)",
                    min_value=256, max_value=32768,
                    value=int(current_entry.max_tokens) if current_entry else 2000,
                    step=256,
                    help="LLM 单次回复的最大长度",
                )

            # 收集 LLM 配置变更到 env_updates
            if editing_model:
                prefix = f"LLM_{editing_model.upper()}"
                if llm_endpoint.strip():
                    env_updates[f"{prefix}_ENDPOINT"] = llm_endpoint.strip()
                if llm_api_key.strip():
                    env_updates[f"{prefix}_API_KEY"] = llm_api_key.strip()
                if llm_model_id.strip():
                    env_updates[f"{prefix}_MODEL"] = llm_model_id.strip()
                env_updates[f"{prefix}_TEMPERATURE"] = llm_temp
                env_updates[f"{prefix}_MAX_TOKENS"] = int(llm_max_tokens)
                if is_default:
                    env_updates["LLM_DEFAULT"] = editing_model.lower()

        except Exception as e:
            st.warning(f"LLM 配置加载失败: {e}")

    # ── 📊 交易与风控 ──
    with st.expander("📊 交易与风控配置", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            new_trading_mode = st.selectbox(
                "TRADING_MODE",
                options=["paper", "live"],
                index=0 if _config_instance.TRADING_MODE == "paper" else 1,
                format_func=lambda x: "📝 模拟盘 (Paper)" if x == "paper" else "🔴 实盘 (Live)",
                help="交易模式: paper=模拟盘, live=实盘",
            )
            if new_trading_mode != _config_instance.TRADING_MODE:
                env_updates["TRADING_MODE"] = new_trading_mode

        with col2:
            new_log_level = st.selectbox(
                "LOG_LEVEL",
                options=["DEBUG", "INFO", "WARNING", "ERROR"],
                index=["DEBUG", "INFO", "WARNING", "ERROR"].index(_config_instance.LOG_LEVEL) if _config_instance.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR"] else 1,
                help="日志级别",
            )
            if new_log_level != _config_instance.LOG_LEVEL:
                env_updates["LOG_LEVEL"] = new_log_level

        with col3:
            new_deploy_mode = st.selectbox(
                "DEPLOY_MODE",
                options=["local", "cloud"],
                index=0 if _config_instance.DEPLOY_MODE == "local" else 1,
                format_func=lambda x: "🖥️ 本地" if x == "local" else "☁️ 云端",
                help="部署模式",
            )
            if new_deploy_mode != _config_instance.DEPLOY_MODE:
                env_updates["DEPLOY_MODE"] = new_deploy_mode

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            new_max_pos = st.number_input(
                "MAX_POSITION_PCT (单票最大仓位)",
                min_value=0.01, max_value=1.0,
                value=float(_config_instance.MAX_POSITION_PCT),
                step=0.05,
                format="%.2f",
                help="单只股票最大仓位占比 (0.30 = 30%)",
            )
            if new_max_pos != _config_instance.MAX_POSITION_PCT:
                env_updates["MAX_POSITION_PCT"] = new_max_pos

        with col2:
            new_max_loss = st.number_input(
                "MAX_DAILY_LOSS (日最大亏损)",
                min_value=0.01, max_value=0.50,
                value=float(_config_instance.MAX_DAILY_LOSS),
                step=0.01,
                format="%.2f",
                help="单日最大亏损比例 (0.05 = 5%)",
            )
            if new_max_loss != _config_instance.MAX_DAILY_LOSS:
                env_updates["MAX_DAILY_LOSS"] = new_max_loss

        with col3:
            new_max_dd = st.number_input(
                "MAX_DRAWDOWN (最大回撤)",
                min_value=0.01, max_value=0.50,
                value=float(_config_instance.MAX_DRAWDOWN),
                step=0.01,
                format="%.2f",
                help="允许的最大回撤比例 (0.15 = 15%)",
            )
            if new_max_dd != _config_instance.MAX_DRAWDOWN:
                env_updates["MAX_DRAWDOWN"] = new_max_dd

    # ── 🎯 选股配置 ──
    with st.expander("🎯 选股配置", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            new_max_candidates = st.number_input(
                "MAX_CANDIDATES (候选股票上限)",
                min_value=1, max_value=50,
                value=int(_config_instance.MAX_CANDIDATES),
                step=1,
                help="调研阶段最多发现多少只候选股票",
            )
            if new_max_candidates != _config_instance.MAX_CANDIDATES:
                env_updates["MAX_CANDIDATES"] = new_max_candidates

        with col2:
            new_top_k = st.number_input(
                "TOP_K_SIGNALS (信号Top-K)",
                min_value=1, max_value=50,
                value=int(_config_instance.TOP_K_SIGNALS),
                step=1,
                help="信号生成阶段选前K只做多",
            )
            if new_top_k != _config_instance.TOP_K_SIGNALS:
                env_updates["TOP_K_SIGNALS"] = new_top_k

        with col3:
            new_screen_size = st.number_input(
                "SCREEN_UNIVERSE_SIZE (预筛选数量)",
                min_value=10, max_value=500,
                value=int(_config_instance.SCREEN_UNIVERSE_SIZE),
                step=10,
                help="预筛选进入详细分析的最大数量",
            )
            if new_screen_size != _config_instance.SCREEN_UNIVERSE_SIZE:
                env_updates["SCREEN_UNIVERSE_SIZE"] = new_screen_size

    # ── 🪙 加密货币配置 ──
    with st.expander("🪙 加密货币配置", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            new_crypto_enabled = st.checkbox(
                "CRYPTO_ENABLED (启用加密货币模块)",
                value=bool(_config_instance.CRYPTO_ENABLED),
            )
            if new_crypto_enabled != _config_instance.CRYPTO_ENABLED:
                env_updates["CRYPTO_ENABLED"] = new_crypto_enabled

        with col2:
            new_crypto_trading = st.checkbox(
                "CRYPTO_TRADING_ENABLED (启用加密货币交易)",
                value=bool(_config_instance.CRYPTO_TRADING_ENABLED),
            )
            if new_crypto_trading != _config_instance.CRYPTO_TRADING_ENABLED:
                env_updates["CRYPTO_TRADING_ENABLED"] = new_crypto_trading

        col1, col2 = st.columns(2)

        with col1:
            new_crypto_source = st.selectbox(
                "CRYPTO_DATA_SOURCE (数据源)",
                options=["alpaca", "ccxt"],
                index=0 if _config_instance.CRYPTO_DATA_SOURCE == "alpaca" else 1,
            )
            if new_crypto_source != _config_instance.CRYPTO_DATA_SOURCE:
                env_updates["CRYPTO_DATA_SOURCE"] = new_crypto_source

        with col2:
            new_crypto_pairs = st.text_input(
                "CRYPTO_TRADING_PAIRS (交易对, 逗号分隔)",
                value=",".join(_config_instance.get_crypto_pairs()) if _config_instance.get_crypto_pairs() else "",
                help="例如: BTC/USD,ETH/USD,SOL/USD",
            )
            if new_crypto_pairs.strip():
                pair_list = [s.strip() for s in new_crypto_pairs.split(",") if s.strip()]
                if pair_list != _config_instance.get_crypto_pairs():
                    env_updates["CRYPTO_TRADING_PAIRS"] = pair_list

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            new_crypto_max_pos = st.number_input(
                "CRYPTO_MAX_POSITION_PCT (单币最大仓位)",
                min_value=0.01, max_value=1.0,
                value=float(_config_instance.CRYPTO_MAX_POSITION_PCT),
                step=0.05,
                format="%.2f",
            )
            if new_crypto_max_pos != _config_instance.CRYPTO_MAX_POSITION_PCT:
                env_updates["CRYPTO_MAX_POSITION_PCT"] = new_crypto_max_pos

        with col2:
            new_crypto_max_loss = st.number_input(
                "CRYPTO_MAX_DAILY_LOSS (日最大亏损)",
                min_value=0.01, max_value=0.50,
                value=float(_config_instance.CRYPTO_MAX_DAILY_LOSS),
                step=0.01,
                format="%.2f",
            )
            if new_crypto_max_loss != _config_instance.CRYPTO_MAX_DAILY_LOSS:
                env_updates["CRYPTO_MAX_DAILY_LOSS"] = new_crypto_max_loss

        with col3:
            new_crypto_max_dd = st.number_input(
                "CRYPTO_MAX_DRAWDOWN (最大回撤)",
                min_value=0.01, max_value=0.50,
                value=float(_config_instance.CRYPTO_MAX_DRAWDOWN),
                step=0.01,
                format="%.2f",
            )
            if new_crypto_max_dd != _config_instance.CRYPTO_MAX_DRAWDOWN:
                env_updates["CRYPTO_MAX_DRAWDOWN"] = new_crypto_max_dd

        col1, col2, col3 = st.columns(3)

        with col1:
            new_crypto_stoploss = st.number_input(
                "CRYPTO_SINGLE_STOP_LOSS (单币止损)",
                min_value=0.01, max_value=0.50,
                value=float(_config_instance.CRYPTO_SINGLE_STOP_LOSS),
                step=0.01,
                format="%.2f",
            )
            if new_crypto_stoploss != _config_instance.CRYPTO_SINGLE_STOP_LOSS:
                env_updates["CRYPTO_SINGLE_STOP_LOSS"] = new_crypto_stoploss

        with col2:
            new_crypto_trade_limit = st.number_input(
                "CRYPTO_DAILY_TRADE_LIMIT (日交易次数)",
                min_value=1, max_value=200,
                value=int(_config_instance.CRYPTO_DAILY_TRADE_LIMIT),
                step=1,
            )
            if new_crypto_trade_limit != _config_instance.CRYPTO_DAILY_TRADE_LIMIT:
                env_updates["CRYPTO_DAILY_TRADE_LIMIT"] = new_crypto_trade_limit

        with col3:
            new_crypto_max_positions = st.number_input(
                "CRYPTO_MAX_TOTAL_POSITIONS (最大持仓币种数)",
                min_value=1, max_value=50,
                value=int(_config_instance.CRYPTO_MAX_TOTAL_POSITIONS),
                step=1,
            )
            if new_crypto_max_positions != _config_instance.CRYPTO_MAX_TOTAL_POSITIONS:
                env_updates["CRYPTO_MAX_TOTAL_POSITIONS"] = new_crypto_max_positions

    # ── ☁️ 部署与系统配置 ──
    with st.expander("☁️ 部署与系统配置", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            new_qlib_enabled = st.checkbox(
                "QLIB_ENABLED (启用 Qlib)",
                value=bool(_config_instance.QLIB_ENABLED),
            )
            if new_qlib_enabled != _config_instance.QLIB_ENABLED:
                env_updates["QLIB_ENABLED"] = new_qlib_enabled

        with col2:
            new_torch_enabled = st.checkbox(
                "TORCH_ENABLED (启用 PyTorch)",
                value=bool(_config_instance.TORCH_ENABLED),
            )
            if new_torch_enabled != _config_instance.TORCH_ENABLED:
                env_updates["TORCH_ENABLED"] = new_torch_enabled

        col1, col2 = st.columns(2)

        with col1:
            new_cloud_url = st.text_input(
                "CLOUD_API_URL (云端 API 地址)",
                value=_config_instance.CLOUD_API_URL or "",
                help="本地上传模型到云端时使用的 API 地址",
            )
            if new_cloud_url != (_config_instance.CLOUD_API_URL or ""):
                env_updates["CLOUD_API_URL"] = new_cloud_url

        with col2:
            new_cloud_port = st.text_input(
                "CLOUD_API_PORT (云端 API 端口)",
                value=_config_instance.CLOUD_API_PORT or "8000",
            )
            if new_cloud_port != (_config_instance.CLOUD_API_PORT or "8000"):
                env_updates["CLOUD_API_PORT"] = new_cloud_port

        # 只读路径信息
        st.markdown("---")
        st.caption("📂 路径信息（只读）")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("QLIB_DATA_DIR", value=str(_config_instance.QLIB_DATA_PATH), disabled=True)
        with col2:
            st.text_input("CACHE_DIR", value=str(_config_instance.CACHE_DIR), disabled=True)

    # ── 💾 保存按钮 ──
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💾 保存配置到 .env", type="primary", use_container_width=True):
            if not env_updates:
                st.info("ℹ️ 没有检测到配置变更")
            else:
                try:
                    msg = _config_instance.save_to_env(env_updates)
                    # 同步更新内存中的 config 对象
                    for key, value in env_updates.items():
                        if hasattr(_config_instance, key):
                            # 类型转换
                            current_val = getattr(_config_instance, key)
                            if isinstance(current_val, bool):
                                setattr(_config_instance, key, bool(value))
                            elif isinstance(current_val, int):
                                setattr(_config_instance, key, int(value))
                            elif isinstance(current_val, float):
                                setattr(_config_instance, key, float(value))
                            else:
                                setattr(_config_instance, key, value)
                    st.success(f"✅ {msg}")
                    # 如果有 LLM 配置变更，重新加载 LLM 配置
                    if any(k.startswith("LLM_") for k in env_updates):
                        try:
                            from src.llm.llm_config import llm_config
                            llm_config.reload()
                            from src.ui.nl_router import nl_router
                            nl_router._llm_checked = False
                            nl_router._llm_active = False
                            st.info("🧠 LLM 配置已重新加载")
                        except Exception as llm_e:
                            st.warning(f"LLM 配置重载失败: {llm_e}")
                    st.info("💡 部分配置可能需要重启服务后生效")
                except Exception as e:
                    st.error(f"❌ 保存失败: {e}")

    # 显示变更预览
    if env_updates:
        with st.expander(f"📝 待保存的变更 ({len(env_updates)} 项)", expanded=False):
            for key, value in env_updates.items():
                if isinstance(value, bool):
                    display_val = "true" if value else "false"
                elif isinstance(value, list):
                    display_val = ",".join(value)
                else:
                    display_val = str(value)
                # API 密钥脱敏显示
                if "KEY" in key or "SECRET" in key:
                    display_val = f"***{str(value)[-4:]}" if len(str(value)) > 4 else "****"
                st.text(f"  {key} = {display_val}")

   
# ═══════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════

def main():
    """仪表盘主入口"""
    # 初始化日志 (仅在 main 中调用)
    try:
        from src.utils.logger import setup_logger
        setup_logger()
    except Exception:
        pass

    page = render_sidebar()

    # ── 切换页面时自动滚动到顶部 ──
    if page != st.session_state.get("_current_page"):
        st.session_state["_current_page"] = page
        st.iframe("""
        <script>
        (function() {
            var doc = window.parent.document;
            // 尝试多种选择器找到可滚动容器 (兼容不同 Streamlit 版本)
            var selectors = [
                'section[data-testid="stMain"]',
                '.main',
                'main',
                '[data-testid="stAppViewContainer"]',
                '.block-container',
                '#root > div > div'
            ];
            for (var i = 0; i < selectors.length; i++) {
                try {
                    var el = doc.querySelector(selectors[i]);
                    if (el && el.scrollHeight > el.clientHeight) {
                        el.scrollTop = 0;
                    }
                } catch(e) {}
            }
            // 兜底: window 滚动
            window.parent.scrollTo(0, 0);
        })();
        </script>
        """, height=1)

    if page == "📊 总览首页":
        page_overview()
    elif page == "📈 行情分析":
        from src.ui.chart_analysis import page_chart_analysis
        page_chart_analysis()
    elif page == "🎯 智能选股":
        page_stock_screen()
    elif page == "📈 策略管理":
        page_strategies()
    elif page == "💰 交易记录":
        page_trades()
    elif page == "🔍 市场调研":
        page_research()
    elif page == "🌐 宏观指标":
        page_macro()
    elif page == "📊 期权数据":
        page_options()
    elif page == "🪙 加密货币":
        page_crypto()
    elif page == "🧠 模型训练":
        from src.ui.training_dashboard import page_model_training
        page_model_training()
    elif page == "⚙️ 系统设置":
        page_settings()

    # ── 悬浮智能助手聊天窗口 (所有页面通用) ──
    try:
        render_floating_chat()
    except Exception as e:
        st.warning(f"智能助手加载失败: {e}")


# ═══════════════════════════════════════════════════════
# 悬浮智能助手聊天窗口 (v1.0.0)
# ═══════════════════════════════════════════════════════
# 设计要点:
# 1. UI 完全独立: FastAPI 服务 (8502端口) + iframe 嵌入
#    -> 主页面加载/报错不影响聊天窗口
# 2. 右下角悬浮按钮: 点击切换聊天窗口显示/隐藏
# 3. 流式输出: SSE (Server-Sent Events) 逐块推送
# 4. 自动 markdown: 前端 JS 简易渲染器
# 5. 对话历史: 文件持久化 (刷新保留)

_CHAT_PORT = 8502
_CHAT_WINDOW_WIDTH = 400
_CHAT_WINDOW_HEIGHT = 620


def _ensure_chat_server():
    """确保 FastAPI 聊天服务已启动 (自动 spawn 独立进程)"""
    import socket
    import subprocess
    import sys as _sys
    import time as _time
    import urllib.request as _urllib

    _project_root = Path(__file__).resolve().parent.parent.parent

    # 检查端口是否已有 FastAPI 服务
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", _CHAT_PORT)) == 0:
                # 端口已占用，检查是否是 FastAPI 聊天服务
                try:
                    resp = _urllib.urlopen(
                        f"http://127.0.0.1:{_CHAT_PORT}/api/health", timeout=1
                    )
                    data = resp.read().decode()
                    if "chat_server" in data:
                        return True  # FastAPI 聊天服务已在运行
                except Exception:
                    pass
                # 端口被其他服务占用 (可能是旧 Streamlit)，不处理直接返回
                return True
    except Exception:
        pass

    # 端口未占用 -> 启动 FastAPI 聊天服务进程
    try:
        subprocess.Popen(
            [_sys.executable, "-m", "uvicorn",
             "src.api.chat_server:app",
             "--host", "0.0.0.0",
             "--port", str(_CHAT_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(_project_root),
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        # 等待服务就绪 (最多 5 秒)
        for _ in range(10):
            _time.sleep(0.5)
            try:
                resp = _urllib.urlopen(
                    f"http://127.0.0.1:{_CHAT_PORT}/api/health", timeout=1
                )
                if resp.status == 200:
                    return True
            except Exception:
                continue
        return True  # 即使健康检查没过也返回 True，让 iframe 尝试加载
    except Exception as e:
        st.warning(f"智能助手服务启动失败: {e}")
        return False


def _inject_chat_widget():
    """注入悬浮聊天窗口 (v1.1.0: JS 创建元素，防止 Streamlit rerun 导致状态丢失)"""
    # ── 第1部分: CSS 样式 (st.markdown 每次 rerun 重新注入，CSS 幂等无副作用) ──
    st.markdown(f"""
    <style>
    /* ══════════════════════════════════════════
       悬浮智能助手 - 右下角按钮 + 聊天窗口
       v1.1.0: 元素由 JS 创建，持久化在 document.body
    ══════════════════════════════════════════ */

    #chat-toggle {{ display: none; }}

    #chat-toggle-label {{
        position: fixed; right: 24px; bottom: 24px; z-index: 99999;
        width: 56px; height: 56px; border-radius: 50%; cursor: pointer;
        font-size: 26px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #4a9eff 0%, #6c5ce7 100%);
        color: #fff; user-select: none;
        border: 2px solid rgba(255, 255, 255, 0.15);
    }}
    #chat-toggle-label:hover {{
        transform: scale(1.08) rotate(6deg);
        box-shadow: 0 6px 24px rgba(74, 158, 255, 0.4);
    }}
    #chat-toggle-label:active {{ transform: scale(0.95); }}
    #chat-toggle-label::after {{
        content: ""; position: absolute; inset: -4px; border-radius: 50%;
        background: radial-gradient(circle, rgba(74, 158, 255, 0.25) 0%, transparent 70%);
        z-index: -1; animation: chat-breathe 2.8s ease-in-out infinite;
    }}
    @keyframes chat-breathe {{
        0%, 100% {{ transform: scale(1); opacity: 0.5; }}
        50% {{ transform: scale(1.18); opacity: 0.15; }}
    }}
    #chat-toggle-label .icon-close {{ display: none; }}
    #chat-toggle:checked ~ #chat-toggle-label .icon-open {{ display: none; }}
    #chat-toggle:checked ~ #chat-toggle-label .icon-close {{ display: flex; }}

    #chat-window {{
        position: fixed; right: 24px; bottom: 92px; z-index: 99998;
        width: {_CHAT_WINDOW_WIDTH}px; height: {_CHAT_WINDOW_HEIGHT}px;
        border-radius: 18px; overflow: hidden;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15), 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.06); background: #fff;
        display: none;
        transition: opacity 0.3s ease, transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        transform-origin: bottom right;
    }}
    #chat-toggle:checked ~ #chat-window {{
        display: block;
        animation: chat-pop 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    }}
    @keyframes chat-pop {{
        0% {{ transform: scale(0.88) translateY(10px); opacity: 0; }}
        100% {{ transform: scale(1) translateY(0); opacity: 1; }}
    }}
    #chat-window::before {{
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #4a9eff, #6c5ce7, #4a9eff);
        background-size: 200% 100%; animation: chat-shimmer 3s linear infinite; z-index: 3;
    }}
    @keyframes chat-shimmer {{
        0% {{ background-position: 0% 0; }}
        100% {{ background-position: 200% 0; }}
    }}
    [data-theme="dark"] #chat-window {{
        background: #1a2332; border-color: rgba(255, 255, 255, 0.06);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 2px 8px rgba(0, 0, 0, 0.2);
    }}
    #chat-window iframe {{
        width: 100%; height: 100%; border: none; background: transparent;
        position: relative; z-index: 0;
    }}
    #chat-loading {{
        position: absolute; inset: 0; display: flex; align-items: center;
        justify-content: center; font-size: 13px; color: #888;
        background: #fff; z-index: 1; flex-direction: column; gap: 8px;
    }}
    #chat-loading .loading-dots {{ display: flex; gap: 4px; }}
    #chat-loading .loading-dots span {{
        width: 7px; height: 7px; border-radius: 50%; background: #4a9eff;
        animation: loading-bounce 1.4s infinite ease-in-out;
    }}
    #chat-loading .loading-dots span:nth-child(1) {{ animation-delay: 0s; }}
    #chat-loading .loading-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
    #chat-loading .loading-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
    @keyframes loading-bounce {{
        0%, 80%, 100% {{ transform: scale(0.5); opacity: 0.3; }}
        40% {{ transform: scale(1); opacity: 1; }}
    }}
    [data-theme="dark"] #chat-loading {{ background: #1a2332; color: #aaa; }}
    </style>
    """, unsafe_allow_html=True)

    # ── 第2部分: JS 创建元素 + 状态持久化 + 页面上下文通信 (v1.4.0) ──
    # v1.4.0: 回退到 st.iframe (iframe 内执行 JS，通过 parent 操作主页面)
    #   原因: st.html(unsafe_allow_javascript=True) 在部分 Streamlit 版本中
    #         DOMPurify 会清理 <script> 标签导致 JS 不执行
    #   st.iframe 在 iframe 中执行 JS，更可靠
    #   幂等设计保证 rerun 安全: 元素已存在时跳过创建，监听器/定时器先清理再注册
    st.iframe(f"""
    <script>
    (function() {{
        // st.iframe 在 iframe 中执行，需通过 parent 访问主页面
        var pwin = window.parent;
        var doc = pwin.document;
        var pStorage = pwin.sessionStorage;

        // ═══ 1. 创建聊天窗口元素 (仅首次) ═══
        if (!doc.getElementById('chat-toggle')) {{
            var toggle = doc.createElement('input');
            toggle.type = 'checkbox';
            toggle.id = 'chat-toggle';
            toggle.setAttribute('aria-label', '切换智能助手');

            var label = doc.createElement('label');
            label.id = 'chat-toggle-label';
            label.htmlFor = 'chat-toggle';
            label.title = 'AI 智能助手';
            label.innerHTML = '<span class="icon-open">💬</span><span class="icon-close">✕</span>';

            var win = doc.createElement('div');
            win.id = 'chat-window';
            // 动态构建聊天服务地址: 使用当前页面的 hostname + 8502 端口
            // 支持 localhost、IP、域名等各种访问方式
            var chatHost = pwin.location.hostname;
            var chatProtocol = pwin.location.protocol;
            var chatUrl = chatProtocol + '//' + chatHost + ':{_CHAT_PORT}';
            win.innerHTML = '<div id="chat-loading"><div>🤖 智能助手加载中</div>'
                + '<div class="loading-dots"><span></span><span></span><span></span></div></div>'
                + '<iframe id="chat-frame" src="' + chatUrl + '" '
                + 'loading="lazy" allow="clipboard-write"></iframe>';

            // 按顺序 append 到 body (CSS ~ 选择器要求兄弟关系)
            doc.body.appendChild(toggle);
            doc.body.appendChild(label);
            doc.body.appendChild(win);

            // iframe 加载完成后隐藏 loading
            var iframe = doc.getElementById('chat-frame');
            var loading = doc.getElementById('chat-loading');
            if (iframe && loading) {{
                iframe.addEventListener('load', function() {{
                    loading.style.display = 'none';
                }});
                pwin.setTimeout(function() {{ if (loading) loading.style.display = 'none'; }}, 8000);
            }}

            // v1.1.0: 状态持久化 - 点击时保存到 sessionStorage
            label.addEventListener('click', function() {{
                pwin.setTimeout(function() {{
                    var t = doc.getElementById('chat-toggle');
                    if (t) pStorage.setItem('chat-open', t.checked ? 'true' : 'false');
                }}, 0);
            }});
        }}

        // ═══ 2. 恢复窗口状态 (每次 rerun 都执行) ═══
        var toggleEl = doc.getElementById('chat-toggle');
        if (toggleEl) {{
            if (pStorage.getItem('chat-open') === 'true') {{
                toggleEl.checked = true;
            }}
        }}

        // ═══ 3. 页面上下文通信 (postMessage) ═══
        // 移除旧监听器防止重复
        if (pwin._chatPageCtxHandler) {{
            pwin.removeEventListener('message', pwin._chatPageCtxHandler);
        }}

        pwin._chatPageCtxHandler = function(event) {{
            if (!event.data || event.data.type !== 'request_page_context') return;

            var iframe = doc.getElementById('chat-frame');
            if (!iframe || event.source !== iframe.contentWindow) return;

            // ── 提取当前页面名称 (多策略容错 v2) ──
            var pageName = '';
            try {{
                // 策略1: label[data-checked="true"] (与CSS一致，最可靠)
                var checkedLabel = doc.querySelector(
                    'section[data-testid="stSidebar"] label[data-checked="true"]'
                ) || doc.querySelector(
                    'section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"]'
                );
                if (checkedLabel && checkedLabel.textContent.trim()) {{
                    pageName = checkedLabel.textContent.trim();
                }}

                // 策略2: [aria-checked="true"] (role="radio" 模式)
                if (!pageName) {{
                    var ariaEl = doc.querySelector(
                        'section[data-testid="stSidebar"] [aria-checked="true"]'
                    );
                    if (ariaEl && ariaEl.textContent.trim()) {{
                        pageName = ariaEl.textContent.trim();
                    }}
                }}

                // 策略3: input[type="radio"]:checked + closest label
                if (!pageName) {{
                    var checkedRadio = doc.querySelector(
                        'section[data-testid="stSidebar"] input[type="radio"]:checked'
                    );
                    if (checkedRadio) {{
                        var container = checkedRadio.closest('label')
                            || checkedRadio.closest('[role="radio"]')
                            || checkedRadio.closest('[data-testid="stRadioContentContainer"]')
                            || checkedRadio.parentElement;
                        if (container && container.textContent.trim()) {{
                            pageName = container.textContent.trim();
                        }}
                    }}
                }}

                // 策略4: .stRadio 容器内的选中项 (旧版 Streamlit 兼容)
                if (!pageName) {{
                    var radioContainer = doc.querySelector(
                        'section[data-testid="stSidebar"] [data-testid="stRadio"]'
                    ) || doc.querySelector('section[data-testid="stSidebar"] .stRadio');
                    if (radioContainer) {{
                        var selected = radioContainer.querySelector('[data-checked="true"]')
                            || radioContainer.querySelector('[aria-checked="true"]')
                            || radioContainer.querySelector('input[type="radio"]:checked');
                        if (selected) {{
                            var selContainer = selected.closest('label')
                                || selected.parentElement
                                || selected;
                            if (selContainer && selContainer.textContent.trim()) {{
                                pageName = selContainer.textContent.trim();
                            }}
                        }}
                    }}
                }}

                // 策略5: 主内容区第一个标题 (兜底)
                if (!pageName) {{
                    var heading = doc.querySelector(
                        'section[data-testid="stMain"] h1, section.main h1, .stAppViewContainer h1, [data-testid="stMain"] h2'
                    );
                    if (heading) pageName = heading.textContent.trim();
                }}

                // 调试日志
                console.log('[页面上下文] 提取到的页面名称:', pageName || '(空)');
            }} catch(e) {{
                console.error('[页面上下文] 提取页面名称失败:', e);
            }}

            // ── 提取主内容区文本 (排除侧边栏和聊天窗口) ──
            var mainContent = '';
            try {{
                var mainEl = doc.querySelector('section[data-testid="stMain"]')
                    || doc.querySelector('section.main')
                    || doc.querySelector('main')
                    || doc.querySelector('.stAppViewContainer');
                if (mainEl) {{
                    // 克隆节点并移除不需要的元素 (侧边栏、聊天窗口、脚本、样式)
                    var clone = mainEl.cloneNode(true);
                    var unwanted = clone.querySelectorAll(
                        'script, style, #chat-window, #chat-toggle, #chat-toggle-label, iframe, .stSpinner'
                    );
                    unwanted.forEach(function(el) {{ el.remove(); }});
                    mainContent = (clone.innerText || clone.textContent || '')
                        .replace(/\\n{{3,}}/g, '\\n\\n')
                        .trim();
                    if (mainContent.length > 3000) {{
                        mainContent = mainContent.substring(0, 3000) + '...';
                    }}
                }}
            }} catch(e) {{
                console.error('[页面上下文] 提取主内容失败:', e);
            }}

            // ── 拼接并返回 ──
            var context = '';
            if (pageName) context += '当前页面: ' + pageName + '\\n\\n';
            if (mainContent) context += '页面内容:\\n' + mainContent;

            iframe.contentWindow.postMessage({{
                type: 'page_context',
                title: pageName,
                content: context
            }}, '*');
        }};

        pwin.addEventListener('message', pwin._chatPageCtxHandler);

        // ═══ 4. 主题同步: 检测 Streamlit 主题并推送给聊天 iframe ═══
        // 移除旧监听器防止重复
        if (pwin._chatThemeHandler) {{
            pwin.removeEventListener('message', pwin._chatThemeHandler);
        }}

        // 检测当前 Streamlit 主题
        function detectStreamlitTheme() {{
            try {{
                // 方法1: 查找 Streamlit 的 data-theme 属性 (最可靠)
                var themedEl = doc.querySelector('[data-theme]');
                if (themedEl) {{
                    var t = themedEl.getAttribute('data-theme');
                    if (t === 'dark' || t === 'light') return t;
                }}
                // 方法2: 检查 body 背景色判断深浅
                var bgColor = pwin.getComputedStyle(doc.body).backgroundColor;
                // 解析 rgb(r, g, b)
                var match = bgColor.match(/\\d+/g);
                if (match && match.length >= 3) {{
                    var brightness = parseInt(match[0]) * 0.299 +
                                     parseInt(match[1]) * 0.587 +
                                     parseInt(match[2]) * 0.114;
                    return brightness < 128 ? 'dark' : 'light';
                }}
            }} catch(e) {{}}
            return null;
        }}

        // 向聊天 iframe 推送主题
        function pushThemeToChat() {{
            var theme = detectStreamlitTheme();
            if (!theme) return;
            var iframe = doc.getElementById('chat-frame');
            if (iframe && iframe.contentWindow) {{
                try {{
                    iframe.contentWindow.postMessage({{
                        type: 'theme_update',
                        theme: theme
                    }}, '*');
                }} catch(e) {{}}
            }}
        }}

        // 监听聊天 iframe 发来的主题请求
        pwin._chatThemeHandler = function(event) {{
            if (!event.data || event.data.type !== 'request_theme') return;
            var iframe = doc.getElementById('chat-frame');
            if (!iframe || event.source !== iframe.contentWindow) return;
            // 延迟一点确保 iframe 已就绪
            pwin.setTimeout(pushThemeToChat, 100);
        }};
        pwin.addEventListener('message', pwin._chatThemeHandler);

        // 定期检测主题变化并推送 (每2秒检查一次，轻量级)
        if (pwin._chatThemeInterval) {{
            pwin.clearInterval(pwin._chatThemeInterval);
        }}
        var lastTheme = null;
        pwin._chatThemeInterval = pwin.setInterval(function() {{
            var currentTheme = detectStreamlitTheme();
            if (currentTheme && currentTheme !== lastTheme) {{
                lastTheme = currentTheme;
                pushThemeToChat();
            }}
        }}, 2000);

        // 首次也推送一次 (延迟1秒等 iframe 就绪)
        pwin.setTimeout(pushThemeToChat, 1000);
    }})();
    </script>
    """, height=1)


def render_floating_chat():
    """渲染右下角悬浮智能助手聊天窗口 (所有页面通用)"""
    # 确保独立聊天服务已启动
    _ensure_chat_server()

    # 注入样式 + HTML + JS (st.iframe 在 iframe 中执行 JS)
    _inject_chat_widget()


if __name__ == "__main__":
    main()
