"""
轻量级国际化模块 (i18n) - Soloquant UI 中英双语支持

设计原则:
1. 中文即 Key: t("📊 总览首页") 直接以中文文本作为查找键
2. 渐进式迁移: EN 词典缺失时原样返回中文，永不崩溃
3. 零第三方依赖: 仅使用 Streamlit session_state + 标准库
4. 自动语言检测: 浏览器 Accept-Language 优先，回退操作系统 locale

使用方式:
    from src.ui.i18n import t, get_language, set_language, render_language_selector

    st.title(t("📊 总览首页"))                    # 普通文本
    st.info(t("账户获取失败: {err}", err=e))       # 带参数插值 (替代 f-string)
    lang = get_language()                          # "zh" / "en"
"""

import os
import re
from pathlib import Path
import locale as _locale

# ═══════════════════════════════════════════════════════
# 语言常量
# ═══════════════════════════════════════════════════════

LANG_ZH = "zh"
LANG_EN = "en"
LANG_AUTO = "auto"

# 语言选择器显示选项 (显示名 -> 内部值)
LANGUAGE_OPTIONS = {
    "🌐 中文": LANG_ZH,
    "🌐 English": LANG_EN,
    "🌐 Auto": LANG_AUTO,
}

# session_state 中存储语言的键
_LANG_KEY = "ui_language"
# 用户显式选择过的标记 (区分 auto 默认值与用户主动选 auto)
_LANG_EXPLICIT_KEY = "ui_language_explicit"


# ═══════════════════════════════════════════════════════
# 语言偏好持久化 (JSON 文件, 跨会话保存)
# ═══════════════════════════════════════════════════════

def _get_lang_file() -> Path:
    """语言偏好文件路径: <项目根>/data/ui_language.json"""
    try:
        from src.utils.config import config
        return config.DATA_DIR / "ui_language.json"
    except Exception:
        return Path("data") / "ui_language.json"


def _load_persisted_lang() -> str:
    """从本地 JSON 读取持久化的语言偏好, 无记录返回空串"""
    import json
    try:
        fp = _get_lang_file()
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            lang = data.get("language", "")
            return lang if lang in (LANG_ZH, LANG_EN, LANG_AUTO) else ""
    except Exception:
        pass
    return ""


def _save_persisted_lang(lang: str) -> None:
    """保存语言偏好到本地 JSON (静默失败, 不影响 UI)"""
    import json
    try:
        fp = _get_lang_file()
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump({"language": lang}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════
# 语言检测
# ═══════════════════════════════════════════════════════

def _detect_browser_language() -> str:
    """
    从 Streamlit 上下文读取浏览器 Accept-Language 头

    Returns:
        "zh" / "en" / "" (检测失败返回空串)
    """
    try:
        import streamlit as st
        # Streamlit >= 1.37 提供 st.context.headers
        ctx = getattr(st, "context", None)
        if ctx is not None:
            headers = getattr(ctx, "headers", None)
            if headers:
                accept = headers.get("Accept-Language", "") or headers.get("accept-language", "")
                if accept:
                    return _parse_accept_language(accept)
    except Exception:
        pass
    return ""


def _parse_accept_language(accept: str) -> str:
    """
    解析 Accept-Language 头, 如 "zh-CN,zh;q=0.9,en;q=0.8"

    中文系 (zh) 优先级最高时返回 "zh", 否则返回 "en"
    """
    if not accept:
        return ""
    # 取第一个语言标签 (最高优先级)
    first = accept.split(",")[0].strip().lower()
    if first.startswith("zh"):
        return LANG_ZH
    if first.startswith("en"):
        return LANG_EN
    # 其他语言默认英文 (国际化惯例)
    return LANG_EN


def _detect_os_language() -> str:
    """
    检测操作系统语言 (回退方案)

    Returns:
        "zh" / "en"
    """
    # 1. LANG 环境变量 (Linux/macOS)
    lang_env = os.environ.get("LANG", "") or os.environ.get("LANGUAGE", "") or os.environ.get("LC_ALL", "")
    if lang_env:
        if lang_env.lower().startswith("zh"):
            return LANG_ZH
        return LANG_EN
    # 2. Windows locale
    try:
        windir = os.environ.get("WINDIR", "")
        if windir:  # Windows 系统
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # GetUserDefaultUILanguage 返回 LANGID, 中文=0x0804/0x0404, 英文=0x0409
            lang_id = kernel32.GetUserDefaultUILanguage()
            primary = lang_id & 0xFF  # 主语言 ID: 中文=0x04, 英文=0x09
            if primary == 0x04:
                return LANG_ZH
            return LANG_EN
    except Exception:
        pass
    # 3. Python locale 模块兜底
    try:
        loc = _locale.getdefaultlocale()[0] or ""
        if loc.lower().startswith("zh"):
            return LANG_ZH
    except Exception:
        pass
    return LANG_ZH  # 最终默认中文 (项目主要用户群)


def _resolve_auto_language() -> str:
    """auto 模式: 浏览器语言优先, 失败回退 OS 语言"""
    browser = _detect_browser_language()
    if browser:
        return browser
    return _detect_os_language()


# ═══════════════════════════════════════════════════════
# 语言状态管理
# ═══════════════════════════════════════════════════════

# 主线程语言快照缓存: 供工作线程 (parallel_map 等) 回退使用
# 解决: 工作线程中 st.session_state 不可访问, 导致语言检测回退到 OS 语言 (中文系统误判为 zh)
_lang_snapshot: str = ""
_lang_snapshot_checked: bool = False


def get_language() -> str:
    """
    获取当前 UI 语言

    优先级: 环境变量 > session_state 显式选择 > 主线程快照缓存 > auto 检测
    工作线程中 session_state 不可访问时, 回退到主线程已缓存的语言快照

    Returns:
        "zh" 或 "en"
    """
    global _lang_snapshot, _lang_snapshot_checked

    # 环境变量覆盖 (CLI 场景 / 测试): SOLOQUANT_UI_LANG 显式设置优先
    env = os.environ.get("SOLOQUANT_UI_LANG", "")
    if env in (LANG_ZH, LANG_EN):
        return env

    # 判断当前线程是否为 Streamlit 脚本线程 (工作线程中 session_state 是空代理, 读取无意义)
    in_script_ctx = False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_script_ctx = get_script_run_ctx() is not None
    except Exception:
        in_script_ctx = False

    if in_script_ctx:
        # Streamlit 脚本线程: 正常读取 session_state
        try:
            import streamlit as st
            saved = st.session_state.get(_LANG_KEY, "")
            if not saved:
                # session_state 无值 (首次启动): 读取持久化文件
                saved = _load_persisted_lang()
                if saved:
                    st.session_state[_LANG_KEY] = saved
                    st.session_state[_LANG_EXPLICIT_KEY] = True
            if saved == LANG_AUTO or (not saved and not st.session_state.get(_LANG_EXPLICIT_KEY)):
                # auto 模式或从未显式选择: 动态检测
                lang = _resolve_auto_language()
            elif saved in (LANG_ZH, LANG_EN):
                lang = saved
            else:
                lang = _resolve_auto_language()
            # 成功读取: 更新快照缓存供工作线程使用
            if lang in (LANG_ZH, LANG_EN):
                _lang_snapshot = lang
                _lang_snapshot_checked = True
            return lang
        except Exception:
            pass

    # 工作线程 / 无 Streamlit 运行时:
    # 优先用主线程语言快照, 避免回退到 OS 语言导致与 UI 不一致
    if _lang_snapshot_checked and _lang_snapshot in (LANG_ZH, LANG_EN):
        return _lang_snapshot
    # 快照未就绪 (如 chat_server 后台线程先于 UI 线程运行): 读取持久化偏好
    persisted = _load_persisted_lang()
    if persisted in (LANG_ZH, LANG_EN):
        _lang_snapshot = persisted
        _lang_snapshot_checked = True
        return persisted
    return _detect_os_language()


def set_language(lang: str) -> None:
    """
    设置 UI 语言 (用户手动切换)

    Args:
        lang: "zh" / "en" / "auto"
    """
    try:
        import streamlit as st
        st.session_state[_LANG_KEY] = lang
        st.session_state[_LANG_EXPLICIT_KEY] = True
    except Exception:
        # 无 Streamlit 环境时写入环境变量 (CLI 场景)
        os.environ["SOLOQUANT_UI_LANG"] = lang
    # 持久化到本地文件 (跨会话保存)
    _save_persisted_lang(lang)


def get_cli_language() -> str:
    """
    CLI 场景语言获取: 环境变量 SOLOQUANT_UI_LANG > OS 检测

    Returns:
        "zh" 或 "en"
    """
    env = os.environ.get("SOLOQUANT_UI_LANG", "")
    if env in (LANG_ZH, LANG_EN):
        return env
    if env == LANG_AUTO:
        return _detect_os_language()
    return _detect_os_language()


def render_language_selector() -> None:
    """
    在 Streamlit 侧边栏渲染语言选择器

    选择结果存入 session_state, 切换后整页 rerun 生效
    """
    try:
        import streamlit as st
    except ImportError:
        return

    current = st.session_state.get(_LANG_KEY, LANG_AUTO)
    # 反查显示名
    display = next((k for k, v in LANGUAGE_OPTIONS.items() if v == current), "🌐 Auto")

    selected = st.sidebar.selectbox(
        "🌐 Language / 语言",
        list(LANGUAGE_OPTIONS.keys()),
        index=list(LANGUAGE_OPTIONS.keys()).index(display),
        key="ui_language_selector",
    )

    new_lang = LANGUAGE_OPTIONS.get(selected, LANG_AUTO)
    if new_lang != current:
        st.session_state[_LANG_KEY] = new_lang
        st.session_state[_LANG_EXPLICIT_KEY] = True
        _save_persisted_lang(new_lang)  # 持久化 (跨会话保存)
        st.rerun()


# ═══════════════════════════════════════════════════════
# 翻译核心
# ═══════════════════════════════════════════════════════

# 占位符提取: {var} / {var:.0%} / {var:,.2f} 等
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_]\w*)(?::[^{}]*)?\}")


def t(key: str, **kwargs) -> str:
    """
    翻译函数: 中文 key -> 当前语言文本

    - 中文模式: 直接返回 key (中文即源文本)
    - 英文模式: 查 EN 词典, 命中则用英文模板, 未命中回退中文
    - kwargs: 占位符插值, 如 t("赚了 {pct}", pct="10%")

    Args:
        key: 中文源文本 (可含 {var} 占位符)
        **kwargs: 占位符变量

    Returns:
        翻译后的字符串 (永远返回 str, 不抛异常)
    """
    if not isinstance(key, str):
        return str(key)

    lang = get_language()
    if lang == LANG_ZH:
        # 中文模式: 原样返回 (含占位符插值)
        return _interpolate(key, kwargs)

    # 英文模式: 查词典
    template = EN.get(key)
    if template is None:
        # 词典缺失: 回退中文原文
        return _interpolate(key, kwargs)
    return _interpolate(template, kwargs)


def _interpolate(template: str, kwargs: dict) -> str:
    """
    安全的占位符插值 (str.format 的容错版)

    - 只替换 kwargs 中存在的变量
    - 格式说明符 ({x:.0%}) 正常生效
    - 缺失变量保留原占位符文本, 不抛 KeyError
    """
    if not kwargs:
        return template
    if not _PLACEHOLDER_RE.search(template):
        return template

    def _sub(m):
        var = m.group(1)
        if var in kwargs:
            # 重新构造带格式说明符的占位符并 format
            try:
                return ("{" + var + m.group(0)[len(m.group(1)) + 1:]).format(**{var: kwargs[var]})
            except Exception:
                return str(kwargs[var])
        return m.group(0)  # 缺失变量保留原样

    return _PLACEHOLDER_RE.sub(_sub, template)


def t_list(items: list) -> list:
    """批量翻译列表 (用于 selectbox options 显示)"""
    return [t(x) if isinstance(x, str) else x for x in items]


# ═══════════════════════════════════════════════════════
# EN 词典 (中文 key -> English)
# ═══════════════════════════════════════════════════════
# 注意: 逻辑值字符串 (页面导航键/周期键/行业值等) 不在此词典中,
#       它们通过 format_func=t 在显示层翻译, 内部逻辑保持中文键。
# ═══════════════════════════════════════════════════════

EN: dict = {
    # ── 通用 ──
    "已连接": "Connected",
    "已配置": "Configured",
    "未安装": "Not installed",
    "可用": "Available",
    "不可用": "Unavailable",
    "运行中": "Running",
    "未启动": "Not started",
    "正常": "Normal",
    "异常": "Error",
    "加载中...": "Loading...",
    "暂无数据": "No data",
    "(空)": "(empty)",
    "是": "Yes",
    "否": "No",
    "确定": "OK",
    "取消": "Cancel",
    "删除": "Delete",
    "保存": "Save",
    "刷新": "Refresh",
    "重试": "Retry",
    "帮助": "Help",
    "退出": "Exit",
    "全部": "All",
    "其他": "Other",
    "无": "None",
    "不限": "Unlimited",
    "0 = 不限": "0 = unlimited",
    "一键": "One-click",
    "下单": "Place order",
    "买入": "Buy",
    "卖出": "Sell",
    "查看持仓": "View positions",
    "系统状态": "System status",
    "紧急清仓": "Emergency liquidation",
    "当前配置": "Current config",
    "执行交易信号": "Execute trade signals",
    "全部卖出": "Sell all",

    # ── 侧边栏 ──
    "📝 模拟盘": "📝 Paper",
    "🔴 实盘": "🔴 Live",
    "📝 模拟盘 (Paper)": "📝 Paper",
    "🔴 实盘 (Live)": "🔴 Live",
    "🖥️ 本地": "🖥️ Local",
    "☁️ 云端": "☁️ Cloud",
    "🔌 数据源切换": "🔌 Data Source",
    "数据来源": "Data source",
    "本地直连": "Local direct",
    "云端API": "Cloud API",
    "云端 API 地址": "Cloud API URL",
    "✅ 云端连接正常": "✅ Cloud connection OK",
    "⚠️ 无法连接云端": "⚠️ Cannot connect to cloud",
    "⚠️ 请输入云端 API 地址": "⚠️ Please enter cloud API URL",
    "💻 本地直连模式": "💻 Local direct mode",
    "📡 数据源": "📡 Data Sources",
    "🛡️ 风控": "🛡️ Risk Control",
    "单票仓位": "Max position",
    "日最大亏损": "Max daily loss",
    "最大回撤": "Max drawdown",
    "🎯 选股": "🎯 Stock Screening",
    "候选上限": "Candidate limit",
    "导航": "Navigation",

    # ── 页面导航 (显示层, 逻辑键保持中文) ──
    "📊 总览首页": "📊 Overview",
    "📈 行情分析": "📈 Market Analysis",
    "🎯 智能选股": "🎯 Smart Screening",
    "📈 策略管理": "📈 Strategies",
    "💰 交易记录": "💰 Trade Records",
    "🌐 宏观指标": "🌐 Macro Indicators",
    "📊 期权数据": "📊 Options",
    "🪙 加密货币": "🪙 Crypto",
    "🧠 模型训练": "🧠 Model Training",
    "⚙️ 系统设置": "⚙️ Settings",

    # ── 数据表列名 ──
    "代码": "Symbol",
    "名称": "Name",
    "价格": "Price",
    "来源": "Source",
    "时间": "Time",
    "级别": "Level",
    "标题": "Title",
    "内容": "Content",
    "行业": "Sector",
    "最新价 ($)": "Price ($)",
    "开盘 ($)": "Open ($)",
    "最高 ($)": "High ($)",
    "最低 ($)": "Low ($)",
    "昨收 ($)": "Prev Close ($)",
    "成交量": "Volume",
    "营收增长": "Rev Growth",
    "毛利率": "Gross Margin",
    "综合评分": "Composite",
    "技术面": "Technical",
    "情绪面": "Sentiment",
    "AI 推荐理由": "AI Reason",
    "项目": "Item",
    "详情": "Details",
    "排名": "Rank",
    "价格 ($)": "Price ($)",
    "策略ID": "Strategy ID",
    "模型": "Model",
    "状态": "Status",
    "股票池": "Stock Pool",
    "因子数": "Factors",
    "最大仓位": "Max Position",
    "调仓频率": "Rebalance Freq",
    "创建时间": "Created",
    "权重": "Weight",
    "情绪分": "Sentiment",
    "提及次数": "Mentions",
    "推荐理由": "Reason",
    "情绪": "Sentiment",
    "新闻数": "News",
    "指标": "Indicator",
    "最新值": "Latest",
    "单位": "Unit",
    "环比": "MoM",
    "同比": "YoY",
    "更新日期": "Updated",
    "频率": "Frequency",
    "币种": "Coin",
    "最新价格 ($)": "Price ($)",
    "24h涨跌幅": "24h Change %",
    "24h成交量": "24h Volume",
    "❌ 不可用": "❌ Unavailable",
    "文件名": "File",
    "模型类型": "Model Type",
    "大小 (KB)": "Size (KB)",
    "修改时间": "Modified",
    "路径": "Path",
    "实验ID": "Experiment ID",
    "训练 IC": "Train IC",
    "验证 IC": "Valid IC",
    "训练 Rank IC": "Train Rank IC",
    "验证 Rank IC": "Valid Rank IC",
    "模型路径": "Model Path",
    "是否最佳": "Best",
    "是否有效": "Effective",
    "因子名称": "Factor",
    "IR (信息比率)": "IR",
    "训练": "Training",
    "验证": "Validation",
    "夏普": "Sharpe",
    "夏普比率": "Sharpe Ratio",

    # ── 周期/时间范围 (显示层) ──
    "1分": "1m",
    "5分": "5m",
    "15分": "15m",
    "30分": "30m",
    "60分": "60m",
    "日K": "Daily",
    "周K": "Weekly",
    "月K": "Monthly",
    "1周": "1W",
    "1月": "1M",
    "3月": "3M",
    "半年": "6M",
    "1年": "1Y",
    "3年": "3Y",
    "5年": "5Y",
    "周期": "Period",
    "时间范围": "Time Range",
    "⏱️ K线周期": "⏱️ K-line Period",
    "📅 时间范围": "📅 Time Range",
    "日内1分钟K线 (最近7天)": "Intraday 1-min candles (last 7 days)",
    "日内5分钟K线 (最近60天)": "Intraday 5-min candles (last 60 days)",
    "日内15分钟K线 (最近60天)": "Intraday 15-min candles (last 60 days)",
    "日内30分钟K线 (最近60天)": "Intraday 30-min candles (last 60 days)",
    "日内60分钟K线 (最近60天)": "Intraday 60-min candles (last 60 days)",
    "日线K线 (前复权)": "Daily candles (adjusted)",
    "周线K线": "Weekly candles",
    "月线K线": "Monthly candles",

    # ── 行业 (显示层, 逻辑值保持中文) ──
    "金融": "Finance",
    "能源": "Energy",
    "医疗": "Healthcare",
    "消费": "Consumer",
    "行业偏好": "Sector preference",
    "关键词": "Keywords",
    "如: 潜力/成长/低价": "e.g. growth/value/low-price",
    "如: AI芯片, 新能源, 半导体": "e.g. AI chips, renewables, semiconductors",
    "💡 参数会自动从自然语言中提取，也可手动调整": "💡 Parameters are auto-extracted from natural language, or adjust manually",

    # ── 权重分配 (显示层) ──
    "等权分配": "Equal Weight",
    "风险平价": "Risk Parity",
    "绩效加权": "Performance Weighted",
    "权重分配方式": "Allocation method",
    "风险预算": "Risk budget",
    "选择策略 (至少2个)": "Select strategies (min 2)",
    "请至少选择 2 个策略": "Please select at least 2 strategies",
    "✅ 组合 '{name}' 创建成功! (ID: {id})": "✅ Portfolio '{name}' created! (ID: {id})",

    # ── 实时行情 ──
    "▶️ 启动实时行情": "▶️ Start realtime quotes",
    "⚡ 实时行情": "⚡ Realtime Quotes",
    "实时连接": "Realtime connection",

    # ── 模型训练 ──
    "➕ 新建模型": "➕ New Model",
    "选择模型编辑": "Select model to edit",
    "多模型竞赛": "Model Competition",
    "多模型 IC / Rank IC 对比": "Multi-model IC / Rank IC comparison",
    "学习率": "Learning rate",
    "子采样": "Subsample",
    "LSTM层数": "LSTM layers",
    "IC 值": "IC value",
    "IC 阈值": "IC threshold",
    "IR 值": "IR value",
    "模型文件数": "Model files",
    "展示已训练模型文件列表": "Show trained model files",
    "将日线数据重采样为周线/月线": "Resample daily data to weekly/monthly",
    "将选股结果序列化保存到本地 JSON": "Save screening results to local JSON",
    "如: mymodel1": "e.g. mymodel1",
    "如: mymodel1 (英文字母数字)": "e.g. mymodel1 (alphanumeric)",
    "如: deepseek-chat, gpt-4o-mini, doubao-seed-2-0-mini-260428": "e.g. deepseek-chat, gpt-4o-mini",
    "如: https://api.deepseek.com/v1 或 https://ark.cn-beijing.volces.com/api/v3": "e.g. https://api.deepseek.com/v1",
    "参与竞赛的模型": "Models in competition",
    "选择模型": "Select model",
    "选择要删除的模型": "Select model to delete",
    "未保存": "Not saved",

    # ── 宏观 ──
    "📋 核心宏观指标": "📋 Core Macro Indicators",
    "📈 通胀指标": "📈 Inflation",
    "👷 就业指标": "👷 Employment",
    "📊 经济增长": "📊 Growth",
    "🏦 利率指标": "🏦 Interest Rates",
    "📉 美债收益率": "📉 Treasury Yields",
    "通胀": "Inflation",
    "就业": "Employment",
    "增长": "Growth",
    "利率": "Rates",
    "收益率": "Yields",
    "宏观经济指标页面": "Macro indicators page",

    # ── 期权 ──
    "选择到期日": "Select expiry",
    "实值": "In the money",
    "🚫 止损线": "🚫 Stop-loss line",

    # ── 加密货币 ──
    "交易对": "Pair",
    "数量": "Quantity",
    "🛒 手动交易": "🛒 Manual Trade",
    "🛡️ 加密货币风控": "🛡️ Crypto Risk Control",
    "🛡️ 单币仓位上限": "🛡️ Per-coin position limit",
    "🪙 加密货币配置": "🪙 Crypto Settings",

    # ── 风险等级 ──
    "🟡 中风险": "🟡 Medium risk",
    "🔴 高风险": "🔴 High risk",
    "🟢 正常": "🟢 Normal",
    "🟠 偏悲观": "🟠 Bearish",
    "🟠 偏空": "🟠 Bearish",
    "🟡 中性偏多": "🟡 Neutral-bullish",
    "🟡 中性偏空": "🟡 Neutral-bearish",
    "🟡 微乐观": "🟡 Slightly bullish",
    "🟢 偏乐观": "🟢 Bullish",
    "🟢 偏多": "🟢 Bullish",
    "🟢 极度看多": "🟢 Extremely bullish",
    "🟢 低风险": "🟢 Low risk",

    # ── 交易/账户 ──
    "账户获取失败: {err}": "Failed to get account: {err}",
    "云端API返回空数据 (检查API地址配置)": "Cloud API returned empty data (check API URL config)",
    "账户数据为空": "Account data is empty",
    "当前模式": "Current mode",
    "交易模式: paper=模拟盘, live=实盘": "Trading mode: paper / live",
    "日志级别": "Log level",
    "部署模式": "Deploy mode",
    "CRYPTO_DATA_SOURCE (数据源)": "CRYPTO_DATA_SOURCE",

    # ── 工作流 ──
    "选择工作流": "Select workflow",
    "工作流引擎未初始化": "Workflow engine not initialized",
    "工作流执行失败: {e}": "Workflow execution failed: {e}",

    # ── 实盘迁移 ──
    "实盘迁移模块未就绪: {e}": "Live migration module not ready: {e}",
    "✅ 确认迁移": "✅ Confirm migration",
    "✅ 确认清仓": "✅ Confirm liquidation",
    "🚨 紧急止损": "🚨 Emergency Stop",

    # ── LLM ──
    "🧠 LLM 智能助手配置": "🧠 LLM Assistant Settings",
    "🧠 LLM 配置已重新加载": "🧠 LLM config reloaded",
    "🧠 LLM 智能对话已启用 (模型: {model})": "🧠 LLM chat enabled (model: {model})",
    "🧠 AI 解析结果: 数量={count}, ": "🧠 AI parsed: count={count}, ",
    "LLM: 📝 规则模式": "LLM: 📝 Rule mode",
    "LLM: 📝 规则模式 (未配置)": "LLM: 📝 Rule mode (not configured)",
    "对话上下文已重置": "Dialogue context reset",
    "对话处理异常: {e}": "Dialogue error: {e}",

    # ── 通知/告警 ──
    "🛡️ 风控体检报告:": "🛡️ Risk check report:",

    # ── 关注列表 ──
    "⭐ 关注列表": "⭐ Watchlist",
    "已关注 {symbol}": "Added {symbol} to watchlist",
    "已取消关注 {symbol}": "Removed {symbol} from watchlist",
    "已关注股票数": "Watchlist count",
    "上次再平衡": "Last rebalance",

    # ── 图表 ──
    "持仓占比分布": "Position allocation",
    "浮动盈亏 ($)": "Unrealized P&L ($)",
    "股票": "Stock",
    "盈亏 ($)": "P&L ($)",
    "均线": "Moving averages",
    "均线系统": "Moving averages",
    "K线": "Candles",
    "BOLL上轨": "BOLL upper",
    "BOLL中轨": "BOLL middle",
    "BOLL下轨": "BOLL lower",
    "MACD柱": "MACD hist",

    # ── CLI ──
    "💬 你: ": "💬 You: ",
    "请选择 (0-9, 或输入 g 进行AI自主交易): ": "Select (0-9, or g for AI auto-trading): ",
    "你的目标: ": "Your goal: ",
    "确认执行? (输入 yes 确认): ": "Confirm? (type yes): ",
    "输入股票代码 (逗号分隔, 如 AAPL,MSFT,GOOGL): ": "Enter symbols (comma-separated, e.g. AAPL,MSFT,GOOGL): ",
    "👋 再见！": "👋 Bye!",
    "📋 功能菜单": "📋 Menu",
    "🎯  AI 自主交易 - 告诉我目标和资金，全自动搞定": "🎯  AI Auto-Trading - tell me your goal & capital",
    "1️⃣  市场调研 - 看看今天有什么机会": "1️⃣  Market research - today's opportunities",
    "2️⃣  策略生成 - AI 自动生成交易策略": "2️⃣  Strategy - AI generates trading strategies",
    "3️⃣  回测验证 - 测试策略历史表现": "3️⃣  Backtest - test strategy on history",
    "4️⃣  执行交易 - 模拟盘下单": "4️⃣  Execute - place paper orders",
    "5️⃣  持仓监控 - 查看持仓和风控": "5️⃣  Monitor - positions & risk",
    "6️⃣  一键全自动 - 全流程自动运转": "6️⃣  Full auto - end-to-end pipeline",
    "7️⃣  紧急止损 - 清仓所有持仓": "7️⃣  Emergency - liquidate all",
    "8️⃣  初始化 Qlib 数据": "8️⃣  Init Qlib data",
    "9️⃣  启动仪表盘 - 可视化界面": "9️⃣  Launch dashboard",
    "🔟  设置关注股票": "🔟  Set watchlist",
    "0️⃣  退出": "0️⃣  Exit",
    "抱歉，处理你的请求时出现了错误: {e}": "Sorry, an error occurred: {e}",
    "请稍后重试，或输入 \"帮助\" 查看可用功能。": "Please retry, or type \"help\".",

    # ── 聊天页面 (chat_page.html 由 JS 词典处理, 此处供后端 SSE 文本) ──
    "🤔 我不太理解你的意思。\n\n": "🤔 I didn't quite understand that.\n\n",
    "🤖 AI 量化交易助手 - 使用指南\n": "🤖 AI Quant Assistant - Guide\n",

    # ── nl_router 响应 ──
    "监控失败，请检查 Alpaca 配置": "Monitor failed, check Alpaca config",
    "❌ 监控失败，请检查 Alpaca 配置。": "❌ Monitor failed, check Alpaca config.",
    "🔄 重试": "🔄 Retry",
    "❓ 帮助": "❓ Help",
    "📊 生成策略": "📊 Generate strategy",
    "🔍 换个板块": "🔍 Try another sector",
    "📈 查看持仓": "📈 View positions",
    "📊 调研这些股票": "📊 Research these stocks",
    "🔄 重新选股": "🔄 Re-screen",
    "📈 回测验证": "📈 Backtest",
    "🚀 执行交易": "🚀 Execute trade",
    "🔄 优化策略": "🔄 Optimize strategy",
    "🔄 重新生成策略": "🔄 Regenerate strategy",
    "📰 生成日报": "📰 Daily report",
    "🔍 市场调研": "🔍 Market research",
    "🎯 AI自主交易": "🎯 AI auto-trading",

    # ── plain_explainer 术语 ──
    "每年平均赚多少": "Average annual gain",
    "风险调整后的收益（越高越好）": "Risk-adjusted return (higher is better)",
    "最惨的时候亏多少": "Worst-case loss",
    "赚钱的交易占多少比例": "Share of profitable trades",
    "因子预测准不准的指标": "Factor predictive accuracy metric",
    "一只股票占总资金的比例": "Share of total capital in one stock",
    "亏到一定程度自动卖出": "Auto-sell at a loss threshold",
    "赚到一定程度自动卖出": "Auto-sell at a profit threshold",
    "从最高点跌下来多少": "Drop from the peak",
    "价格上下波动的剧烈程度": "Price volatility magnitude",
    "只考虑亏损的夏普比率": "Sharpe using downside only",
    "模拟盘（不用真钱）": "Paper trading (no real money)",
    "微软开源的量化框架": "Microsoft's open-source quant framework",
    "一种机器学习模型": "A machine learning model",
    "微软Qlib的158个因子库": "Qlib's 158-factor library",
    "比较安全，但收益可能也有限": "Relatively safe, but returns may be limited",
    "有一定波动，适合大多数投资者": "Some volatility, suits most investors",
    "波动较大，请谨慎对待": "High volatility, proceed with caution",
    "暂无解释: {term}": "No explanation: {term}",
    "⭐⭐⭐⭐⭐ 非常优秀！": "⭐⭐⭐⭐⭐ Excellent!",
    "⭐⭐⭐⭐ 表现不错": "⭐⭐⭐⭐ Good",
    "⭐⭐⭐ 一般般": "⭐⭐⭐ Average",
    "⭐⭐ 表现不太好": "⭐⭐ Poor",
    "评级: {grade}": "Rating: {grade}",
    "💰 每年平均赚 {ar}，非常厉害！": "💰 Earns {ar} per year on average, impressive!",
    "💰 每年平均赚 {ar}，比存银行好不少": "💰 Earns {ar} per year, much better than a bank",
    "💰 每年平均赚 {ar}，勉强跑赢银行存款": "💰 Earns {ar} per year, barely beats a bank",
    "💰 每年平均赚 {ar}，不太理想": "💰 Earns {ar} per year, not great",
    "📉 最惨的时候只亏了 {mdd}，很安全": "📉 Worst loss was only {mdd}, very safe",
    "📉 最惨的时候亏了 {mdd}，可以接受": "📉 Worst loss was {mdd}, acceptable",
    "📉 最惨的时候亏了 {mdd}，需要心理准备": "📉 Worst loss was {mdd}, brace yourself",
    "📊 风险控制能力: 优秀 (夏普 {sr})": "📊 Risk control: excellent (Sharpe {sr})",
    "📊 风险控制能力: 不错 (夏普 {sr})": "📊 Risk control: good (Sharpe {sr})",
    "📊 风险控制能力: 一般 (夏普 {sr})": "📊 Risk control: average (Sharpe {sr})",
    "📊 风险控制能力: 需改善 (夏普 {sr})": "📊 Risk control: needs work (Sharpe {sr})",
    "🎯 每10次交易赢 {wr} 次，胜率较高": "🎯 Wins {wr} of 10 trades, high win rate",
    "🎯 每10次交易赢 {wr} 次，还行": "🎯 Wins {wr} of 10 trades, decent",
    "🎯 每10次交易赢 {wr} 次，胜率偏低": "🎯 Wins {wr} of 10 trades, low win rate",
    "⚠️ 记住: 过去的成绩不代表未来，投资有风险！": "⚠️ Remember: past performance ≠ future results. Investing is risky!",
    "小仓位 ({weight})，先试试水": "Small position ({weight}), test the waters",

    # ── 自动补全 (i18n_convert 生成) ──
    '\n\n👋 再见！': '\n\n👋 Bye!',
    '\n\n👋 再见！祝投资顺利！': '\n\n👋 Bye! Happy investing！',
    '\n        **Alpha158 因子库** 包含以下类别:\n\n        - **价格类**: KMID, KLEN, KUPP, KLOW 等\n        - **动量类**: ROC, MA, STD, BETA 等\n        - **成交量类**: VSTD, WVMA, VSUMP 等\n        - **技术指标类**: RSI, KDJ, MACD, CCI 等\n\n        **筛选流程:**\n        1. 计算每个因子的 IC / Rank IC / IR\n        2. 按 |IR| 排序\n        3. 去相关 (因子间相关性 < 阈值)\n        4. 返回 Top-N 有效因子\n\n        ⚠️ 因子搜索需要 Qlib 数据支持\n        ': '\n        **Alpha158 factor library** categories:\n\n        - **Price**: KMID, KLEN, KUPP, KLOW, etc.\n        - **Momentum**: ROC, MA, STD, BETA, etc.\n        - **Volume**: VSTD, WVMA, VSUMP, etc.\n        - **Technical**: RSI, KDJ, MACD, CCI, etc.\n\n        **Screening workflow:**\n        1. Compute IC / Rank IC / IR for each factor\n        2. Sort by |IR|\n        3. Decorrelate (inter-factor correlation < threshold)\n        4. Return top-N effective factors\n\n        ⚠️ Factor search requires Qlib data\n        ',
    '\n        **Alpha158 因子库** 包含以下类别:\n\n        - **价格类**: KMID, KLEN, KUPP, KLOW 等\n        - **成交量类**: VOLUME, VSTD, VWAP 等\n        - **滚动窗口类**: ROC, MA, STD, BETA 等\n        - **交互类**: CNTP, CNTP5, SUMP5 等\n        \n        搜索将计算每个因子的 IC/IR 并排名。': '\n        **Alpha158 factor library** categories:\n\n        - **Price**: KMID, KLEN, KUPP, KLOW, etc.\n        - **Volume**: VOLUME, VSTD, VWAP, etc.\n        - **Rolling window**: ROC, MA, STD, BETA, etc.\n        - **Interaction**: CNTP, CNTP5, SUMP5, etc.\n        \n        The search computes IC/IR for each factor and ranks them.',
    '\n        **PCR (Put-Call Ratio) 看跌看涨比率:**\n        - PCR < 0.5: 极度看多 (市场情绪非常乐观)\n        - PCR 0.5-0.7: 偏多 (情绪乐观)\n        - PCR 0.7-1.0: 中性偏多\n        - PCR 1.0-1.3: 中性偏空\n        - PCR > 1.3: 偏空 (情绪悲观)\n        - PCR > 1.5: 极度看空 (市场恐慌)\n        \n        **解读:** PCR 基于成交量时反映短期交易情绪，基于未平仓合约时反映中期持仓倾向。': '\n        **PCR (Put-Call Ratio):**\n        - PCR < 0.5: Extremely bullish (very optimistic)\n        - PCR 0.5-0.7: Bullish (optimistic)\n        - PCR 0.7-1.0: Neutral-bullish\n        - PCR 1.0-1.3: Neutral-bearish\n        - PCR > 1.3: Bearish (pessimistic)\n        - PCR > 1.5: Extremely bearish (panic)\n        \n        **Reading:** Volume-based PCR reflects short-term sentiment; open-interest-based PCR reflects medium-term positioning.',
    '\n        **PCR (Put-Call Ratio) 看跌看涨比率:**\n        - PCR < 0.5: 极度看多 (市场情绪非常乐观)\n        - PCR 0.5-0.8: 偏多\n        - PCR 0.8-1.0: 中性偏多\n        - PCR 1.0-1.2: 中性偏空\n        - PCR 1.2-1.5: 偏空\n        - PCR > 1.5: 极度看空 (可能为过度恐慌，反而视为反向指标)\n\n        **隐含波动率 (IV):**\n        - IV 越高表示市场预期未来波动越大\n        - IV 低于 20% 为低波动环境\n        - IV 高于 50% 为高波动环境\n        - 隐含波动率微笑曲线显示不同行权价的 IV 差异\n\n        **成交量 vs 未平仓量:**\n        - 成交量: 当日新交易的合约数\n        - 未平仓量: 尚未结算的合约总数\n        - 高未平仓量的行权价通常是关键支撑/阻力位\n\n        ⚠️ 期权数据仅供参考，不构成投资建议。\n        ': '\n        **PCR (Put-Call Ratio):**\n        - PCR < 0.5: Extremely bullish (very optimistic)\n        - PCR 0.5-0.8: Bullish\n        - PCR 0.8-1.0: Neutral-bullish\n        - PCR 1.0-1.2: Neutral-bearish\n        - PCR 1.2-1.5: Bearish\n        - PCR > 1.5: Extremely bearish (may signal panic — contrarian indicator)\n\n        **Implied Volatility (IV):**\n        - Higher IV = larger expected future swings\n        - IV below 20% = low-volatility regime\n        - IV above 50% = high-volatility regime\n        - The IV smile shows IV differences across strikes\n\n        **Volume vs Open Interest:**\n        - Volume: contracts traded that day\n        - Open interest: total unsettled contracts\n        - Strikes with high OI are often key support/resistance\n\n        ⚠️ Options data is for reference only, not investment advice.\n        ',
    '\n        **多模型竞赛流程:**\n        1. 同时训练选中的模型\n        2. 每个模型使用相同的数据和因子\n        3. 比较验证集 IC / Rank IC\n        4. 自动选出 |IC| 最大的模型\n\n        **模型特点:**\n        - **LightGBM**: 速度快，适合表格数据\n        - **LSTM**: 捕捉时序依赖，需 PyTorch\n        - **Linear**: 最简基线，快速对比\n\n        ⚠️ LSTM 训练时间较长 (5-15分钟)\n        ': '\n        **Model competition workflow:**\n        1. Train selected models simultaneously\n        2. Each model uses the same data and factors\n        3. Compare validation IC / Rank IC\n        4. The model with the largest |IC| wins automatically\n\n        **Model traits:**\n        - **LightGBM**: fast, great for tabular data\n        - **LSTM**: captures temporal dependencies, needs PyTorch\n        - **Linear**: simplest baseline, quick comparison\n\n        ⚠️ LSTM training takes longer (5-15 min)\n        ',
    '\n        **多模型竞赛流程:**\n        1. 同时训练选中的模型\n        2. 每个模型使用相同的数据和因子\n        3. 比较验证集 IC / Rank IC\n        4. 自动选出最佳模型': '\n        **Model competition workflow:**\n        1. Train selected models simultaneously\n        2. Each model uses the same data and factors\n        3. Compare validation IC / Rank IC\n        4. Best model is selected automatically',
    '\n  ⏰ 正在启动定时任务调度器...': '\n  ⏰ Starting scheduler...',
    '\n  ⏹️ API 已停止。': '\n  ⏹️ API stopped。',
    '\n  ☁️ 启动云端服务...': '\n  ☁️ Starting cloud services...',
    '\n  ⚠️ 实时行情未能启动 (可能 API Key 无效或网络问题)': '  ⚠️ Failed to start realtime quotes (invalid API key or network issue)',
    '\n  ⚡ 正在启动实时行情 WebSocket...': '\n  ⚡ Starting realtime WebSocket...',
    '\n  仪表盘已退出。': '\n  Dashboard exited。',
    '\n  按 Ctrl+C 停止调度器\n': '\n  Press Ctrl+C to stop the scheduler\n',
    '\n  正在停止实时行情...': '\n  Stopping realtime quotes...',
    '\n  正在停止所有服务...': '\n  Stopping all services...',
    '\n  正在停止调度器...': '\n  Stopping scheduler...',
    '\n  请选择 (0-9, 或输入 g 进行AI自主交易): ': '\n  Select (0-9, or g for AI auto-trading): ',
    '\n  🎯 AI 自主交易模式': '\n  🎯 AI Auto-Trading Mode',
    '\n  💰 正在执行交易...': '\n  💰 Executing trades...',
    '\n  📈 正在启动 Streamlit 仪表盘...': '\n  📈 Starting Streamlit dashboard...',
    '\n  📊 数据源:': '\n  📊 Data sources:',
    '\n  📊 正在回测验证...': '\n  📊 Backtesting...',
    '\n  📊 正在查询持仓...': '\n  📊 Querying positions...',
    '\n  📋 当前配置:': '\n  📋 Current config:',
    '\n  🔌 启动模型同步 API...': '\n  🔌 Starting model sync API...',
    '\n  🔍 正在扫描市场...': '\n  🔍 Scanning market...',
    '\n  🚀 AI 正在为你自主决策，请耐心等待...': '\n  🚀 AI is deciding for you, please wait...',
    '\n  🚀 启动全自动循环...': '\n  🚀 Starting full-auto loop...',
    '\n  🚨 启动紧急止损流程...': '\n  🚨 Starting emergency stop...',
    '\n  🧠 LLM 智能对话:': '\n  🧠 LLM chat:',
    '\n  🧪 正在生成策略...': '\n  🧪 Generating strategy...',
    '\n═══════════════════════════════════════════════════════════\n     🤖 AI 全自动美股量化交易系统 v0.4.0                    \n     ─────────────────────────────────────────────          \n     📊 数据: AKShare + Finnhub                             \n     🧠 引擎: Microsoft Qlib + LightGBM                     \n     🤖 Agent: 调研->策略->回测->执行->监控                     \n     💰 交易: Alpaca Paper Trading                          \n     🛡️  风控: 7 条规则全检                                 \n     📈 仪表盘: Streamlit 可视化                             \n     ⏰ 调度器: 定时自动运行                                  \n     🧠 LLM: 智能对话 (多模型支持)                            \n═══════════════════════════════════════════════════════════\n    ': '\n═══════════════════════════════════════════════════════════\n     🤖 AI Fully-Automated US Stock Quant System v0.4.0    \n     ─────────────────────────────────────────────          \n     📊 Data: AKShare + Finnhub                             \n     🧠 Engine: Microsoft Qlib + LightGBM                   \n     🤖 Agents: Research -> Strategy -> Backtest -> Exec -> Monitor \n     💰 Trading: Alpaca Paper Trading                       \n     🛡️  Risk: 7-rule full checks                           \n     📈 Dashboard: Streamlit visualization                  \n     ⏰ Scheduler: automated runs                           \n     🧠 LLM: AI chat (multi-model support)                 \n═══════════════════════════════════════════════════════════\n    ',
    '\n💬 你: ': '\n💬 You: ',
    '\n📝 规则对话模式 (未配置 LLM API Key)': '\n📝 Rule-based mode (no LLM API key configured)',
    '     如需重新启动, 请先关闭旧的仪表盘进程': '     To restart, close the existing dashboard process first',
    '     请在 .env 中设置 REALTIME_ENABLED=true': 'Please .env Settings REALTIME_ENABLED=true',
    '     请在 .env 中配置 ALPACA_API_KEY 或 FINNHUB_API_KEY': 'Please .env Config ALPACA_API_KEY or FINNHUB_API_KEY',
    '     请确保 streamlit 已正确打包': '     Make sure streamlit is packaged correctly',
    '     请确保打包时已包含 src/ui/dashboard.py': '     Make sure src/ui/dashboard.py is included in the package',
    '     请运行: pip install fastapi uvicorn python-multipart': 'Please Run: pip install fastapi uvicorn python-multipart',
    '     降级提示: 系统可使用 REST 轮询模式获取行情': '     Fallback: the system can use REST polling for quotes',
    '   在 .env 中配置 LLM_{NAME}_* 环境变量可启用智能对话。': '   Set LLM_{NAME}_* env vars in .env to enable AI chat.',
    '   支持自然语言理解、多轮对话上下文记忆。': '   Supports NLU and multi-turn context memory.',
    '   请稍后重试，或输入 "帮助" 查看可用功能。': '   Please retry, or type "help" for available features.',
    '  0️⃣  退出': '  0️⃣  Exit',
    '  1️⃣  市场调研 - 看看今天有什么机会': "  1️⃣  Market research - see today's opportunities",
    '  2️⃣  策略生成 - AI 自动生成交易策略': '  2️⃣  Strategy generation - AI generates trading strategies',
    '  3️⃣  回测验证 - 测试策略历史表现': '  3️⃣  Backtest - test strategy on historical data',
    '  4️⃣  执行交易 - 模拟盘下单': '  4️⃣  Execute trades - place paper orders',
    '  5️⃣  持仓监控 - 查看持仓和风控': '  5️⃣  Position monitoring - view positions & risk',
    '  6️⃣  一键全自动 - 全流程自动运转': '  6️⃣  Full auto - end-to-end automation',
    '  7️⃣  紧急止损 - 清仓所有持仓': '  7️⃣  Emergency stop - liquidate all positions',
    '  8️⃣  初始化 Qlib 数据': '  8️⃣  Init Qlib data',
    '  9️⃣  启动仪表盘 - 可视化界面': '  9️⃣  Launch dashboard - visual UI',
    '  python main.py goal "5万本金，目标翻倍"': '  python main.py goal "50k capital, target 2x"',
    '  python main.py goal "我有10万美元，想赚20%"': '  python main.py goal "I have 100k USD, want 20% gain"',
    '  • 在上方 **🎯 四维评分详情** 中查看K线图、关注股票、创建交易策略\n  • 前往 **📈 策略管理** 查看和管理已创建的策略\n  • 或在对话中输入: "为 AAPL 生成交易策略"': '  • See K-line charts, watchlist and strategy creation in **🎯 Four-dimension Scoring** above\n  • Go to **📈 Strategies** to view and manage created strategies\n  • Or type in chat: "generate a strategy for AAPL"',
    '  • 在上方 **🎯 四维评分详情** 中查看K线图、关注股票、创建交易策略\n  • 前往 **📈 策略管理** 查看和管理已创建的策略\n  • 或在对话中输入: "为 AAPL 生成交易策略"\n  • ⚠️ 以上分析仅供参考，不构成投资建议': '  • See K-line charts, watchlist and strategy creation in **🎯 Four-dimension Scoring** above\n  • Go to **📈 Strategies** to view and manage created strategies\n  • Or type in chat: "generate a strategy for AAPL"\n  • ⚠️ For reference only, not investment advice',
    '  ⏹️ Dashboard 已停止': '⏹️ Dashboard Stop',
    '  ⏹️ 实时行情已停止。': '  ⏹️ Realtime quotes stopped。',
    '  ⏹️ 所有服务已停止。': '  ⏹️ All services stopped。',
    '  ⏹️ 智能助手聊天页面已停止': '  ⏹️ AI assistant chat page stopped',
    '  ⏹️ 调度器已停止': '  ⏹️ Scheduler stopped',
    '  ⏹️ 调度器已停止。': '  ⏹️ Scheduler stopped。',
    '  ⚠️ Alpaca API Key 未配置': '⚠️ Alpaca API Key not Config',
    '  ⚠️ Finnhub API Key 未配置': '⚠️ Finnhub API Key not Config',
    '  ⚠️ 当前 DEPLOY_MODE 不是 cloud, 建议在 .env 中设置 DEPLOY_MODE=cloud': '⚠️ Current DEPLOY_MODE notis cloud, Suggestions .env Settings DEPLOY_MODE=cloud',
    '  ⚠️ 无效选择，请重试': '  ⚠️ Invalid choice, please retry',
    '  ⚠️ 未输入股票代码': '  ⚠️ No symbols entered',
    '  ⚠️ 端口 8501 已被占用, 可能已有仪表盘在运行': '  ⚠️ Port 8501 in use, a dashboard may already be running',
    '  ⚠️ 端口 8501 已被占用, 跳过 Dashboard 启动 (可能已在运行)': '  ⚠️ Port 8501 in use, skipping dashboard launch (may already be running)',
    '  ⚠️ 端口 8502 已被占用, 跳过聊天服务启动 (可能已在运行)': '  ⚠️ Port 8502 in use, skipping chat service launch (may already be running)',
    '  ⚠️ 请先执行市场调研 (菜单 1)': '  ⚠️ Run market research first (menu 1)',
    '  ⚠️ 请先生成策略 (菜单 2)': '  ⚠️ Generate a strategy first (menu 2)',
    '  ⚠️ 这将清仓所有持仓！': '  ⚠️ This will liquidate all positions！',
    '  ✅ Dashboard 已启动 (线程模式)': '  ✅ Dashboard started (thread mode)',
    '  ✅ 一切正常，没有触发任何风控警报': '  ✅ All normal, no risk alerts triggered',
    '  ✅ 实时行情已启动! 等待数据流入...\n': '  ✅ Realtime quotes started! waiting for data...\n',
    '  ✅ 智能助手聊天服务已启动 (线程模式)': '  ✅ AI assistant chat service started (thread mode)',
    '  ✅ 调度器已启动': '  ✅ Scheduler started',
    '  ✅ 调度器已启动!': '  ✅ Scheduler started!',
    '  ❌ AKShare 数据获取失败': '  ❌ AKShare data fetch failed',
    '  ❌ Alpaca 连接失败': '  ❌ Alpaca connection failed',
    '  ❌ FastAPI/uvicorn 未安装': '  ❌ FastAPI/uvicorn not installed',
    '  ❌ 交易执行失败': '  ❌ Trade execution failed',
    '  ❌ 回测失败': '  ❌ Backtest failed',
    '  ❌ 实时行情已禁用 (REALTIME_ENABLED=false)': '❌ Realtime quotesDisable (REALTIME_ENABLED=false)',
    '  ❌ 无可用数据源!': '  ❌ No available data source!',
    '  ❌ 监控失败，请检查 Alpaca 配置': '  ❌ Monitoring failed, check Alpaca config',
    '  ❌ 策略生成失败': '  ❌ Strategy generation failed',
    '  ❌ 请提供投资目标，例如:': '  ❌ Please provide an investment goal, e.g.:',
    '  ❌ 调研失败': '  ❌ Research failed',
    '  例: "10万块，想赚5万"': '  e.g. "100k capital, want 50k profit"',
    '  例: "5万本金，目标翻倍，风险别太大"': '  e.g. "50k capital, target 2x, low risk"',
    '  例: "我有10万美元，想赚20%"': '  e.g. "I have 100k USD, want 20% gain"',
    '  健康检查完成。': '  Health check complete。',
    '  告诉我你的资金和目标，AI 全自动搞定:': '  Tell me your capital and goal, AI handles everything:',
    '  已取消。': '  Cancelled。',
    '  按 Ctrl+C 停止\n': '  Press Ctrl+C to stop\n',
    '  按 Ctrl+C 停止所有服务\n': '  Press Ctrl+C to stop all services\n',
    '  按 Ctrl+C 退出仪表盘\n': '  Press Ctrl+C to exit the dashboard\n',
    '  服务地址: http://localhost:8501': '  URL: http://localhost:8501',
    '  确认执行? (输入 yes 确认): ': 'ConfirmExecution? (Enter yes Confirm):',
    '  输入股票代码 (逗号分隔, 如 AAPL,MSFT,GOOGL): ': '  Enter symbols (comma-separated, e.g. AAPL,MSFT,GOOGL): ',
    '  这可能需要几分钟，请耐心等待...': '  This may take a few minutes, please wait...',
    '  这可能需要几分钟，请耐心等待。': '  This may take a few minutes, please wait.',
    '  🎯  AI 自主交易 - 告诉我目标和资金，全自动搞定': '  🎯  AI Auto-trading - tell me your goal & capital, fully automated',
    '  💬 你的目标: ': '  💬 Your goal: ',
    '  📋 功能菜单': '  📋 Menu',
    '  📋 已注册任务:': '  📋 Registered jobs:',
    '  🔍 快速健康检查...': '  🔍 Quick health check...',
    '  🔒 加密货币 24/7 调度已注册': '  🔒 Crypto 24/7 scheduler registered',
    '  🔟  设置关注股票': '  🔟  Set watchlist',
    '  🤖 智能助手聊天页面已停止': '  🤖 AI assistant chat page stopped',
    '#### ⚡ 快捷操作': '#### ⚡ QuickActions',
    '#### ⭐ 关注管理': '#### ⭐ WatchlistManagement',
    '#### 因子库说明': '#### FactorLibraryNotes',
    '#### 基本配置': '#### Basic Config',
    '#### 搜索配置': '#### SearchConfig',
    '#### 竞赛说明': '#### CompetitionNotes',
    '#### 竞赛配置': '#### CompetitionConfig',
    '#### 超参数配置': '#### Hyperparameters',
    '#### 🎛️ 控制面板': '#### 🎛️ Control Panel',
    '#### 🏢 公司简介': '#### 🏢 CompanyProfile',
    '#### 📈 创建交易策略': '#### 📈 CreateTradeStrategy',
    '#### 📊 技术指标': '#### 📊 Technical indicators',
    '#### 📋 关键财务指标': '#### 📋 KeyFinancialIndicators',
    '#### 🔍 环境检查': '#### 🔍 EnvironmentCheck',
    '**入选股票一览**': '**Selected Stocks**',
    '**模型文件路径:**': '**ModelFilePath:**',
    '**训练参数:**': '**TrainingParameters:**',
    '**🏆 设置最佳模型**': '**🏆 Set Best Model**',
    '**📋 点击查看个股详情:**': '**📋 ClickViewStockDetails:**',
    '**📰 相关新闻:**': '**📰 RelatedNews:**',
    '**🗑️ 删除模型文件**': '**🗑️ DeleteModelFile**',
    '**🚨 最近价格异动告警:**': '**🚨 RecentPriceMoversAlerts:**',
    '24h 涨跌幅 (%)': '24h Change % (%)',
    'AI自主交易': 'AIAuto-trading',
    'API 提供商指定的模型标识': 'API ProviderSpecifiedModelIdentifier',
    'ATM 隐含波动率': 'ATM Implied volatility',
    'Alpaca API 基础 URL': 'Alpaca API base URL',
    'Alpaca 交易 API Key': 'Alpaca Trade API Key',
    'Alpaca 交易 API Secret Key': 'Alpaca Trade API Secret Key',
    'CLOUD_API_PORT (云端 API 端口)': 'CLOUD_API_PORT (cloud API port)',
    'CLOUD_API_URL (云端 API 地址)': 'CLOUD_API_URL (cloud API URL)',
    'CRYPTO_DAILY_TRADE_LIMIT (日交易次数)': 'CRYPTO_DAILY_TRADE_LIMIT (daily trade limit)',
    'CRYPTO_ENABLED (启用加密货币模块)': 'CRYPTO_ENABLED (enable crypto module)',
    'CRYPTO_MAX_DAILY_LOSS (日最大亏损)': 'CRYPTO_MAX_DAILY_LOSS (max daily loss)',
    'CRYPTO_MAX_DRAWDOWN (最大回撤)': 'CRYPTO_MAX_DRAWDOWN (MaxDrawdown)',
    'CRYPTO_MAX_POSITION_PCT (单币最大仓位)': 'CRYPTO_MAX_POSITION_PCT (max per-coin position)',
    'CRYPTO_MAX_TOTAL_POSITIONS (最大持仓币种数)': 'CRYPTO_MAX_TOTAL_POSITIONS (max coins held)',
    'CRYPTO_SINGLE_STOP_LOSS (单币止损)': 'CRYPTO_SINGLE_STOP_LOSS (per-coin stop-loss)',
    'CRYPTO_TRADING_ENABLED (启用加密货币交易)': 'CRYPTO_TRADING_ENABLED (enable crypto trading)',
    'CRYPTO_TRADING_PAIRS (交易对, 逗号分隔)': 'CRYPTO_TRADING_PAIRS (pairs, comma-separated)',
    'FRED 宏观经济数据 API Key': 'FRED MacroEconomicData API Key',
    'Financial Modeling Prep 财务数据 API Key': 'Financial Modeling Prep FinancialData API Key',
    'Finnhub 行情数据 API Key': 'Finnhub MarketData API Key',
    'IC / Rank IC 对比': 'IC / Rank IC Comparison',
    'LLM 单次回复的最大长度': 'LLM Per-callResponseMaxLength',
    'LSTM (深度学习)': 'LSTM (Deep Learning)',
    'LightGBM (梯度提升树)': 'LightGBM (gradient boosting)',
    'Linear (线性回归)': 'Linear (LinearRegression)',
    'Linear 模型无需超参数配置': 'Linear model needs no hyperparameters',
    'MAX_CANDIDATES (候选股票上限)': 'MAX_CANDIDATES (candidate limit)',
    'MAX_DAILY_LOSS (日最大亏损)': 'MAX_DAILY_LOSS (max daily loss)',
    'MAX_DRAWDOWN (最大回撤)': 'MAX_DRAWDOWN (MaxDrawdown)',
    'MAX_POSITION_PCT (单票最大仓位)': 'MAX_POSITION_PCT (max per-stock position)',
    'MLflow 记录数': 'MLflow Records',
    'Max Tokens (最大 Token 数)': 'Max Tokens',
    'Monitor Agent 未就绪': 'Monitor Agent not ready',
    'OpenAI 兼容 API 的地址': 'OpenAI-compatible API URL',
    'PCR (成交量)': 'PCR (Volume)',
    'PCR (持仓量)': 'PCR (PositionsVolume)',
    'Paper Trading -> Live Trading 安全迁移 (6 项预检 + 渐进式仓位)': 'Paper -> Live safe migration (6 pre-checks + progressive sizing)',
    'Paper 最少天数': 'Minimum paper days',
    'Put-Call Ratio < 1 偏多, > 1 偏空': 'Put-Call Ratio < 1 bullish, > 1 bearish',
    'QLIB_ENABLED (启用 Qlib)': 'QLIB_ENABLED (Enable Qlib)',
    'Research Agent 未就绪': 'Research Agent not ready',
    'SCREEN_UNIVERSE_SIZE (预筛选数量)': 'SCREEN_UNIVERSE_SIZE (pre-screen size)',
    'TOP_K_SIGNALS (信号Top-K)': 'TOP_K_SIGNALS (SignalsTop-K)',
    'TORCH_ENABLED (启用 PyTorch)': 'TORCH_ENABLED (Enable PyTorch)',
    'Temperature (温度)': 'Temperature',
    '— (非必需)': '— (optional)',
    'ℹ️ 没有检测到配置变更': 'ℹ️ No config changes detected',
    '⏰ 定时任务调度器': '⏰ Scheduled jobsScheduler',
    '⏳ 正在加载实时行情与财务指标...': '⏳ Loading realtime quotes & fundamentals...',
    '⏹️ 停止实时行情': '⏹️ StopRealtime quotes',
    '⏹️ 停止调度器': '⏹️ StopScheduler',
    '⏹️ 实时行情已停止': '⏹️ Realtime quotes stopped',
    '▶️ 启动调度器': '▶️ StartScheduler',
    '▶️ 运行工作流': '▶️ Run workflow',
    '☀️ 下午好！市场正在交易中': '☀️ Good afternoon! Market is open',
    '☁️ 部署与系统配置': '☁️ Deployment & System Config',
    '⚔️ 多模型竞赛': '⚔️ Model competition',
    '⚔️ 开始竞赛': '⚔️ Start competition',
    '⚙️ 手动调整参数 (可选)': '⚙️ Adjust parameters manually (optional)',
    '⚙️ 模型操作': '⚙️ ModelActions',
    '⚠️ PyTorch 未安装，LSTM 将跳过。其他模型继续训练。': '⚠️ PyTorch not installed, LSTM skipped. Other models continue.',
    '⚠️ Qlib 未初始化，因子搜索可能无法正常工作。请确保已运行数据初始化。': '⚠️ Qlib not initialized, factor search may fail. Run data init first.',
    '⚠️ 不可用 (降级模式)': '⚠️ Unavailable (degraded mode)',
    '⚠️ 以上分析仅供参考，不构成投资建议': '⚠️ For reference only, not investment advice',
    '⚠️ 以下操作不可撤销，请谨慎执行!': '⚠️ These actions are irreversible, proceed with caution!',
    '⚠️ 加密货币 24/7 交易，订单使用 GTC (撤销前有效)': '⚠️ Crypto trades 24/7, orders use GTC (good-til-cancelled)',
    '⚠️ 加密货币模块未启用\n\n请在 `.env` 文件中设置 `CRYPTO_ENABLED=true` 并重启系统。': '⚠️ Crypto module disabled\n\nSet `CRYPTO_ENABLED=true` in `.env` and restart.',
    '⚠️ 无法检查 PyTorch，LSTM 可能失败': '⚠️ Cannot check PyTorch, LSTM may fail',
    '⚠️ 确认要迁移到实盘交易吗？此操作将切换到真实资金交易!': '⚠️ Migrate to live trading? This switches to REAL money!',
    '⚠️ 请先生成交易策略 (输入 "帮我生成一个策略")，然后才能回测。': '⚠️ Generate a strategy first (type "generate a strategy"), then backtest.',
    '⚠️ 请先生成交易策略，然后才能执行交易。': '⚠️ Generate a strategy first, then execute trades.',
    '⚠️ 迁移需通过 6 项预检': '⚠️ Migration requires 6 pre-checks',
    '⚠️ 部分预检未通过，请先解决以上问题': '⚠️ Some pre-checks failed, fix the issues above first',
    '⚠️ 风控告警!': '⚠️ Risk controlAlerts!',
    '⚪ 未启动': '⚪ not Start',
    '✅ 交易循环执行完成!': '✅ Trade loop completed!',
    '✅ 可用': '✅ Available',
    '✅ 多模型竞赛完成!': '✅ Model competition completed!',
    '✅ 已回滚到模拟盘模式': '✅ Rolled back to paper mode',
    '✅ 市场调研完成!': '✅ Market research completed!',
    '✅ 所有预检通过，可以安全迁移!': '✅ All pre-checks passed, safe to migrate!',
    '✅ 监控完成': '✅ Monitoring done',
    '✅ 紧急清仓完成': '✅ Emergency liquidation done',
    '✅ 组合已删除': '✅ PortfolioDelete',
    '✅ 迁移成功! 进入渐进式仓位阶段': '✅ Migration successful! Progressive sizing phase',
    '✅ 风控正常, 无告警': '✅ Risk normal, no alerts',
    '❌ PyTorch 未安装，无法训练 LSTM 模型。请运行: pip install torch': '❌ PyTorch not installed, cannot train LSTM. Run: pip install torch',
    '❌ 交易执行失败。': '❌ Trade execution failed。',
    '❌ 取消': '❌ Cancelled',
    '❌ 回测失败，可能是数据不足或策略配置有误。': '❌ Backtest failed: insufficient data or bad strategy config.',
    '❌ 无法检查 PyTorch 状态': '❌ Cannot check PyTorch status',
    '❌ 未安装': '❌ Not installed',
    '❌ 策略生成失败，请先进行市场调研或指定股票池。': '❌ Strategy generation failed: run research first or specify a stock pool.',
    '❌ 调研失败，请检查数据源配置。\n\n💡 可能的原因:\n  • Finnhub API Key 未配置\n  • 网络连接问题\n\n你可以在 .env 文件中配置 FINNHUB_API_KEY。': '❌ Research failed, check data source config.\n\n💡 Possible causes:\n  • Finnhub API key not configured\n  • Network issues\n\nSet FINNHUB_API_KEY in .env.',
    '➕ 创建新组合': '➕ New portfolio',
    '⬆️ 最高': '⬆️ High',
    '⬇️ 最低': '⬇️ Low',
    '⭐ 关注该股票': '⭐ Add to watchlist',
    '⭐ 我的关注股票调研': '⭐ Watchlist research',
    '⭐ 调研全部关注股票': '⭐ ResearchAllWatchlist',
    '为该股票创建策略': 'Create strategy for this stock',
    '买价': 'Bid',
    '交易': 'Trade',
    '交易执行员': 'Trade Executor',
    '交易执行失败': 'Trade execution failed',
    '交易日': 'Trading days',
    '从 MLflow (mlruns/) 和模型文件中提取历史训练记录': 'Extract training history from MLflow (mlruns/) and model files',
    '仓位': 'Position',
    '仓位系数': 'Position scale',
    '价格筛选后': 'After price filter',
    '使用 Alpha158 因子库，自动搜索有效因子并计算 IC/IR': 'Auto-search effective factors from Alpha158 and compute IC/IR',
    '使用策略模板 (自动填充因子和参数)': 'Use strategy template (auto-fill factors & params)',
    '使用默认参数': 'Use defaults',
    '例如: BTC/USD,ETH/USD,SOL/USD': 'e.g. BTC/USD,ETH/USD,SOL/USD',
    '例如: 帮我选5支10美元的潜力股': 'e.g. pick 5 potential stocks under $10',
    '信号Top-K': 'SignalsTop-K',
    '信号生成阶段选前K只做多': 'Long top-K stocks in signal generation',
    '信心一般': 'Moderate confidence',
    '信心很强': 'High confidence',
    '信心较弱': 'Low confidence',
    '修改后点击底部保存按钮，配置将写入 .env 文件': 'Click Save at the bottom to write config to .env',
    '候选股票上限': 'Candidate limit',
    '候选股票数': 'Candidates',
    '值越高回复越随机，值越低越确定保守': 'Higher = more random, lower = deterministic',
    '允许的最大回撤比例 (0.15 = 15%)': 'Max drawdown allowed (0.15 = 15%)',
    '入选数量': 'Selected count',
    '全周期K线图 · 技术指标 · 实时行情 · 基本面数据': 'All-period K-line charts · Technical indicators · Realtime quotes · FundamentalsData',
    '全自动循环': 'Full-auto loop',
    '关注主题 (可选)': 'Watch themes (optional)',
    '净值': 'Equity',
    '净值 ($)': 'Equity ($)',
    '准备开始竞赛...': 'Preparing competition...',
    '列采样': 'Colsample',
    '创建多策略组合，支持等权/风险平价/绩效加权分配': 'Create multi-strategy portfolios: equal-weight / risk-parity / performance-weighted',
    '创建组合': 'CreatePortfolio',
    '删除失败': 'Deletefailed',
    '到期日': 'Expiry',
    '加密货币持仓、风控状态、价格看板 (24/7 交易)': 'Crypto positions, risk status, price board (24/7)',
    '加密货币持仓分布': 'Crypto position allocation',
    '加密货币浮动盈亏 ($)': 'Crypto unrealized P&L ($)',
    '勾选后此模型将作为智能助手的默认 LLM': "Check to make this model the AI assistant's default LLM",
    '单只股票最大仓位占比 (0.30 = 30%)': 'Max position per stock (0.30 = 30%)',
    '单币仓位上限': 'Per-coin position limit',
    '单日最大亏损比例 (0.05 = 5%)': 'Max daily loss (0.05 = 5%)',
    '卖价': 'Ask',
    '历史订单、执行状态、风控检查记录': 'Order history、Execution status、Risk checksRecords',
    '叶子数': 'Leaves',
    '同时训练多个模型，对比 IC/Rank IC，自动选出最佳模型': 'Train multiple models, compare IC/Rank IC, auto-select the best',
    '告警数': 'Alerts',
    '员工数': 'Employees',
    '周报': 'Weekly',
    '回撤': 'Drawdown',
    '回测净值曲线': 'Backtest equity curve',
    '回测失败': 'Backtest failed',
    '回测验证': 'Backtest',
    '回测验证师': 'Backtest Validator',
    '回看窗口': 'Lookback window',
    '因子': 'Factor',
    '因子 IC 排名 (信息系数)': 'Factor IC ranking (information coefficient)',
    '因子 IR 排名 (信息比率 = IC均值/IC标准差)': 'Factor IR ranking (IR = IC mean / IC std)',
    '因子相关性阈值': 'Factor correlation threshold',
    '基于未平仓合约量的 PCR': 'PCR based on open interest',
    '处理请求': 'Processing request',
    '如: AAPL  或  AAPL,NVDA,TSLA  (逗号分隔多个)': 'e.g. AAPL or AAPL,NVDA,TSLA (comma-separated)',
    '如: AAPL, TSLA, NVDA': 'e.g. AAPL, TSLA, NVDA',
    '市值': 'Market Cap',
    '市场情绪': 'MarketSentiment',
    '市场调研': 'Market research',
    '布林带 BOLL': 'Bollinger Bands',
    '平值期权隐含波动率': 'ATM implied volatility',
    '平均综合分': 'Avg composite score',
    '年化收益': 'AnnualizedReturn',
    '开始日期': 'Start date',
    '开盘': 'Open',
    '当前价格': 'CurrentPrice',
    '总大小': 'Total size',
    '我的组合': 'My portfolios',
    '手动买入/卖出加密货币 (Paper Trading 模拟盘)': 'Manual crypto buy/sell (paper trading)',
    '手动选股': 'Manual selection',
    '执行': 'Execution',
    '执行: 调研→信号→风控→执行→监控': 'Run: research → signals → risk → execute → monitor',
    '执行交易': 'Execute trades',
    '扫描 data/models/ 目录，展示所有已保存的模型文件': 'Scan data/models/ and list saved model files',
    '扫描总数': 'Total scanned',
    '批次大小': 'Batch size',
    '报告': 'Report',
    '抱歉，我暂时没有生成有效回复，请换个说法试试。': 'Sorry, no valid response was generated. Try rephrasing.',
    '持仓市值': 'PositionsMarket Cap',
    '持仓数量': 'Position count',
    '持仓监控': 'Position monitoring',
    '收盘': 'Close',
    '文件大小 (KB)': 'FileSize (KB)',
    '新模型名称': 'New model name',
    '无可删除的模型文件': 'No model files to delete',
    '无可设置的模型文件': 'No model files to set',
    '无告警，组合运行正常': 'No alerts, portfolio running normally',
    '无法获取持仓': 'Failed to get positions',
    '无看涨期权数据': 'No call option data',
    '无看跌期权数据': 'No put option data',
    '无订阅股票 (无持仓且关注列表为空)': 'No subscribed symbols (no positions, empty watchlist)',
    '无需理解因子和模型，选择一个模板即可快速创建策略': 'No need to understand factors — pick a template to create a strategy fast',
    '日交易次数上限': 'Daily trade limit',
    '日报': 'Daily',
    '日收益率': 'Daily return',
    '日收益率分布': 'Daily return distribution',
    '智能选股': 'Smart Screening',
    '暂无交易相关通知': 'No trade notifications',
    '暂无可用模板': 'No templates available',
    '暂无通知记录': 'No notifications',
    '最低': 'Low',
    '最低价格 ($)': 'LowPrice ($)',
    '最佳 Rank IC': 'Best Rank IC',
    '最佳模型': 'Best Model',
    '最佳验证 IC': 'BestVerify IC',
    '最大持仓数': 'Max positions',
    '最大深度': 'MaxDeep',
    '最新价': 'LatestPrice',
    '最新调研报告、候选股票、新闻情绪、关注股票深度调研': 'Latest research reports、Candidate stocks、News sentiment、WatchlistDeepResearch',
    '最高': 'High',
    '最高价格 ($)': 'HighPrice ($)',
    '期权数据不可用': 'Options data unavailable',
    '期权链 · PCR比率 · 隐含波动率 · 到期日选择 (Yahoo Finance)': 'Options chain · PCR ratio · Implied volatility · Expiry selection (Yahoo Finance)',
    '未保存模型文件': 'not SaveModelFile',
    '未发现有效因子，尝试降低 IC 阈值后重新搜索': 'No effective factors found; lower the IC threshold and retry',
    '未回测': 'not Backtest',
    '未平仓': 'Open interest',
    '未执行': 'not Execution',
    '未找到符合条件的因子，尝试调整阈值': 'No factors matched; adjust thresholds',
    '未找到符合条件的股票': 'No stocks matched the criteria',
    '未迁移': 'Not migrated',
    '本地上传模型到云端时使用的 API 地址': 'API URL for uploading local models to cloud',
    '本地模型训练 · 多模型竞赛 · 因子搜索 · 训练历史管理': 'Local model training · Model competition · Factor search · Training historyManagement',
    '权重总和': 'Total weight',
    '标的现价': 'Underlying price',
    '模型 ID': 'Model ID',
    '模型名称': 'ModelName',
    '模型总数': 'Total models',
    '模型文件创建时间线': 'ModelFileCreateTimeLine',
    '模型标识名，用于区分不同模型配置': 'Model identifier to distinguish configs',
    '模型标识名，自定义命名': 'Model identifier, custom name',
    '模型类型占比': 'Model type share',
    '模型能力对比 (绝对值)': 'Model capability comparison (absolute)',
    '模板不存在': 'Template not found',
    '止损': 'Stop-loss',
    '止损线': 'Stop-lossLine',
    '止盈': 'Take-profit',
    '正在回滚...': 'Rolling back...',
    '正在执行 6 项预检...': 'Running 6 pre-checks...',
    '正在执行加密货币交易循环...': 'Running crypto trade loop...',
    '正在执行紧急清仓...': 'Running emergency liquidation...',
    '正在执行迁移...': 'Migrating...',
    '正在搜索有效因子... (可能需要 1-3 分钟)': 'Searching effective factors... (may take 1-3 min)',
    '正在检查持仓...': 'Checking positions...',
    '正在调研市场...': 'Researching market...',
    '步骤执行结果': 'Step results',
    '没有可训练的模型': 'No models to train',
    '波动率': 'Volatility',
    '浮动盈亏': 'Unrealized P&L',
    '涨跌幅 (%)': 'Change % (%)',
    '涨跌幅(%)': 'Change %(%)',
    '清仓': 'Liquidate',
    '用自然语言选股，AI 自动扫描全美股并多维度评分': 'Screen stocks in natural language; AI scans all US stocks with multi-dimension scoring',
    '留空表示不修改现有 Key': 'Leave blank to keep the current key',
    '留空表示不修改，显示为脱敏值。输入新值将覆盖原配置。': 'Leave blank to keep; shown masked. New values overwrite.',
    '看涨 IV': 'Call IV',
    '看涨期权 Top 20 成交量 & 未平仓 (按行权价)': 'Top 20 calls by volume & OI (by strike)',
    '看跌 IV': 'Put IV',
    '看跌期权 Top 20 成交量 & 未平仓 (按行权价)': 'Top 20 puts by volume & OI (by strike)',
    '确认': 'Confirm',
    '确认要清仓所有持仓吗？此操作不可撤销!': 'Liquidate all positions? This cannot be undone!',
    '竞赛完成!': 'Competition done!',
    '策略列表、回测结果、评级展示、模板创建、组合管理': 'Strategy list, backtests, ratings, templates, portfolios',
    '策略名称': 'StrategyName',
    '策略数': 'Strategies',
    '策略生成': 'Strategy generation',
    '策略生成失败': 'Strategy generation failed',
    '筛选条件': 'Filters',
    '索提诺': 'Sortino',
    '紧急止损': 'Emergency stop',
    '组合名称': 'PortfolioName',
    '结束日期': 'End date',
    '综合评分 (0-100, 越高越好)': 'Composite score (0-100, higher is better)',
    '缓存价格': 'Cached price',
    '网站': 'Website',
    '股息率': 'Dividend yield',
    '股票代码': 'Symbols',
    '股票池 (逗号分隔)': 'Stock pool (comma-separated)',
    '胜率': 'Win rate',
    '获取实时行情...': 'Fetching realtime quotes...',
    '获取最新的持仓和风控状态': 'Get latest positions and risk status',
    '行权价': 'Strike',
    '订单': 'Orders',
    '订单提交失败，请检查 API 配置': 'Order submission failed, check API config',
    '训练开始日期': 'Training start',
    '训练次数': 'Training runs',
    '训练结束日期': 'Training end',
    '训练轮数': 'Epochs',
    '训练集 IC': 'Train IC',
    '训练集 Rank IC': 'Train Rank IC',
    '设为最佳模型': 'Set as best model',
    '设为默认模型': 'Set as default model',
    '评级': 'Rating',
    '试试输入: "帮我选5支10美元的潜力股" 或 "选3只20-50美元的科技股"': 'Try: "pick 5 potential stocks under $10" or "3 tech stocks between $20-50"',
    '该策略暂无回测结果': 'No backtest results for this strategy',
    '请先生成交易策略，然后才能回测': 'Generate a strategy first, then backtest',
    '请先生成交易策略，然后才能执行交易': 'Generate a strategy first, then execute trades',
    '请至少选择一个模型': 'Select at least one model',
    '请输入股票代码': 'Please EnterSymbols',
    '请输入股票池': 'Enter a stock pool',
    '请输入至少一只股票': 'Enter at least one symbol',
    '请选择交易对并输入有效数量': 'Select a pair and enter a valid quantity',
    '请选择或输入股票代码': 'Please SelectorEnterSymbols',
    '调度器已停止': 'Scheduler stopped',
    '调度器已启动': 'Scheduler started',
    '调度器未启动': 'Schedulernot Start',
    '调研失败，请检查数据源配置': 'Research failed, check data source config',
    '调研阶段最多发现多少只候选股票': 'Max candidate stocks found in research',
    '账户资产、持仓分布、今日盈亏、告警状态': 'Account & Assets、Position allocation、Today P&L、Alert status',
    '输入 API Key': 'Enter API Key',
    '输入参数 (可选)': 'EnterParameters (Optional)',
    '输入美股代码，如 AAPL, TSLA, NVDA': 'Enter US symbols, e.g. AAPL, TSLA, NVDA',
    '输入股票代码': 'EnterSymbols',
    '输入行业/主题关键词，调研时优先匹配相关股票': 'Enter sector/theme keywords to prioritize in research',
    '输入选股需求': 'Enter screening request',
    '迁移阶段': 'Migration phase',
    '返回 Top-N 因子': 'Return top-N factors',
    '选择模型类型、配置参数，一键启动本地训练': 'Pick model type, configure params, one-click local training',
    '选择模板': 'Select template',
    '选择策略': 'SelectStrategy',
    '选择股票查看四维评分': 'Select a stock to view four-dimension scores',
    '选股失败，请检查数据源配置': 'Screening failed, check data source config',
    '选股数量': 'Stock count',
    '配置一个 OpenAI 兼容 API。支持 DeepSeek、豆包、通义千问、OpenAI 等。': 'Configure an OpenAI-compatible API. Supports DeepSeek, Doubao, Qwen, OpenAI, etc.',
    '配置查看、工作流手动触发、紧急止损、实盘迁移': 'Config viewer、Manual workflow trigger、Emergency stop、Live migration',
    '隐含波动率 (%)': 'Implied volatility (%)',
    '隐含波动率(%)': 'Implied volatility(%)',
    '隐含波动率数据不完整': 'Implied volatilityDatanotFull',
    '隐藏层维度': 'Hidden dim',
    '预筛选进入详细分析的最大数量': 'Max pre-screened stocks entering detailed analysis',
    '频次': 'Frequency',
    '验证集 IC': 'Valid IC',
    '验证集 IC 历史趋势': 'Valid IC history',
    '验证集 Rank IC': 'Valid Rank IC',
    '🌅 早上好！新的一天开始了': '🌅 Good morning! A new day begins',
    '🌆 晚上好！来复盘一下今天的交易吧': "🌆 Good evening! Time to review today's trades",
    '🌐 全市场调研': '🌐 Full-market research',
    '🌙 夜深了，还在研究市场吗？': "🌙 It's late — still researching the market?",
    '🎯 候选股票': '🎯 Candidate stocks',
    '🎯 四维评分详情': '🎯 Four-dimension Scoring',
    '🎯 模型能力雷达图': '🎯 Model capability radar',
    '🎯 选股配置': '🎯 ScreeningConfig',
    '🎯 选股配置（运行时修改）': '🎯 Screening config (runtime)',
    '🏆 将最佳模型设为全局最佳': '🏆 Set best model as global best',
    '🏆 将此模型设为最佳模型': '🏆 Set this model as best',
    '🏆 当前最佳模型类型': '🏆 Current Best Model Type',
    '🏆 最佳模型': '🏆 Best Model',
    '🏢 公司详情': '🏢 CompanyDetails',
    '🏦 市值': '🏦 Market Cap',
    '👉 前往「📈 策略管理」查看详情和回测': '👉 Go to "📈 Strategies" for details and backtests',
    '👋 再见！祝投资顺利！': '👋 Bye! Happy investing！',
    '💔 取消关注': '💔 Remove from watchlist',
    '💡 Alpaca 未配置, 无法获取账户数据': '💡 Alpaca not configured, cannot fetch account data',
    '💡 Alpaca 未配置或无法连接，显示模拟数据': '💡 Alpaca not configured or unreachable, showing mock data',
    '💡 Alpaca 未配置，无法进行交易': '💡 Alpaca not configured, trading unavailable',
    '💡 交易在 Alpaca Paper Trading 模拟盘执行，不涉及真实资金': '💡 Trades run on Alpaca paper trading, no real money',
    '💡 后续操作': '💡 Next steps',
    '💡 尚未设置最佳模型。训练完成后可标记最佳模型。': '💡 No best model set yet. Mark one after training.',
    '💡 提示: 日内分时数据需要安装 yfinance (`pip install yfinance`)': '💡 Tip: intraday data requires yfinance (`pip install yfinance`)',
    '💡 无法获取账户信息': '💡 Cannot fetch account info',
    '💡 暂无可用价格数据，请检查 Alpaca API 配置': '💡 No price data available, check Alpaca API config',
    '💡 暂无实时价格数据。点击「启动实时行情」开始接收盘中数据。': '💡 No realtime prices yet. Click "Start realtime quotes".',
    '💡 点击上方按钮执行市场调研、单股调研或关注股票调研': '💡 Click the buttons above for market / single-stock / watchlist research',
    '💡 输入选股需求或调整参数后，点击「开始选股」': '💡 Enter a request or adjust params, then click "Start screening"',
    '💡 部分配置可能需要重启服务后生效': '💡 Some config changes require a service restart',
    '💪 购买力': '💪 Buying power',
    '💪 贴买力': '💪 Buying power',
    '💬 自然语言选股': '💬 NL stock screening',
    '💰 最新价': '💰 LatestPrice',
    '💰 账户概览': '💰 Account Overview',
    '💲 价格看板': '💲 Price Board',
    '💵 现金': '💵 Cash',
    '💼 总资产': '💼 Total Assets',
    '💾 保存选股配置': '💾 SaveScreeningConfig',
    '💾 保存配置到 .env': '💾 Save config to .env',
    '📂 路径信息（只读）': '📂 Path info (read-only)',
    '📅 IPO日期': '📅 IPODate',
    '📅 模型文件时间线': '📅 ModelFileTimeLine',
    '📅 结束': '📅 End',
    '📅 起始': '📅 Start',
    '📈 IC 对比图': '📈 IC ComparisonChart',
    '📈 为该股票生成交易策略': '📈 Generate strategy for this stock',
    '📈 买入': '📈 Buy',
    '📈 净值曲线': '📈 Equity curve',
    '📈 加密货币持仓': '📈 Crypto positions',
    '📈 区间收益': '📈 Period return',
    '📈 因子 IC 排名': '📈 Factor IC ranking',
    '📈 因子 IR 排名': '📈 Factor IR ranking',
    '📈 成交量': '📈 Volume',
    '📈 持仓分布': '📈 Position allocation',
    '📈 验证集 IC 趋势': '📈 Valid IC trend',
    '📉 卖出': '📉 Sell',
    '📊 K线条数': '📊 Candles',
    '📊 交易与风控配置': '📊 Trade&Risk controlConfig',
    '📊 交易对数': '📊 Pairs',
    '📊 刷新加密货币快照': '📊 Refresh crypto snapshot',
    '📊 各币种浮动盈亏': '📊 P&L by coin',
    '📊 各持仓浮动盈亏': '📊 P&L by position',
    '📊 回测绩效': '📊 Backtest performance',
    '📊 因子搜索结果': '📊 Factor searchResults',
    '📊 多策略组合管理': '📊 Portfolio management',
    '📊 日收益率分布': '📊 Daily return distribution',
    '📊 模型类型分布': '📊 ModelTypeAllocation',
    '📊 流通股': '📊 Float',
    '📊 涨跌幅': '📊 Change %',
    '📊 竞赛结果对比': '📊 CompetitionResultsComparison',
    '📊 系统状态': '📊 SystemStatus',
    '📊 综合评分对比': '📊 Score comparison',
    '📊 训练结果': '📊 TrainingResults',
    '📊 运行监控快照': '📊 Runtime snapshot',
    '📋 MLflow 训练记录': '📋 MLflow TrainingRecords',
    '📋 交易对列表': '📋 Pair list',
    '📋 交易模式': '📋 Trading mode',
    '📋 从模板创建策略': '📋 Create strategy from template',
    '📋 入选股票详情': '📋 Selected stock details',
    '📋 回滚到模拟盘': '📋 Roll back to paper',
    '📋 当前持仓': '📋 CurrentPositions',
    '📋 执行预检': '📋 Run pre-checks',
    '📋 昨收': '📋 Prev Close',
    '📋 期权概况': '📋 Options overview',
    '📋 系统配置': '📋 System config',
    '📖 期权数据解读指南': '📖 Options reading guide',
    '📖 查看公司简介详情': '📖 ViewCompanyProfileDetails',
    '📖 查看完整公司简介': '📖 ViewFullCompanyProfile',
    '📜 交易通知历史': '📜 Trade notification history',
    '📜 训练历史': '📜 Training history',
    '📝 未配置 LLM 模型，智能助手使用规则引擎模式。配置后可启用 AI 智能对话。': '📝 No LLM configured; the assistant uses rule-based mode. Configure one to enable AI chat.',
    '📝 调研总结': '📝 Research summary',
    '📝 选股总结': '📝 Screening summary',
    '📦 已训练模型': '📦 TrainingModel',
    '📭 当前无加密货币持仓\n\n💡 你可以在下方「🛒 手动交易」面板中买入加密货币，或运行加密货币交易循环自动建仓。': '📭 No crypto positions\n\n💡 Buy crypto in the "🛒 Manual Trade" panel below, or run the crypto trade loop.',
    '📭 当前无持仓': '📭 No positions',
    '📭 当前无持仓，运行工作流后将显示持仓信息': '📭 No positions; run a workflow to see positions',
    '📭 暂无关注股票调研结果': '📭 No watchlist research results',
    '📭 暂无模型文件。前往「训练新模型」标签页开始训练。': '📭 No model files. Go to the "New Model" tab to train.',
    '📭 暂无相关新闻': '📭 No related news',
    '📭 暂无策略，请从上方模板创建，或运行工作流自动生成': '📭 No strategies; create from templates above or run a workflow',
    '📭 暂无组合，请在上方创建': '📭 No portfolios; create one above',
    '📭 暂无该公司简介信息': '📭 No company profile',
    '📭 未找到 MLflow 训练记录 (mlruns/ 目录为空)': '📭 No MLflow records (mlruns/ is empty)',
    '📰 近期关键事件': '📰 Recent key events',
    '🔄 刷新数据': '🔄 Refresh data',
    '🔄 可用工作流:': '🔄 Available workflows:',
    '🔄 工作流管理': '🔄 Workflow management',
    '🔄 执行市场调研': '🔄 ExecutionMarket research',
    '🔄 数据源': '🔄 Data sources',
    '🔄 运行加密货币交易循环': '🔄 Run crypto trade loop',
    '🔄 重新分配权重': '🔄 Rebalance weights',
    '🔍 开始因子搜索': '🔍 Start factor search',
    '🔍 开始选股': '🔍 Start screening',
    '🔍 正在扫描全美股并评分... (可能需要 30-60 秒)': '🔍 Scanning all US stocks and scoring... (may take 30-60s)',
    '🔍 策略详情': '🔍 StrategyDetails',
    '🔍 调研指定股票': '🔍 ResearchSpecifiedStock',
    '🔍 输入股票代码': '🔍 EnterSymbols',
    '🔍 迁移就绪检查': '🔍 Migration readiness check',
    '🔍 选择标的': '🔍 Select symbol',
    '🔎 单股深度调研': '🔎 Single-stock deep research',
    '🔑 API 密钥': '🔑 API Key',
    '🔔 最近通知': '🔔 Recent notifications',
    '🔬 因子搜索': '🔬 Factor search',
    '🔴 实盘迁移管理': '🔴 Live migrationManagement',
    '🔴 悲观': '🔴 Bearish',
    '🔴 极度看空': '🔴 ExtremeBearish',
    '🔴 迁移到实盘': '🔴 Migrate to live',
    '🔴亏': '🔴Loss',
    '🔵 开盘': '🔵 Open',
    '🕐 时间': '🕐 Time',
    '🗑️ 删除组合': '🗑️ DeletePortfolio',
    '😊 隐含波动率微笑曲线': '😊 IV smile curve',
    '🚀 创建策略': '🚀 CreateStrategy',
    '🚀 启动全自动循环...': '🚀 Starting full-auto loop...',
    '🚀 开始训练': '🚀 Start training',
    '🚀 快速操作': '🚀 QuickActions',
    '🚀 训练新模型': '🚀 Train new model',
    '🚨 启动紧急止损流程...': '🚨 Starting emergency stop...',
    '🚨 紧急操作': '🚨 Emergency actions',
    '🚨 紧急清仓': '🚨 Emergency liquidation',
    '🟢 运行中': '🟢 Run',
    '🟢赚': '🟢Gain',
    '🤖 Agent 状态:': '🤖 Agent Status:',

    # ── f-string 模板翻译 (i18n_fstring_convert 生成) ──
    '\n\n⚠️ 链中断: {v1}': '\n\n⚠️ Chain interrupted: {v1}',
    '\n    <div class="info-card">\n        <div class="info-card-title">🎯 选股</div>\n        <div class="info-row">\n            <span>候选上限</span>\n            <span class="value">{v1}</span>\n        </div>\n        <div class="info-row">\n            <span>信号 Top-K</span>\n            <span class="value">{v2}</span>\n        </div>\n    </div>\n    ': '\n    <div class="info-card">\n        <div class="info-card-title">🎯 Screening</div>\n        <div class="info-row">\n            <span>Candidate limit</span>\n            <span class="value">{v1}</span>\n        </div>\n        <div class="info-row">\n            <span>Signal Top-K</span>\n            <span class="value">{v2}</span>\n        </div>\n    </div>\n    ',
    '\n    <div class="info-card">\n        <div class="info-card-title">📡 数据源</div>\n        <div class="info-row">\n            <span><span class="source-dot {v1}"></span>Alpaca</span>\n            <span class="value {v2}">{v3}</span>\n        </div>\n        <div class="info-row">\n            <span><span class="source-dot {v4}"></span>Finnhub</span>\n            <span class="value {v5}">{v6}</span>\n        </div>\n    </div>\n    ': '\n    <div class="info-card">\n        <div class="info-card-title">📡 Data Sources</div>\n        <div class="info-row">\n            <span><span class="source-dot {v1}"></span>Alpaca</span>\n            <span class="value {v2}">{v3}</span>\n        </div>\n        <div class="info-row">\n            <span><span class="source-dot {v4}"></span>Finnhub</span>\n            <span class="value {v5}">{v6}</span>\n        </div>\n    </div>\n    ',
    '\n    <div class="info-card">\n        <div class="info-card-title">🛡️ 风控</div>\n        <div class="info-row">\n            <span>单票仓位</span>\n            <span class="value value-blue">{v1:.0%}</span>\n        </div>\n        <div class="info-row">\n            <span>日最大亏损</span>\n            <span class="value value-red">{v2:.0%}</span>\n        </div>\n        <div class="info-row">\n            <span>最大回撤</span>\n            <span class="value value-red">{v3:.0%}</span>\n        </div>\n    </div>\n    ': '\n    <div class="info-card">\n        <div class="info-card-title">🛡️ Risk Control</div>\n        <div class="info-row">\n            <span>Max position</span>\n            <span class="value value-blue">{v1:.0%}</span>\n        </div>\n        <div class="info-row">\n            <span>Max daily loss</span>\n            <span class="value value-red">{v2:.0%}</span>\n        </div>\n        <div class="info-row">\n            <span>Max drawdown</span>\n            <span class="value value-red">{v3:.0%}</span>\n        </div>\n    </div>\n    ',
    '\n   理由: {v1}': '\n   Reason: {v1}',
    '\n  ── 状态 [{v1}] ──': '\n  ── Status [{v1}] ──',
    '\n  📦 正在初始化 Qlib 数据: {v1}': '\n  📦 Initializing Qlib data: {v1}',
    '\n❌ AI 自主交易失败: {v1}': '\n❌ AI auto-trading failed: {v1}',
    '\n❌ 全自动循环失败: {v1}': '\n❌ Full-auto loop failed: {v1}',
    '\n❌ 紧急止损失败: {v1}': '\n❌ Emergency stop failed: {v1}',
    '\n全自动循环完成: {v1}': '\nFull-auto loop completed: {v1}',
    '\n紧急止损完成: {v1}': '\nEmergency stop completed: {v1}',
    '\n📋 {v1}报告:': '\n📋 {v1} report:',
    '\n🤖 抱歉，处理你的请求时出现了错误: {v1}': '\n🤖 Sorry, an error occurred: {v1}',
    '\n🧠 智能对话模式已启用 (LLM: {v1})': '\n🧠 AI chat mode enabled (LLM: {v1})',
    '     LLM: 📝 规则模式': '     LLM: 📝 Rule mode',
    '     LLM: 📝 规则模式 (已配置 {v1} 个模型但Key未填)': '     LLM: 📝 Rule mode ({v1} models configured, keys missing)',
    '     LLM: 📝 规则模式 (未配置)': '     LLM: 📝 Rule mode (not configured)',
    '     LLM: 🧠 智能模式 (模型: {v1})': '     LLM: 🧠 AI mode (model: {v1})',
    '     交易模式: {v1}': '     Trading mode: {v1}',
    '     最大仓位: {v1:.0%} | 日最大亏损: {v2:.0%} | 最大回撤: {v3:.0%}': '     Max position: {v1:.0%} | Max daily loss: {v2:.0%} | Max drawdown: {v3:.0%}',
    '     模型列表: {v1}': '     Models: {v1}',
    '     请在 .env 中填入对应的 API Key': '     Fill in the API keys in .env',
    '  LLM: 📝 规则模式 (已配置但Key未填)': '  LLM: 📝 Rule mode (configured, key missing)',
    '  LLM: 📝 规则模式 (未配置)': '  LLM: 📝 Rule mode (not configured)',
    '  LLM: 🧠 智能模式 (模型: {v1})': '  LLM: 🧠 AI mode (model: {v1})',
    '  Qlib数据: {v1}': '  Qlib data: {v1}',
    '  {v1}: {v2:.2f}股 | {v3} ${v4:+,.2f} ({v5:+.1%}) | 占仓位 {v6:.1%}': '  {v1}: {v2:.2f} sh | {v3} ${v4:+,.2f} ({v5:+.1%}) | {v6:.1%} of portfolio',
    '  ℹ️ 未配置 LLM (使用规则引擎模式)': '  ℹ️ No LLM configured (rule-engine mode)',
    '  ⏰ 调度器: 自动注册默认任务': '  ⏰ Scheduler: default jobs auto-registered',
    '  ⚠️ Alpaca 异常: {v1}': '  ⚠️ Alpaca error: {v1}',
    '  ⚠️ Dashboard 启动失败: {v1}': '  ⚠️ Dashboard failed to start: {v1}',
    '  ⚠️ Dashboard 异常: {v1}': '  ⚠️ Dashboard error: {v1}',
    '  ⚠️ Finnhub API 异常: {v1}': '  ⚠️ Finnhub API error: {v1}',
    '  ⚠️ LLM 检测异常: {v1}': '  ⚠️ LLM check error: {v1}',
    '  ⚠️ Qlib 数据未初始化': '  ⚠️ Qlib data not initialized',
    '  ⚠️ 已配置 {v1} 个模型但 Key 未填写': '  ⚠️ {v1} models configured but keys missing',
    '  ⚠️ 智能助手聊天服务启动失败: {v1}': '  ⚠️ AI chat service failed to start: {v1}',
    '  ⚠️ 有 {v1} 条警报:': '  ⚠️ {v1} alerts:',
    '  ⚠️ 聊天服务异常: {v1}': '  ⚠️ Chat service error: {v1}',
    '  ⚠️ 调度器启动失败: {v1}': '  ⚠️ Scheduler failed to start: {v1}',
    '  ✅ AKShare 数据获取正常 (AAPL: {v1} 条)': '  ✅ AKShare data OK (AAPL: {v1} rows)',
    '  ✅ Alpaca 连接正常 (Paper: ${v1:,.2f})': '  ✅ Alpaca connected (Paper: ${v1:,.2f})',
    '  ✅ Dashboard 已启动 (PID: {v1})': '  ✅ Dashboard started (PID: {v1})',
    '  ✅ Finnhub API 正常': '  ✅ Finnhub API OK',
    '  ✅ LLM 已启用 (默认模型: {v1})': '  ✅ LLM enabled (default: {v1})',
    '  ✅ Qlib 数据目录: {v1}': '  ✅ Qlib data dir: {v1}',
    '  ✅ {v1}: 已完成': '  ✅ {v1}: done',
    '  ✅ 成功: {v1}': '  ✅ Success: {v1}',
    '  ✅ 智能助手聊天服务已启动 (PID: {v1})': '  ✅ AI chat service started (PID: {v1})',
    '  ❌ AI 自主交易出错: {v1}': '  ❌ AI auto-trading error: {v1}',
    '  ❌ AKShare 异常: {v1}': '  ❌ AKShare error: {v1}',
    '  ❌ API 启动失败: {v1}': '  ❌ API failed to start: {v1}',
    '  ❌ Streamlit 启动失败: {v1}': '  ❌ Streamlit failed to start: {v1}',
    '  ❌ {v1}: 失败或跳过': '  ❌ {v1}: failed or skipped',
    '  ❌ 交易执行过程出错: {v1}': '  ❌ Trade execution error: {v1}',
    '  ❌ 仪表盘启动失败: {v1}': '  ❌ Dashboard failed to start: {v1}',
    '  ❌ 全自动循环出错: {v1}': '  ❌ Full-auto loop error: {v1}',
    '  ❌ 回测过程出错: {v1}': '  ❌ Backtest error: {v1}',
    '  ❌ 找不到仪表盘文件: {v1}': '  ❌ Dashboard file not found: {v1}',
    '  ❌ 未知配置项: {v1}': '  ❌ Unknown config item: {v1}',
    '  ❌ 监控过程出错: {v1}': '  ❌ Monitoring error: {v1}',
    '  ❌ 策略生成过程出错: {v1}': '  ❌ Strategy generation error: {v1}',
    '  ❌ 紧急止损失败: {v1}': '  ❌ Emergency stop failed: {v1}',
    '  ❌ 调度器启动失败: {v1}': '  ❌ Scheduler failed to start: {v1}',
    '  ❌ 调研过程出错: {v1}': '  ❌ Research error: {v1}',
    '  交易模式: {v1}': '  Trading mode: {v1}',
    '  今日盈亏: {v1} ${v2:+,.2f} ({v3:+.1%})': '  Today P&L: {v1} ${v2:+,.2f} ({v3:+.1%})',
    '  可用现金: ${v1:,.2f}': '  Cash: ${v1:,.2f}',
    '  失败: {v1}': '  Failed: {v1}',
    '  当前回撤 {v1:.1%}，已超过上限，建议暂停！': '  Drawdown {v1:.1%} exceeds the limit — consider pausing!',
    '  当前回撤 {v1:.1%}，很安全': '  Drawdown {v1:.1%}, very safe',
    '  当前回撤 {v1:.1%}，接近警戒线！': '  Drawdown {v1:.1%}, near the warning line!',
    '  当前回撤 {v1:.1%}，需要注意': '  Drawdown {v1:.1%}, keep an eye on it',
    '  总资产: ${v1:,.2f}': '  Total assets: ${v1:,.2f}',
    '  日最大亏损: {v1:.0%}': '  Max daily loss: {v1:.0%}',
    '  最大仓位: {v1:.0%}': '  Max position: {v1:.0%}',
    '  最大回撤: {v1:.0%}': '  Max drawdown: {v1:.0%}',
    '  步骤: {v1}': '  Step: {v1}',
    '  缓存价格: {v1} 只 | 告警: {v2} 条': '  Cached prices: {v1} | Alerts: {v2}',
    '  购买力: ${v1:,.2f}': '  Buying power: ${v1:,.2f}',
    '  📊 K线完成: {v1} | O:{v2:.2f} H:{v3:.2f} L:{v4:.2f} C:{v5:.2f} V:{v6}': '  📊 Candles done: {v1} | O:{v2:.2f} H:{v3:.2f} L:{v4:.2f} C:{v5:.2f} V:{v6}',
    '  📋 API 文档: http://{v1}:{v2}/docs': '  📋 API docs: http://{v1}:{v2}/docs',
    '  📋 全自动循环完成!': '  📋 Full-auto loop completed!',
    '  📋 系统配置:': '  📋 System config:',
    '  📋 紧急止损完成!': '  📋 Emergency stop completed!',
    '  📋 订阅股票: {v1}': '  📋 Subscribed symbols: {v1}',
    '  📡 API 端口: {v1}': '  📡 API port: {v1}',
    '  📡 数据源: Alpaca {v1} | Finnhub {v2}': '  📡 Data sources: Alpaca {v1} | Finnhub {v2}',
    '  📡 监听: http://{v1}:{v2}': '  📡 Listening: http://{v1}:{v2}',
    '  🔧 数据源模式: {v1}': '  🔧 Data source mode: {v1}',
    '  🚨 价格异动告警阈值: {v1:.1%}': '  🚨 Price-move alert threshold: {v1:.1%}',
    '  🤖 智能助手聊天服务已启动 (PID: {v1}, 端口 8502)': '  🤖 AI chat service started (PID: {v1}, port 8502)',
    '  🤖 智能助手聊天服务已启动 (线程模式, 端口 8502)': '  🤖 AI chat service started (thread mode, port 8502)',
    ' ...等 {v1} 只': ' ...and {v1} more',
    '**步骤 {v1}/{v2}: {v3}**\n\n': '**Step {v1}/{v2}: {v3}**\n\n',
    'AI 解析失败 ({v1})，使用手动参数': 'AI parse failed ({v1}), using manual params',
    'AKShare 数据获取失败: {v1}': 'AKShare data fetch failed: {v1}',
    'Function Agent 执行异常: {v1}': 'Function Agent error: {v1}',
    'K线图加载失败: {v1}': 'K-line chart load failed: {v1}',
    'LLM 配置加载失败: {v1}': 'LLM config load failed: {v1}',
    'LLM 配置重载失败: {v1}': 'LLM config reload failed: {v1}',
    'VIX 数据获取失败: {v1}': 'VIX data fetch failed: {v1}',
    'explain_term 期望 str 或 None 类型，得到 {v1}': 'explain_term expects str or None, got {v1}',
    'yfinance 数据获取失败: {v1}': 'yfinance data fetch failed: {v1}',
    '{v1} ({v2} 个指标)': '{v1} ({v2} indicators)',
    '{v1} - {v2} | {v3} 情绪: {v4} ({v5:+.2f})': '{v1} - {v2} | {v3} sentiment: {v4} ({v5:+.2f})',
    '{v1} 只': '{v1}',
    '{v1} 四维评分雷达图': '{v1} four-dimension score radar',
    '{v1} 天': '{v1} days',
    '{v1} 条': '{v1}',
    '{v1} 策略权重分布': '{v1} strategy weight allocation',
    '{v1} 训练失败: {v2}': '{v1} training failed: {v2}',
    '{v1} 隐含波动率曲线 - {v2}': '{v1} IV curve - {v2}',
    '{v1}, 选 {v2} 只': '{v1}, pick {v2}',
    '{v1}执行完成。': '{v1} completed.',
    '⚠️ {v1} 训练完成，但可能未产生有效结果 (数据不足?)': '⚠️ {v1} training finished, but may have no valid results (insufficient data?)',
    '⚠️ 处理请求时出现异常: {v1}\n\n请稍后重试。': '⚠️ Request error: {v1}\n\nPlease retry later.',
    '⚠️ 已配置 {v1} 个模型但无可用项 (Key 可能未填)。 模型: {v2}': '⚠️ {v1} models configured but none available (keys missing?). Models: {v2}',
    '⚠️ 执行完成 (有异常): {v1}': '⚠️ Completed (with errors): {v1}',
    '⚠️ 账户数据获取失败: {v1}': '⚠️ Account data fetch failed: {v1}',
    '⚠️ 预估金额 ${v1:.2f} < $10, Alpaca 最小订单金额为 $10, 请增加数量': '⚠️ Estimated ${v1:.2f} < $10 (Alpaca minimum), increase quantity',
    '⚡ [{v1}/{v2}] {v3}中，请稍候...': '⚡ [{v1}/{v2}] {v3}, please wait...',
    '⚡ 实时行情已启动! 订阅 {v1} 只股票': '⚡ Realtime quotes started! Subscribed to {v1} symbols',
    '✅ LLM 已启用 | 默认模型: {v1} | 可用: {v2}': '✅ LLM enabled | default: {v1} | available: {v2}',
    '✅ {v1} 个交易对可获取价格': '✅ {v1} pairs can fetch prices',
    '✅ {v1} 只关注股票调研完成!': '✅ {v1} watchlist symbols researched!',
    '✅ {v1} 只股票调研完成!': '✅ {v1} symbols researched!',
    '✅ {v1} 训练完成!': '✅ {v1} training completed!',
    '✅ {v1}完成!': '✅ {v1} completed!',
    '✅ 买入订单已提交! ID: {v1}': '✅ Buy order submitted! ID: {v1}',
    '✅ 全自动循环完成!': '✅ Full-auto loop completed!',
    '✅ 卖出订单已提交! ID: {v1}': '✅ Sell order submitted! ID: {v1}',
    '✅ 工作流完成: {v1}': '✅ Workflow completed: {v1}',
    '✅ 已删除模型文件: {v1}': '✅ Model file deleted: {v1}',
    "✅ 已将 '{v1}' 设为最佳模型!": "✅ '{v1}' set as best model!",
    '✅ 已更新: 候选上限={v1}, Top-K={v2}': '✅ Updated: candidate limit={v1}, Top-K={v2}',
    '✅ 找到 {v1} 个有效因子!': '✅ Found {v1} effective factors!',
    '✅ 权重已更新: {v1}': '✅ Weights updated: {v1}',
    "✅ 策略 '{v1}' 创建成功! (ID: {v2})": "✅ Strategy '{v1}' created! (ID: {v2})",
    '✅ 紧急止损完成!': '✅ Emergency stop completed!',
    "✅ 组合 '{v1}' 创建成功! (ID: {v2})": "✅ Portfolio '{v1}' created! (ID: {v2})",
    '❌ AI 自主交易执行失败: {v1}\n\n请稍后重试。': '❌ AI auto-trading failed: {v1}\n\nPlease retry later.',
    '❌ {v1} 个交易对不可用': '❌ {v1} pairs unavailable',
    '❌ 交易执行过程中出现错误: {v1}': '❌ Trade execution error: {v1}',
    '❌ 保存失败: {v1}': '❌ Save failed: {v1}',
    '❌ 全自动循环失败: {v1}': '❌ Full-auto loop failed: {v1}',
    '❌ 回测过程中出现错误: {v1}': '❌ Backtest error: {v1}',
    '❌ 无法获取 {v1} 的K线数据，请检查股票代码或网络连接': '❌ Cannot fetch K-line data for {v1}; check the symbol or network',
    '❌ 监控过程中出现错误: {v1}': '❌ Monitoring error: {v1}',
    '❌ 策略生成过程中出现错误: {v1}': '❌ Strategy generation error: {v1}',
    '❌ 紧急止损失败: {v1}': '❌ Emergency stop failed: {v1}',
    '❌ 订单金额 ${v1:.2f} < $10 (Alpaca 最小限额), 请增加数量': '❌ Order ${v1:.2f} < $10 (Alpaca minimum), increase quantity',
    '❌ 调研过程中出现错误: {v1}\n\n请稍后重试。': '❌ Research error: {v1}\n\nPlease retry later.',
    '❌ 选股过程中出现错误: {v1}': '❌ Screening error: {v1}',
    '⭐ 关注股票调研结果 ({v1} 只)': '⭐ Watchlist research results ({v1})',
    '中等仓位 ({v1})': 'Medium position ({v1})',
    '买入失败: {v1}': 'Buy failed: {v1}',
    '交易执行失败: {v1}': 'Trade execution failed: {v1}',
    '会话上下文已恢复 ({v1} 条消息)': 'Session context restored ({v1} messages)',
    '候选上限: {v1}': 'Candidate limit: {v1}',
    '关注列表保存失败: {v1}': 'Watchlist save failed: {v1}',
    '关注股票调研失败: {v1}': 'Watchlist research failed: {v1}',
    '分配失败: {v1}': 'Allocation failed: {v1}',
    '创建': 'Create',
    '创建失败: {v1}': 'Creation failed: {v1}',
    '初始化完成: {v1}': 'Initialization done: {v1}',
    '删除失败: {v1}': 'Delete failed: {v1}',
    '加载 {v1} {v2} K线数据...': 'Loading {v1} {v2} candles...',
    '卖出失败: {v1}': 'Sell failed: {v1}',
    '发现 {v1} 个有效因子，可前往「📈 策略管理」创建使用这些因子的策略': 'Found {v1} effective factors; go to "📈 Strategies" to create a strategy with them',
    '启动失败: {v1}': 'Start failed: {v1}',
    '回测失败: {v1}': 'Backtest failed: {v1}',
    '回滚失败: {v1}': 'Rollback failed: {v1}',
    '因子搜索失败: {v1}': 'Factor search failed: {v1}',
    '实盘迁移模块未就绪: {v1}': 'Live migration module not ready: {v1}',
    '对话处理异常: {v1}': 'Dialogue error: {v1}',
    '小仓位 ({v1})，先试试水': 'Small position ({v1}), test the waters',
    '工作流执行失败: {v1}': 'Workflow execution failed: {v1}',
    '已关注 {v1}': 'Added {v1} to watchlist',
    '已取消关注 {v1}': 'Removed {v1} from watchlist',
    '建议{v1} {v2}': 'Suggest {v1} {v2}',
    '异常: {v1}': 'Error: {v1}',
    '当前: ***{v1} (留空不修改)': 'Current: ***{v1} (leave blank to keep)',
    '当前: {v1}': 'Current: {v1}',
    '恢复会话上下文失败: {v1}': 'Session context restore failed: {v1}',
    '意图路由异常 [{v1}]: {v2}': 'Intent routing error [{v1}]: {v2}',
    '手动交易面板加载异常: {v1}': 'Manual trade panel load error: {v1}',
    '执行失败: {v1}': 'Execution failed: {v1}',
    '智能助手加载失败: {v1}': 'AI assistant load failed: {v1}',
    '智能助手服务启动失败: {v1}': 'AI assistant service failed to start: {v1}',
    '暂无 {v1} 的K线数据，请检查数据源': 'No K-line data for {v1}; check data sources',
    '暂无解释: {v1}': 'No explanation: {v1}',
    '有效因子: {v1}': 'Effective factors: {v1}',
    '未识别为选股意图 (识别为: {v1})，使用手动参数': 'Not recognized as screening intent (got: {v1}), using manual params',
    '检测到 {v1} 条告警:': '{v1} alerts detected:',
    '模型: {v1} | 因子: {v2}个': 'Model: {v1} | Factors: {v2}',
    '模板加载失败: {v1}': 'Template load failed: {v1}',
    '正在买入 {v1} {v2}...': 'Buying {v1} {v2}...',
    '正在加载 {v1} K线数据...': 'Loading {v1} candles...',
    '正在加载 {v1} 公司信息...': 'Loading {v1} company info...',
    '正在卖出 {v1} {v2}...': 'Selling {v1} {v2}...',
    '正在深度调研 {v1} 只股票...': 'Deep-researching {v1} symbols...',
    '正在获取 {v1} 到期日期权数据...': 'Fetching {v1} expiry options data...',
    '正在获取 {v1} 期权链数据...': 'Fetching {v1} options chain...',
    '正在训练 {v1} ({v2}/{v3})...': 'Training {v1} ({v2}/{v3})...',
    '正在训练 {v1} 模型... (可能需要 1-5 分钟)': 'Training {v1} model... (may take 1-5 min)',
    '正在调研 {v1} 只关注股票... (可能需要30-60秒)': 'Researching {v1} watchlist symbols... (may take 30-60s)',
    '正在运行 {v1}...': 'Running {v1}...',
    '清仓或减仓到 {v1}': 'Liquidate or reduce to {v1}',
    '现价 ': 'Current price ',
    '监控失败: {v1}': 'Monitoring failed: {v1}',
    '目标驱动交易失败: {v1}': 'Goal-driven trading failed: {v1}',
    '竞赛失败: {v1}': 'Competition failed: {v1}',
    '策略生成失败: {v1}': 'Strategy generation failed: {v1}',
    '紧急清仓失败: {v1}': 'Emergency liquidation failed: {v1}',
    '组合管理模块未就绪: {v1}': 'Portfolio module not ready: {v1}',
    '股票池: {v1}只 | 仓位: {v2:.0%}': 'Stock pool: {v1} | Position: {v2:.0%}',
    '获取 {v1} 期权链失败，可能该股票没有期权数据': 'Failed to fetch {v1} options chain; the symbol may have no options',
    '获取价格数据异常: {v1}': 'Price data error: {v1}',
    '获取持仓信息异常: {v1}': 'Position info error: {v1}',
    '获取收益率曲线失败: {v1}': 'Yield curve fetch failed: {v1}',
    '获取期权链失败: {v1}': 'Options chain fetch failed: {v1}',
    '获取账户信息异常: {v1}': 'Account info error: {v1}',
    '行情策略_{v1}_{v2}': 'market_strategy_{v1}_{v2}',
    '训练 {v1} ({v2}/{v3})': 'Training {v1} ({v2}/{v3})',
    '训练失败: {v1}': 'Training failed: {v1}',
    '设置失败: {v1}': 'Setting failed: {v1}',
    '评级: {v1}': 'Rating: {v1}',
    '调度器模块未就绪: {v1}': 'Scheduler module not ready: {v1}',
    '调度器运行中 | {v1} 个任务': 'Scheduler running | {v1} jobs',
    '调研失败: {v1}': 'Research failed: {v1}',
    '账户获取失败: {v1}': 'Account fetch failed: {v1}',
    '迁移失败: {v1}': 'Migration failed: {v1}',
    '选股失败: {v1}': 'Screening failed: {v1}',
    '选股策略_{v1}_{v2}': 'screening_strategy_{v1}_{v2}',
    '选股结果缓存保存失败: {v1}': 'Screening result cache save failed: {v1}',
    '重仓 ({v1})，说明模型很看好': 'Heavy position ({v1}) — the model is very bullish',
    '链式步骤 {v1} 回复生成失败: {v2}': 'Chain step {v1} response generation failed: {v2}',
    '预估金额: ${v1:,.2f}': 'Estimated amount: ${v1:,.2f}',
    '预检失败: {v1}': 'Pre-check failed: {v1}',
    '风控检查不可用: {v1}': 'Risk check unavailable: {v1}',
    '🌍 国家: {v1}': '🌍 Country: {v1}',
    '🎯 每10次交易赢 {v1:.0f} 次，胜率偏低': '🎯 Wins {v1:.0f} of 10 trades, low win rate',
    '🎯 每10次交易赢 {v1:.0f} 次，胜率较高': '🎯 Wins {v1:.0f} of 10 trades, high win rate',
    '🎯 每10次交易赢 {v1:.0f} 次，还行': '🎯 Wins {v1:.0f} of 10 trades, decent',
    '🏛️ 交易所: {v1}': '🏛️ Exchange: {v1}',
    '🏷️ 行业: {v1}': '🏷️ Sector: {v1}',
    '👥 员工: {v1:,}': '👥 Employees: {v1:,}',
    '💡 AI 推荐理由: {v1}': '💡 AI reason: {v1}',
    '💡 实时行情模块加载中... ({v1})': '💡 Loading realtime module... ({v1})',
    '💡 暂无 {v1} 的公司概况数据 (需配置 Finnhub API Key)': '💡 No company profile for {v1} (Finnhub API key required)',
    '💡 股票池: {v1} | 模型: {v2} | 仓位: {v3:.0%}': '💡 Stock pool: {v1} | Model: {v2} | Position: {v3:.0%}',
    '💰 你的账户:': '💰 Your account:',
    '💰 每年平均赚 {v1:.0%}，不太理想': '💰 Earns {v1:.0%} per year, not great',
    '💰 每年平均赚 {v1:.0%}，勉强跑赢银行存款': '💰 Earns {v1:.0%} per year, barely beats a bank',
    '💰 每年平均赚 {v1:.0%}，比存银行好不少': '💰 Earns {v1:.0%} per year, much better than a bank',
    '💰 每年平均赚 {v1:.0%}，非常厉害！': '💰 Earns {v1:.0%} per year, impressive!',
    '📅 数据范围: {v1} ~ {v2} | 共 {v3} 根K线': '📅 Data range: {v1} ~ {v2} | {v3} candles',
    '📈 看涨期权 - {v1} ({v2} 个合约)': '📈 Calls - {v1} ({v2} contracts)',
    '📉 最惨的时候亏了 {v1:.0%}，可以接受': '📉 Worst loss was {v1:.0%}, acceptable',
    '📉 最惨的时候亏了 {v1:.0%}，需要心理准备': '📉 Worst loss was {v1:.0%}, brace yourself',
    '📉 最惨的时候只亏了 {v1:.0%}，很安全': '📉 Worst loss was only {v1:.0%}, very safe',
    '📉 看跌期权 - {v1} ({v2} 个合约)': '📉 Puts - {v1} ({v2} contracts)',
    '📊 PCR (成交量) 解读: {v1} | 看涨 {v2:,} vs 看跌 {v3:,}': '📊 PCR (volume) reading: {v1} | calls {v2:,} vs puts {v3:,}',
    '📊 {v1} K线图': '📊 {v1} K-line chart',
    '📊 {v1} · {v2} K线': '📊 {v1} · {v2} candles',
    '📊 {v1} × {v2:.4f} @ 市价': '📊 {v1} × {v2:.4f} @ market',
    '📊 选股结果 (扫描 {v1} 只 -> 入选 {v2} 只)': '📊 Screening results (scanned {v1} -> selected {v2})',
    '📊 风险控制能力: 一般 (夏普 {v1:.2f})': '📊 Risk control: average (Sharpe {v1:.2f})',
    '📊 风险控制能力: 不错 (夏普 {v1:.2f})': '📊 Risk control: good (Sharpe {v1:.2f})',
    '📊 风险控制能力: 优秀 (夏普 {v1:.2f})': '📊 Risk control: excellent (Sharpe {v1:.2f})',
    '📊 风险控制能力: 需改善 (夏普 {v1:.2f})': '📊 Risk control: needs work (Sharpe {v1:.2f})',
    '📋 {v1} {v2} 原始数据 (最近20条)': '📋 {v1} {v2} raw data (last 20)',
    '📋 {v1}报告:': '📋 {v1} report:',
    '📋 关注列表 ({v1} 只): {v2}': '📋 Watchlist ({v1}): {v2}',
    '📋 关注列表: {v1}': '📋 Watchlist: {v1}',
    '📋 已关注 {v1} 只股票': '📋 Watching {v1} symbols',
    '📋 策略列表 ({v1} 个)': '📋 Strategies ({v1})',
    '📌 已迁移到实盘 | 迁移时间: {v1}': '📌 Migrated to live | at {v1}',
    '📝 待保存的变更 ({v1} 项)': '📝 Pending changes ({v1})',
    '📡 连接: `{v1}`': '📡 Connection: `{v1}`',
    '📭 公司信息加载失败: {v1}': '📭 Company info load failed: {v1}',
    '🔎 单股调研结果 ({v1} 只)': '🔎 Single-stock research results ({v1})',
    '🔗 [官网]({v1})': '🔗 [Website]({v1})',
    '🔧 模式: `{v1}`': '🔧 Mode: `{v1}`',
    '🤔 我不太理解你的意思。\n\n你可以试试这样说:\n  • "我有10万美元，想赚20%" -> AI 全自主交易\n  • "帮我选5支10美元的潜力股" -> 智能选股\n  • "帮我看看今天美股有什么机会" -> 市场调研\n  • "帮我生成一个交易策略" -> 策略生成\n  • "回测一下策略" -> 回测验证\n  • "执行交易" -> 执行信号\n  • "查看持仓" -> 监控持仓\n  • "一键全自动" -> 全流程自动运转\n  • "紧急清仓" -> 紧急止损\n  • "帮助" -> 查看所有功能\n': '🤔 I didn\'t quite understand that.\n\nTry something like:\n  • "I have $100k, want 20% gain" -> full AI auto-trading\n  • "pick 5 potential stocks under $10" -> smart screening\n  • "any opportunities in the US market today" -> market research\n  • "generate a trading strategy" -> strategy generation\n  • "backtest the strategy" -> backtest\n  • "execute trades" -> execute signals\n  • "view positions" -> monitor positions\n  • "full auto" -> end-to-end pipeline\n  • "emergency liquidation" -> emergency stop\n  • "help" -> show all features\n',
    '🤖 AI 量化交易助手 - 使用指南\n{v1}\n\n📋 我能帮你做这些事:\n\n  🎯 目标驱动 AI 自主交易 (最省心!)\n     "我有10万美元，想赚20%"\n     "5万本金，目标翻倍，风险别太大"\n     "10万块，想赚5万"\n     -> AI 全自主: 选股/选策略/训练/回测/下单\n\n  1️⃣ 智能选股\n     "帮我选5支10美元的潜力股"\n     "选3只20-50美元的科技股"\n\n  2️⃣ 市场调研\n     "帮我看看今天有什么机会"\n     "扫描一下市场新闻"\n\n  3️⃣ 策略生成\n     "帮我生成一个交易策略"\n     "为 AAPL 和 MSFT 创建策略"\n\n  4️⃣ 回测验证\n     "回测一下策略"\n     "测试策略的历史表现"\n\n  5️⃣ 执行交易\n     "执行交易信号"\n     "用模拟盘下单"\n\n  6️⃣ 持仓监控\n     "查看持仓"\n     "今天的交易日报"\n\n  7️⃣ 全自动模式\n     "一键全自动"\n     "帮我搞定全流程"\n\n  8️⃣ 紧急止损\n     "紧急清仓"\n     "全部卖出"\n\n  9️⃣ 系统状态\n     "系统状态"\n     "当前配置"\n\n💡 提示: 用大白话告诉我想做什么就行，我会理解你的意思！\n⚠️ 风险提示: 所有分析仅供参考，不构成投资建议。': '🤖 AI Quant Trading Assistant - Guide\n{v1}\n\n📋 Here\'s what I can do:\n\n  🎯 Goal-driven AI auto-trading (easiest!)\n     "I have $100k, want 20% gain"\n     "50k capital, target 2x, low risk"\n     "100k, want 50k profit"\n     -> Full AI: screening / strategy / training / backtest / orders\n\n  1️⃣ Smart screening\n     "pick 5 potential stocks under $10"\n     "3 tech stocks between $20-50"\n\n  2️⃣ Market research\n     "any opportunities today"\n     "scan market news"\n\n  3️⃣ Strategy generation\n     "generate a trading strategy"\n     "create strategies for AAPL and MSFT"\n\n  4️⃣ Backtest\n     "backtest the strategy"\n     "test the strategy on history"\n\n  5️⃣ Execute trades\n     "execute trade signals"\n     "place paper orders"\n\n  6️⃣ Position monitoring\n     "view positions"\n     "today\'s trade report"\n\n  7️⃣ Full-auto mode\n     "full auto"\n     "run the whole pipeline"\n\n  8️⃣ Emergency stop\n     "emergency liquidation"\n     "sell everything"\n\n  9️⃣ System status\n     "system status"\n     "current config"\n\n💡 Tip: just tell me what you want in plain language — I\'ll understand!\n⚠️ Disclaimer: all analysis is for reference only, not investment advice.',
    '🧠 AI 解析结果: 数量={v1}, 价格={v2}~{v3}, 行业={v4}, 关键词={v5}': '🧠 AI parsed: count={v1}, price={v2}~{v3}, sector={v4}, keywords={v5}',

    # ── chat_server (AI 助手后端) ──
    '会话不存在': 'Session not found',
    '会话已删除': 'Session deleted',
    '历史对话': 'History',
    '对话已清空': 'Conversation cleared',
    '抱歉，处理请求时出错: {v1}': 'Sorry, an error occurred: {v1}',
    '聊天页面加载失败': 'Failed to load chat page',
    "止损触发": "Stop-loss triggered",
    "组合风控告警": "Portfolio risk alert",
    "价格异动": "Price anomaly",
    "加密货币风控拦截": "Crypto risk blocked",
    "加密货币止损触发": "Crypto stop-loss triggered",
    "加密货币组合风控告警": "Crypto portfolio risk alert",
    "模拟盘": "Paper",
    "实盘": "Live",
    "亏损": "loss",
    "自动卖出": "auto-sold",
    "持仓数": "Position count",
    "超过上限": "exceeds limit",
    "单票最大仓位": "Max single position",
    "被拦截": "blocked",
    "买入后仓位": "post-buy position",
    "上限": "limit",
    "当前": "current",
    "触发": "triggered",
    "告警": "alert",
    "风控": "Risk control",

    # ─── AI 智能助手 (chat_server / function_agent) ───
    "新会话": "New Chat",
    "消息不能为空": "Message cannot be empty",
    "AI 思考过程出现异常": "AI thinking error",
    "正在理解你的请求": "Understanding your request",
    "正在执行": "Executing",

    # ─── 宏观经济指标页面 (page_macro) ───
    "🌐 宏观经济指标": "🌐 Macro Indicators",
    "美联储宏观数据 (FRED) · 美债收益率曲线 · 宏观风险评分": "Fed Macro Data (FRED) · Treasury Yield Curve · Macro Risk Score",
    "⚠️ FRED API 未配置，无法获取宏观数据。请在 .env 中设置 FRED_API_KEY": "⚠️ FRED API not configured. Set FRED_API_KEY in .env to get macro data",
    "📊 宏观风险评分": "📊 Macro Risk Score",
    "正在获取宏观数据...": "Fetching macro data...",
    "获取宏观数据失败，请检查 FRED API 配置": "Failed to fetch macro data. Check FRED API config",
    "风险评分": "Risk Score",
    "风险等级": "Risk Level",
    "风险等级: {v1}": "Risk Level: {v1}",
    "💡 建议: {v1}": "💡 Suggestion: {v1}",
    "**风险信号:**": "**Risk Signals:**",
    "未知": "Unknown",
    "极高": "Very High",
    "极低": "Very Low",
    "高": "High",
    "中等": "Medium",
    "低": "Low",
    "建议降低仓位，增加防御性配置": "Consider reducing positions and increasing defensive allocation",
    "建议谨慎操作，控制仓位": "Trade cautiously and control position sizes",
    "保持中性仓位，关注风险信号": "Keep neutral positions and watch risk signals",
    "宏观环境健康，可正常配置": "Macro environment is healthy, normal allocation OK",
    "🔴 收益率曲线倒挂 (经济衰退信号)": "🔴 Yield curve inverted (recession signal)",
    "🟡 收益率曲线趋平": "🟡 Yield curve flattening",
    "🔴 高通胀 (CPI同比 {v1}%)": "🔴 High inflation (CPI YoY {v1}%)",
    "🟡 通胀偏高 (CPI同比 {v1}%)": "🟡 Elevated inflation (CPI YoY {v1}%)",
    "🟡 通缩风险 (CPI同比 {v1}%)": "🟡 Deflation risk (CPI YoY {v1}%)",
    "🔴 高失业率 ({v1}%)": "🔴 High unemployment ({v1}%)",
    "🔴 失业率快速上升": "🔴 Unemployment rising rapidly",
    "🔴 高利率环境 ({v1}%)": "🔴 High rate environment ({v1}%)",
    "🟡 利率偏高 ({v1}%)": "🟡 Elevated rates ({v1}%)",
    "获取宏观数据失败: {v1}": "Failed to fetch macro data: {v1}",
    "📉 美债收益率曲线": "📉 Treasury Yield Curve",
    "10Y-2Y 利差": "10Y-2Y Spread",
    "10Y-3M 利差": "10Y-3M Spread",
    "🚨 {v1} - 收益率倒挂是经济衰退的领先指标!": "🚨 {v1} - Yield inversion is a leading recession indicator!",
    "美债收益率曲线 (当前)": "Treasury Yield Curve (Current)",
    "期限": "Maturity",
    "收益率 (%)": "Yield (%)",
    "😱 VIX 恐慌指数": "😱 VIX Fear Index",
    "VIX 值": "VIX Value",
    "涨跌": "Change",
    "涨跌幅": "Change %",
    "获取 VIX 数据失败": "Failed to fetch VIX data",
    "{v1} VIX 风险等级: {v2} | 数据时间: {v3}": "{v1} VIX Risk Level: {v2} | Data as of: {v3}",
    "📡 数据源信息": "📡 Data Sources",
    "数据来源: FRED (Federal Reserve Economic Data)": "Source: FRED (Federal Reserve Economic Data)",
    "缓存策略: 宏观数据6小时, 收益率曲线12小时, VIX 5分钟": "Cache: macro 6h, yield curve 12h, VIX 5min",
    "⚠️ 宏观数据有发布延迟，CPI/就业等月度指标通常滞后1-2个月": "⚠️ Macro data has publication lag; monthly indicators (CPI/jobs) are typically 1-2 months behind",
    "未配置": "Not configured",
    # ── 宏观指标页 (FRED) ──
    "CPI (居民消费价格指数)": "CPI (Consumer Price Index)",
    "核心CPI (剔除食品能源)": "Core CPI (ex Food & Energy)",
    "PCE物价指数": "PCE Price Index",
    "10年期通胀预期": "10Y Inflation Expectation",
    "失业率": "Unemployment Rate",
    "非农就业人数": "Non-Farm Payrolls",
    "初请失业金人数": "Initial Jobless Claims",
    "GDP (国内生产总值)": "GDP (Gross Domestic Product)",
    "实际GDP": "Real GDP",
    "工业生产指数": "Industrial Production Index",
    "联邦基金利率": "Federal Funds Rate",
    "有效联邦基金利率(日)": "Effective Fed Funds Rate (Daily)",
    "2年期美债收益率": "2Y Treasury Yield",
    "5年期美债收益率": "5Y Treasury Yield",
    "10年期美债收益率": "10Y Treasury Yield",
    "30年期美债收益率": "30Y Treasury Yield",
    "10年-2年利差": "10Y-2Y Spread",
    "10年-3月利差": "10Y-3M Spread",
    "VIX恐慌指数(FRED)": "VIX Fear Index (FRED)",
    "2年": "2Y",
    "10年": "10Y",
    "30年": "30Y",
    "⚠️ 收益率倒挂 (10Y-2Y)": "⚠️ Yield curve inverted (10Y-2Y)",
    "⚠️ 收益率倒挂 (10Y-3M)": "⚠️ Yield curve inverted (10Y-3M)",
    "潜力": "Momentum",
    "元数据": "Metadata",
    "智能助手加载中": "AI Assistant loading",
    # ── 策略模板 (engine/strategy_templates.py 渲染层翻译) ──
    "稳健成长": "Steady Growth",
    "动量追踪": "Momentum Rider",
    "价值挖掘": "Value Hunter",
    "科技先锋": "Tech Pioneer",
    "均衡配置": "Balanced All",
    "大盘蓝筹股 + LightGBM 模型，追求稳定增长。适合风险偏好较低的投资者，持仓集中在大市值公司，换手率低。": "Large-cap blue chips + LightGBM model for steady growth. Suits risk-averse investors, focused on large-cap companies with low turnover.",
    "以动量因子为核心，捕捉趋势行情。适合能承受一定波动的投资者，追求超额收益，换手率中等偏高。": "Momentum-factor driven, captures trending markets. Suits investors who can tolerate some volatility, seeking excess returns with medium-high turnover.",
    "以基本面因子为主，寻找被低估的优质公司。适合长期投资者，换手率低，持仓周期长。": "Fundamental-factor based, finds undervalued quality companies. Suits long-term investors with low turnover and long holding periods.",
    "专注科技板块，使用 LSTM 深度学习模型捕捉非线性模式。适合看好科技行业、能承受较高波动的投资者。": "Tech-sector focused, uses LSTM deep learning to capture non-linear patterns. Suits investors bullish on tech who can tolerate higher volatility.",
    "多因子均衡策略，兼顾技术面和基本面。适合大多数投资者，风险和收益均衡，中等仓位。": "Multi-factor balanced strategy combining technical and fundamental analysis. Suits most investors, balanced risk-return with medium positions.",
    "蓝筹": "Blue-chip",
    "稳健": "Steady",
    "低换手": "Low turnover",
    "适合新手": "Beginner-friendly",
    "动量": "Momentum",
    "趋势": "Trend",
    "高换手": "High turnover",
    "追求超额": "Alpha-seeking",
    "价值": "Value",
    "基本面": "Fundamental",
    "长期持有": "Long-term",
    "科技": "Tech",
    "深度学习": "Deep learning",
    "高波动": "High volatility",
    "均衡": "Balanced",
    "多因子": "Multi-factor",
    "中等风险": "Medium risk",
    "适合大多数": "For most investors",
    # ── 宏观指标单位 (fred_client.py 渲染层翻译) ──
    "指数": "Index",
    "千人": "Thousands",
    "万人": "10k persons",
    "亿美元": "$100M",
    "十亿美元": "Billion USD",
    "万亿美元": "$T",
    "百分比": "Percent",
    "美元/桶": "USD/bbl",
    "美元/盎司": "USD/oz",
    "立方英尺": "Cu ft",
    "月": "Monthly",
    "周": "Weekly",
    "日": "Daily",
    "季": "Quarterly",
    "年": "Yearly",
    "自动策略_{v1}": "Auto Strategy_{v1}",
    "停止失败: {v1}": "Stop failed: {v1}",
}
