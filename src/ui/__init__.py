# ui package
"""
小白体验层 - 用户交互界面
=====================================================
- plain_explainer: 通俗解读引擎 (量化黑话翻译成人话)
- nl_router: 自然语言意图路由 (理解小白输入)
- cli: 命令行交互界面
- dashboard: Streamlit 可视化仪表盘

注意: 子模块按需导入，不在 __init__ 中触发完整依赖链
"""

__all__ = [
    "PlainExplainer", "plain_explainer",
    "NLRouter", "Intent", "nl_router",
]


def __getattr__(name):
    """PEP 562: 延迟导入，避免 __init__ 触发完整依赖链"""
    if name in ("PlainExplainer", "plain_explainer"):
        from src.ui.plain_explainer import PlainExplainer, plain_explainer
        return PlainExplainer if name == "PlainExplainer" else plain_explainer
    if name in ("NLRouter", "Intent", "nl_router"):
        from src.ui.nl_router import NLRouter, Intent, nl_router
        if name == "NLRouter":
            return NLRouter
        if name == "Intent":
            return Intent
        return nl_router
    raise AttributeError(f"module 'src.ui' has no attribute {name!r}")
