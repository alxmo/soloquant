"""
通俗解读引擎 - 将量化专业术语翻译成小白语言
=====================================================
职责:
- 术语翻译 (年化收益/夏普/回撤/IC 等翻译成人话)
- 风险等级解读
- 建议生成
- 情绪化表达 (让交互更亲切)
"""

import re
from typing import Any, Optional

from src.models import (
    BacktestResult, StrategyConfig, StrategyGrade,
    TradeSignal, RiskStatus, Account, Position,
)


class PlainExplainer:
    """通俗解读引擎"""

    # 术语对照表
    GLOSSARY = {
        "年化收益": "每年平均赚多少",
        "夏普比率": "风险调整后的收益（越高越好）",
        "最大回撤": "最惨的时候亏多少",
        "胜率": "赚钱的交易占多少比例",
        "IC": "因子预测准不准的指标",
        "仓位": "一只股票占总资金的比例",
        "止损": "亏到一定程度自动卖出",
        "止盈": "赚到一定程度自动卖出",
        "回撤": "从最高点跌下来多少",
        "波动率": "价格上下波动的剧烈程度",
        "夏普": "风险调整后的收益（越高越好）",
        "索提诺": "只考虑亏损的夏普比率",
        "Paper Trading": "模拟盘（不用真钱）",
        "Qlib": "微软开源的量化框架",
        "LightGBM": "一种机器学习模型",
        "Alpha158": "微软Qlib的158个因子库",
    }

    # 风险等级
    RISK_LEVELS = {
        "low": ("🟢 低风险", "比较安全，但收益可能也有限"),
        "medium": ("🟡 中风险", "有一定波动，适合大多数投资者"),
        "high": ("🔴 高风险", "波动较大，请谨慎对待"),
    }

    def explain_term(self, term: str) -> str:
        """
        解释单个术语

        - 已知术语: 返回 GLOSSARY 中的解释
        - 未知字符串: 返回 "暂无解释: {term}"
        - None: 返回 "暂无解释: None"
        - 非字符串且非 None (int/float/bool/list/dict 等): 抛出 TypeError
        """
        if not isinstance(term, str) and term is not None:
            raise TypeError(
                f"explain_term 期望 str 或 None 类型，得到 {type(term).__name__}"
            )
        return self.GLOSSARY.get(term, f"暂无解释: {term}")

    def explain_backtest(self, result: BacktestResult) -> str:
        """通俗解读回测结果"""
        lines = []

        # 评级 - 通过 .value 比较，兼容不同枚举实例
        grade_val = result.grade.value if hasattr(result.grade, 'value') else str(result.grade)
        grade_map = {
            "A": "⭐⭐⭐⭐⭐ 非常优秀！",
            "B": "⭐⭐⭐⭐ 表现不错",
            "C": "⭐⭐⭐ 一般般",
            "D": "⭐⭐ 表现不太好",
        }
        lines.append(f"评级: {grade_map.get(grade_val, '未知')}")
        lines.append("")

        # 年化收益
        ar = result.annual_return
        if ar > 0.20:
            lines.append(f"💰 每年平均赚 {ar:.0%}，非常厉害！")
        elif ar > 0.10:
            lines.append(f"💰 每年平均赚 {ar:.0%}，比存银行好不少")
        elif ar > 0.03:
            lines.append(f"💰 每年平均赚 {ar:.0%}，勉强跑赢银行存款")
        else:
            lines.append(f"💰 每年平均赚 {ar:.0%}，不太理想")

        # 最大回撤
        mdd = result.max_drawdown
        if mdd < 0.05:
            lines.append(f"📉 最惨的时候只亏了 {mdd:.0%}，很安全")
        elif mdd < 0.15:
            lines.append(f"📉 最惨的时候亏了 {mdd:.0%}，可以接受")
        else:
            lines.append(f"📉 最惨的时候亏了 {mdd:.0%}，需要心理准备")

        # 夏普
        sr = result.sharpe_ratio
        if sr > 1.5:
            lines.append(f"📊 风险控制能力: 优秀 (夏普 {sr:.2f})")
        elif sr > 1.0:
            lines.append(f"📊 风险控制能力: 不错 (夏普 {sr:.2f})")
        elif sr > 0.5:
            lines.append(f"📊 风险控制能力: 一般 (夏普 {sr:.2f})")
        else:
            lines.append(f"📊 风险控制能力: 需改善 (夏普 {sr:.2f})")

        # 胜率
        wr = result.win_rate
        if wr > 0.6:
            lines.append(f"🎯 每10次交易赢 {wr*10:.0f} 次，胜率较高")
        elif wr > 0.5:
            lines.append(f"🎯 每10次交易赢 {wr*10:.0f} 次，还行")
        else:
            lines.append(f"🎯 每10次交易赢 {wr*10:.0f} 次，胜率偏低")

        lines.append("")
        lines.append("⚠️ 记住: 过去的成绩不代表未来，投资有风险！")

        return "\n".join(lines)

    def explain_signal(self, signal: TradeSignal) -> str:
        """通俗解读交易信号"""
        direction = "买入" if signal.direction.value == "buy" else "卖出"
        weight = f"{signal.target_weight:.0%}"

        parts = [f"建议{direction} {signal.symbol}"]

        if signal.direction.value == "buy":
            if signal.target_weight > 0.20:
                parts.append(f"重仓 ({weight})，说明模型很看好")
            elif signal.target_weight > 0.10:
                parts.append(f"中等仓位 ({weight})")
            else:
                parts.append(f"小仓位 ({weight})，先试试水")
        else:
            parts.append(f"清仓或减仓到 {weight}")

        confidence_label = ""
        if signal.confidence > 0.7:
            confidence_label = "信心很强"
        elif signal.confidence > 0.4:
            confidence_label = "信心一般"
        else:
            confidence_label = "信心较弱"

        result = "，".join(parts)
        if confidence_label:
            result += f" ({confidence_label})"

        if signal.reason:
            result += f"\n   理由: {signal.reason}"

        return result

    def explain_risk(self, risk: RiskStatus) -> str:
        """通俗解读风控状态"""
        lines = []
        lines.append("🛡️ 风控体检报告:")

        if not risk.is_alert:
            lines.append("  ✅ 一切正常，没有触发任何风控警报")
        else:
            lines.append(f"  ⚠️ 有 {len(risk.alerts)} 条警报:")
            for alert in risk.alerts:
                lines.append(f"    • {alert}")

        # 回撤解读
        dd = risk.current_drawdown
        if dd > 0:
            if dd < 0.05:
                lines.append(f"  当前回撤 {dd:.1%}，很安全")
            elif dd < 0.10:
                lines.append(f"  当前回撤 {dd:.1%}，需要注意")
            elif dd < 0.15:
                lines.append(f"  当前回撤 {dd:.1%}，接近警戒线！")
            else:
                lines.append(f"  当前回撤 {dd:.1%}，已超过上限，建议暂停！")

        return "\n".join(lines)

    def explain_account(self, account: Account) -> str:
        """通俗解读账户状态"""
        lines = []
        lines.append(f"💰 你的账户:")
        lines.append(f"  总资产: ${account.equity:,.2f}")

        mode = "📝 模拟盘" if account.trading_mode.value == "paper" else "🔴 实盘"
        lines.append(f"  交易模式: {mode}")
        lines.append(f"  可用现金: ${account.cash:,.2f}")
        lines.append(f"  购买力: ${account.buying_power:,.2f}")

        # 日盈亏
        if account.last_equity > 0:
            daily_pnl = account.equity - account.last_equity
            daily_pct = daily_pnl / account.last_equity
            icon = "📈" if daily_pnl >= 0 else "📉"
            lines.append(f"  今日盈亏: {icon} ${daily_pnl:+,.2f} ({daily_pct:+.1%})")

        return "\n".join(lines)

    def explain_position(self, position: Position) -> str:
        """通俗解读单个持仓"""
        lines = []
        pnl_icon = "🟢赚" if position.unrealized_pnl >= 0 else "🔴亏"
        lines.append(f"  {position.symbol}: {position.qty:.2f}股 | "
                      f"{pnl_icon} ${position.unrealized_pnl:+,.2f} "
                      f"({position.unrealized_pnl_pct:+.1%}) | "
                      f"占仓位 {position.weight:.1%}")
        return "\n".join(lines)

    def risk_level(self, backtest: BacktestResult) -> tuple[str, str]:
        """根据回测结果判断风险等级"""
        mdd = backtest.max_drawdown
        if mdd < 0.08:
            level = "low"
        elif mdd < 0.15:
            level = "medium"
        else:
            level = "high"
        return self.RISK_LEVELS[level]

    def friendly_greeting(self, hour: int = None) -> str:
        """根据时间生成问候语"""
        from datetime import datetime
        if hour is None:
            hour = datetime.now().hour

        if hour < 6:
            return "🌙 夜深了，还在研究市场吗？"
        elif hour < 12:
            return "🌅 早上好！新的一天开始了"
        elif hour < 18:
            return "☀️ 下午好！市场正在交易中"
        else:
            return "🌆 晚上好！来复盘一下今天的交易吧"

    def translate(self, text: str) -> str:
        """
        翻译专业术语 (单次替换，避免级联)

        使用正则表达式一次性匹配所有术语，按长度降序排列
        确保长术语（如"夏普比率"）优先于短术语（如"夏普"）被匹配，
        避免短术语替换后破坏长术语。
        """
        if not text:
            return text
        # 按长度降序排列，长术语优先匹配
        sorted_terms = sorted(self.GLOSSARY.keys(), key=len, reverse=True)
        # 构建正则模式，转义特殊字符
        pattern = "|".join(re.escape(term) for term in sorted_terms)

        def _replace_match(m):
            term = m.group(0)
            explanation = self.GLOSSARY[term]
            return f"{term}({explanation})"

        return re.sub(pattern, _replace_match, text)


# 全局实例
plain_explainer = PlainExplainer()
