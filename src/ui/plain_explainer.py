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

from src.models import (
    BacktestResult,
    TradeSignal, RiskStatus, Account, Position,
)

# i18n: 翻译函数
from src.ui.i18n import t


class PlainExplainer:
    """通俗解读引擎"""

    # 术语对照表
    GLOSSARY = {
        t("年化收益"): t("每年平均赚多少"),
        t("夏普比率"): t("风险调整后的收益（越高越好）"),
        t("最大回撤"): t("最惨的时候亏多少"),
        t("胜率"): t("赚钱的交易占多少比例"),
        "IC": t("因子预测准不准的指标"),
        t("仓位"): t("一只股票占总资金的比例"),
        t("止损"): t("亏到一定程度自动卖出"),
        t("止盈"): t("赚到一定程度自动卖出"),
        t("回撤"): t("从最高点跌下来多少"),
        t("波动率"): t("价格上下波动的剧烈程度"),
        t("夏普"): t("风险调整后的收益（越高越好）"),
        t("索提诺"): t("只考虑亏损的夏普比率"),
        "Paper Trading": t("模拟盘（不用真钱）"),
        "Qlib": t("微软开源的量化框架"),
        "LightGBM": t("一种机器学习模型"),
        "Alpha158": t("微软Qlib的158个因子库"),
    }

    # 风险等级
    RISK_LEVELS = {
        "low": (t("🟢 低风险"), t("比较安全，但收益可能也有限")),
        "medium": (t("🟡 中风险"), t("有一定波动，适合大多数投资者")),
        "high": (t("🔴 高风险"), t("波动较大，请谨慎对待")),
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
                t('explain_term 期望 str 或 None 类型，得到 {v1}', v1=type(term).__name__)
            )
        return self.GLOSSARY.get(term, t('暂无解释: {v1}', v1=term))

    def explain_backtest(self, result: BacktestResult) -> str:
        """通俗解读回测结果"""
        lines = []

        # 评级 - 通过 .value 比较，兼容不同枚举实例
        grade_val = result.grade.value if hasattr(result.grade, 'value') else str(result.grade)
        grade_map = {
            "A": t("⭐⭐⭐⭐⭐ 非常优秀！"),
            "B": t("⭐⭐⭐⭐ 表现不错"),
            "C": t("⭐⭐⭐ 一般般"),
            "D": t("⭐⭐ 表现不太好"),
        }
        lines.append(t('评级: {v1}', v1=grade_map.get(grade_val, '未知')))
        lines.append("")

        # 年化收益
        ar = result.annual_return
        if ar > 0.20:
            lines.append(t('💰 每年平均赚 {v1:.0%}，非常厉害！', v1=ar))
        elif ar > 0.10:
            lines.append(t('💰 每年平均赚 {v1:.0%}，比存银行好不少', v1=ar))
        elif ar > 0.03:
            lines.append(t('💰 每年平均赚 {v1:.0%}，勉强跑赢银行存款', v1=ar))
        else:
            lines.append(t('💰 每年平均赚 {v1:.0%}，不太理想', v1=ar))

        # 最大回撤
        mdd = result.max_drawdown
        if mdd < 0.05:
            lines.append(t('📉 最惨的时候只亏了 {v1:.0%}，很安全', v1=mdd))
        elif mdd < 0.15:
            lines.append(t('📉 最惨的时候亏了 {v1:.0%}，可以接受', v1=mdd))
        else:
            lines.append(t('📉 最惨的时候亏了 {v1:.0%}，需要心理准备', v1=mdd))

        # 夏普
        sr = result.sharpe_ratio
        if sr > 1.5:
            lines.append(t('📊 风险控制能力: 优秀 (夏普 {v1:.2f})', v1=sr))
        elif sr > 1.0:
            lines.append(t('📊 风险控制能力: 不错 (夏普 {v1:.2f})', v1=sr))
        elif sr > 0.5:
            lines.append(t('📊 风险控制能力: 一般 (夏普 {v1:.2f})', v1=sr))
        else:
            lines.append(t('📊 风险控制能力: 需改善 (夏普 {v1:.2f})', v1=sr))

        # 胜率
        wr = result.win_rate
        if wr > 0.6:
            lines.append(t('🎯 每10次交易赢 {v1:.0f} 次，胜率较高', v1=wr*10))
        elif wr > 0.5:
            lines.append(t('🎯 每10次交易赢 {v1:.0f} 次，还行', v1=wr*10))
        else:
            lines.append(t('🎯 每10次交易赢 {v1:.0f} 次，胜率偏低', v1=wr*10))

        lines.append("")
        lines.append(t("⚠️ 记住: 过去的成绩不代表未来，投资有风险！"))

        return "\n".join(lines)

    def explain_signal(self, signal: TradeSignal) -> str:
        """通俗解读交易信号"""
        direction = t("买入") if signal.direction.value == "buy" else t("卖出")
        weight = f"{signal.target_weight:.0%}"

        parts = [t('建议{v1} {v2}', v1=direction, v2=signal.symbol)]

        if signal.direction.value == "buy":
            if signal.target_weight > 0.20:
                parts.append(t('重仓 ({v1})，说明模型很看好', v1=weight))
            elif signal.target_weight > 0.10:
                parts.append(t('中等仓位 ({v1})', v1=weight))
            else:
                parts.append(t('小仓位 ({v1})，先试试水', v1=weight))
        else:
            parts.append(t('清仓或减仓到 {v1}', v1=weight))

        confidence_label = ""
        if signal.confidence > 0.7:
            confidence_label = t("信心很强")
        elif signal.confidence > 0.4:
            confidence_label = t("信心一般")
        else:
            confidence_label = t("信心较弱")

        result = "，".join(parts)
        if confidence_label:
            result += f" ({confidence_label})"

        if signal.reason:
            result += t('\n   理由: {v1}', v1=signal.reason)

        return result

    def explain_risk(self, risk: RiskStatus) -> str:
        """通俗解读风控状态"""
        lines = []
        lines.append(t("🛡️ 风控体检报告:"))

        if not risk.is_alert:
            lines.append(t("  ✅ 一切正常，没有触发任何风控警报"))
        else:
            lines.append(t('  ⚠️ 有 {v1} 条警报:', v1=len(risk.alerts)))
            for alert in risk.alerts:
                lines.append(f"    • {alert}")

        # 回撤解读
        dd = risk.current_drawdown
        if dd > 0:
            if dd < 0.05:
                lines.append(t('  当前回撤 {v1:.1%}，很安全', v1=dd))
            elif dd < 0.10:
                lines.append(t('  当前回撤 {v1:.1%}，需要注意', v1=dd))
            elif dd < 0.15:
                lines.append(t('  当前回撤 {v1:.1%}，接近警戒线！', v1=dd))
            else:
                lines.append(t('  当前回撤 {v1:.1%}，已超过上限，建议暂停！', v1=dd))

        return "\n".join(lines)

    def explain_account(self, account: Account) -> str:
        """通俗解读账户状态"""
        lines = []
        lines.append(t('💰 你的账户:'))
        lines.append(t('  总资产: ${v1:,.2f}', v1=account.equity))

        mode = t("📝 模拟盘") if account.trading_mode.value == "paper" else t("🔴 实盘")
        lines.append(t('  交易模式: {v1}', v1=mode))
        lines.append(t('  可用现金: ${v1:,.2f}', v1=account.cash))
        lines.append(t('  购买力: ${v1:,.2f}', v1=account.buying_power))

        # 日盈亏
        if account.last_equity > 0:
            daily_pnl = account.equity - account.last_equity
            daily_pct = daily_pnl / account.last_equity
            icon = "📈" if daily_pnl >= 0 else "📉"
            lines.append(t('  今日盈亏: {v1} ${v2:+,.2f} ({v3:+.1%})', v1=icon, v2=daily_pnl, v3=daily_pct))

        return "\n".join(lines)

    def explain_position(self, position: Position) -> str:
        """通俗解读单个持仓"""
        lines = []
        pnl_icon = t("🟢赚") if position.unrealized_pnl >= 0 else t("🔴亏")
        lines.append(t('  {v1}: {v2:.2f}股 | {v3} ${v4:+,.2f} ({v5:+.1%}) | 占仓位 {v6:.1%}', v1=position.symbol, v2=position.qty, v3=pnl_icon, v4=position.unrealized_pnl, v5=position.unrealized_pnl_pct, v6=position.weight))
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
            return t("🌙 夜深了，还在研究市场吗？")
        elif hour < 12:
            return t("🌅 早上好！新的一天开始了")
        elif hour < 18:
            return t("☀️ 下午好！市场正在交易中")
        else:
            return t("🌆 晚上好！来复盘一下今天的交易吧")

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
