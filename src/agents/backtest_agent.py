"""
Backtest Agent - 回测验证师
=====================================================
职责:
1. 接收策略配置 (来自 Strategy Agent)
2. 运行历史回测
3. 绩效评估 (年化/夏普/回撤/胜率)
4. 策略评级 (A/B/C/D)
5. 评级不达标时回退建议

工具链:
- backtest_engine.run_backtest() -> 回测执行
- strategy_manager.record_backtest() -> 保存结果
"""

from typing import Any
from datetime import datetime, timedelta

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.engine.backtest_engine import backtest_engine
from src.engine.strategy_manager import strategy_manager
from src.models import BacktestResult, StrategyConfig, StrategyGrade


class BacktestAgent(BaseAgent):
    """回测验证师 Agent"""

    def __init__(self):
        super().__init__(name="backtest", role="回测验证师")

    def _execute(
        self,
        strategy: StrategyConfig = None,
        strategy_result: dict = None,
        start: str = None,
        end: str = None,
        initial_capital: float = 100000,
        quick_mode: bool = False,
        **kwargs,
    ) -> BacktestResult:
        """
        执行回测验证

        Args:
            strategy: 策略配置 (优先)
            strategy_result: Strategy Agent 的输出 (从中提取 strategy)
            start: 回测开始日期 (默认1年前)
            end: 回测结束日期 (默认今天)
            initial_capital: 初始资金
            quick_mode: 快速模式 (最近30天)

        Returns:
            BacktestResult
        """
        # 1. 获取策略
        if strategy is None and strategy_result:
            strategy = strategy_result.get("strategy")
        if strategy is None:
            raise ValueError("未提供策略配置")

        # 2. 设置回测时间范围
        if quick_mode:
            start = start or (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        else:
            start = start or (datetime.now().replace(year=datetime.now().year - 1)).strftime("%Y-%m-%d")
        end = end or datetime.now().strftime("%Y-%m-%d")

        logger.info(f"🔬 回测开始 | 策略: {strategy.name}")
        logger.info(f"  时间范围: {start} ~ {end}")
        logger.info(f"  初始资金: ${initial_capital:,.2f}")
        logger.info(f"  股票池: {strategy.stock_pool}")

        # 3. 运行回测
        result = backtest_engine.run_backtest(
            strategy=strategy,
            start=start,
            end=end,
            initial_capital=initial_capital,
        )

        # 4. 保存回测结果
        try:
            strategy_manager.record_backtest(strategy.strategy_id, result)
        except Exception as e:
            logger.warning(f"保存回测结果失败: {e}")

        # 5. 更新策略状态
        if result.grade in [StrategyGrade.A, StrategyGrade.B]:
            strategy_manager.activate_strategy(strategy.strategy_id)
            logger.info(f"  策略评级 {result.grade} -> 已激活")
        else:
            strategy_manager.set_testing(strategy.strategy_id)
            logger.info(f"  策略评级 {result.grade} -> 继续测试")

        # 6. 生成增强的小白解读 (AI 增强: 优先使用 AI 深度解读)
        result.plain_summary = self._enhance_plain_summary(result, strategy)
        from src.llm.ai_analyzer import ai_analyzer
        ai_analysis = ai_analyzer.analyze_backtest(
            strategy_name=strategy.name,
            grade=result.grade.value if hasattr(result.grade, 'value') else str(result.grade),
            annual_return=result.annual_return,
            max_drawdown=result.max_drawdown,
            sharpe_ratio=result.sharpe_ratio,
            win_rate=result.win_rate,
            total_return=result.total_return,
            initial_capital=result.initial_capital,
            final_value=result.final_value,
            period=result.period,
            stock_pool=strategy.stock_pool,
        )
        if ai_analysis:
            result.plain_summary += f"\n\n🤖 AI 深度解读:\n{'─' * 40}\n{ai_analysis}"

        logger.info(f"🔬 回测完成 | 评级: {result.grade} | "
                     f"年化: {result.annual_return:.1%} | "
                     f"夏普: {result.sharpe_ratio:.2f} | "
                     f"回撤: {result.max_drawdown:.1%}")

        return result

    def _enhance_plain_summary(self, result: BacktestResult, strategy: StrategyConfig) -> str:
        """增强回测报告的小白解读"""
        lines = []
        lines.append(f"🔬 回测验证报告")
        lines.append(f"{'─' * 40}")
        lines.append(f"策略名称: {strategy.name}")
        lines.append(f"回测区间: {result.period[0]} ~ {result.period[1]}")
        lines.append(f"初始资金: ${result.initial_capital:,.2f}")
        lines.append("")

        # 评级解读
        grade_labels = {
            StrategyGrade.A: ("⭐⭐⭐⭐⭐ 优秀", "这个策略表现非常好！"),
            StrategyGrade.B: ("⭐⭐⭐⭐ 良好", "策略表现不错，可以考虑使用。"),
            StrategyGrade.C: ("⭐⭐⭐ 一般", "策略表现一般，建议进一步优化。"),
            StrategyGrade.D: ("⭐⭐ 较差", "策略表现不理想，需要重新设计。"),
        }
        grade_label, grade_comment = grade_labels.get(
            result.grade, ("未知", "")
        )
        lines.append(f"📋 策略评级: {grade_label}")
        lines.append(f"   {grade_comment}")
        lines.append("")

        # 核心指标
        lines.append("📊 核心指标:")
        lines.append(f"  {'指标':<12} {'数值':>10}  通俗解读")
        lines.append(f"  {'─' * 45}")

        # 年化收益
        ar = result.annual_return
        ar_label = "跑赢银行存款" if ar > 0.03 else "不如存银行"
        if ar > 0.20:
            ar_label = "非常优秀"
        elif ar > 0.15:
            ar_label = "表现良好"
        elif ar > 0.08:
            ar_label = "还不错"
        lines.append(f"  {'年化收益':<12} {ar:>10.1%}  {ar_label}")

        # 最大回撤
        mdd = result.max_drawdown
        mdd_label = "很安全" if mdd < 0.05 else "可控" if mdd < 0.15 else "波动较大"
        lines.append(f"  {'最大回撤':<12} {mdd:>10.1%}  {mdd_label} (最惨时亏这么多)")

        # 夏普比率
        sr = result.sharpe_ratio
        sr_label = "优秀" if sr > 1.5 else "不错" if sr > 1.0 else "一般" if sr > 0.5 else "较差"
        lines.append(f"  {'夏普比率':<12} {sr:>10.2f}  {sr_label} (风险调整后收益)")

        # 胜率
        wr = result.win_rate
        wr_label = "胜率较高" if wr > 0.55 else "胜率一般" if wr > 0.45 else "胜率偏低"
        lines.append(f"  {'胜率':<12} {wr:>10.1%}  {wr_label}")

        # 总收益
        lines.append(f"  {'总收益':<12} {result.total_return:>10.1%}")
        lines.append(f"  {'最终资金':<12} ${result.final_value:>9,.2f}")

        lines.append("")

        # 建议
        if result.grade in [StrategyGrade.A, StrategyGrade.B]:
            lines.append("✅ 建议: 策略通过验证，可以进入模拟盘执行阶段。")
        elif result.grade == StrategyGrade.C:
            lines.append("⚠️ 建议: 策略表现一般，建议优化因子或调整参数后重新回测。")
        else:
            lines.append("❌ 建议: 策略表现较差，建议重新设计策略。")

        lines.append("")
        lines.append("⚠️ 风险提示: 回测基于历史数据，过去的表现不代表未来。")
        lines.append("   模拟盘验证满意后再考虑实盘，请谨慎对待。")

        return "\n".join(lines)

    def plain_explain(self, result: Any = None) -> str:
        """通俗解读"""
        if result is None:
            result = self._last_result
        if isinstance(result, BacktestResult):
            return result.plain_summary
        return "回测完成，请查看报告。"
