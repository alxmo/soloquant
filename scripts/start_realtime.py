"""
实时行情 WebSocket 启动脚本
=====================================================
Phase 7 T1: 独立启动实时行情数据流

用法:
  python scripts/start_realtime.py                    # 使用默认关注列表
  python scripts/start_realtime.py AAPL,MSFT,NVDA     # 指定股票
  python scripts/start_realtime.py --source alpaca    # 指定数据源
  python scripts/start_realtime.py --monitor           # 同时启动组合监控

按 Ctrl+C 停止。
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import config
from src.utils.logger import setup_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="启动实时行情 WebSocket 数据流",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/start_realtime.py                        # 默认关注列表
  python scripts/start_realtime.py AAPL,MSFT,NVDA,GOOGL   # 指定股票
  python scripts/start_realtime.py --source finnhub       # 仅用 Finnhub
  python scripts/start_realtime.py --monitor              # 同时监控持仓
  python scripts/start_realtime.py --alert 0.05            # 设置 5% 告警阈值
        """,
    )
    parser.add_argument(
        "symbols",
        nargs="?",
        default=None,
        help="订阅股票代码, 逗号分隔 (默认使用关注列表)",
    )
    parser.add_argument(
        "--source",
        choices=["alpaca", "finnhub", "both"],
        default=None,
        help="数据源 (默认读取 .env 配置)",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="同时启动组合监控 (止损止盈自动触发)",
    )
    parser.add_argument(
        "--alert",
        type=float,
        default=None,
        help=f"价格异动告警阈值 (默认 {config.REALTIME_PRICE_ALERT_THRESHOLD:.0%})",
    )
    return parser.parse_args()


def print_banner(symbols, source, monitor, alert_threshold):
    """打印启动横幅"""
    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  ⚡ 实时行情 WebSocket 数据流                               ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print(f"║  📋 订阅股票: {len(symbols)} 只")
    print(f"║  📡 数据源:   Alpaca {'✅' if config.has_alpaca_keys() else '❌'} | Finnhub {'✅' if config.has_finnhub_key() else '❌'}")
    print(f"║  🔧 模式:     {source}")
    print(f"║  🚨 告警阈值: {alert_threshold:.1%}")
    print(f"║  📊 组合监控: {'✅ 启用' if monitor else '❌ 禁用'}")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()
    print("  按 Ctrl+C 停止")
    print()


async def main():
    args = parse_args()
    setup_logger()

    # ─── 检查前提条件 ───

    if not config.REALTIME_ENABLED:
        print("❌ 实时行情已禁用 (REALTIME_ENABLED=false)")
        print("   请在 .env 中设置 REALTIME_ENABLED=true")
        return

    alpaca_ok = config.has_alpaca_keys()
    finnhub_ok = config.has_finnhub_key()

    if not alpaca_ok and not finnhub_ok:
        print("❌ 无可用数据源!")
        print("   请在 .env 中配置:")
        print("     ALPACA_API_KEY=xxx")
        print("     ALPACA_SECRET_KEY=xxx")
        print("   和/或")
        print("     FINNHUB_API_KEY=xxx")
        return

    # ─── 解析参数 ───

    # 股票列表
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = config.WATCHLIST or config.DEFAULT_WATCHLIST

    # 数据源
    source = args.source or config.REALTIME_DATA_SOURCE

    # 告警阈值
    alert_threshold = args.alert or config.REALTIME_PRICE_ALERT_THRESHOLD
    if args.alert:
        config.REALTIME_PRICE_ALERT_THRESHOLD = alert_threshold

    # 覆盖数据源
    if args.source:
        config.REALTIME_DATA_SOURCE = source

    print_banner(symbols, source, args.monitor, alert_threshold)

    # ─── 导入实时管理器 ───

    from src.data.realtime_manager import realtime_manager

    # ─── 注册回调 ───

    def on_price_update(quote):
        """价格更新回调 - 实时打印"""
        change_str = f"{quote.change_pct:+.2f}%" if quote.change_pct else "N/A"
        arrow = "🔺" if quote.change >= 0 else "🔻" if quote.change < 0 else "➡️"
        print(
            f"  {arrow} {datetime.now().strftime('%H:%M:%S')} | "
            f"{quote.symbol:5s} | ${quote.price:>10.2f} | {change_str:>8s} | {quote.source}"
        )

    def on_price_alert(alert):
        """价格异动告警回调"""
        print(f"  🚨🚨🚨 [{datetime.now().strftime('%H:%M:%S')}] {alert.message}")

    def on_bar_complete(bar):
        """1分钟K线完成回调"""
        print(
            f"  📊 K线完成 {datetime.now().strftime('%H:%M:%S')} | "
            f"{bar.symbol:5s} | O:{bar.open:.2f} H:{bar.high:.2f} "
            f"L:{bar.low:.2f} C:{bar.close:.2f} V:{bar.volume}"
        )

    realtime_manager.register_callback("on_price_update", on_price_update)
    realtime_manager.register_callback("on_price_alert", on_price_alert)
    realtime_manager.register_callback("on_bar_complete", on_bar_complete)

    # ─── 启动 ───

    await realtime_manager.start(symbols)

    if not realtime_manager.is_running:
        print("\n⚠️ 实时行情未能启动 (可能 API Key 无效或网络问题)")
        print("   降级提示: 系统可使用 REST 轮询模式获取行情 (realtime_fetcher)")
        return

    print("✅ 实时行情已启动! 等待数据流入...\n")

    # ─── 可选: 启动组合监控 ───

    if args.monitor:
        from src.execution.portfolio_monitor import portfolio_monitor

        try:
            await portfolio_monitor.start_realtime_monitoring(symbols)
            print("📊 组合监控已启动 (止损止盈自动触发)\n")
        except Exception as e:
            print(f"⚠️ 组合监控启动失败: {e}")

    # ─── 主循环: 定时打印状态 ───

    try:
        while realtime_manager.is_running:
            await asyncio.sleep(60)
            status = realtime_manager.get_status()
            print(f"\n{'─' * 60}")
            print(f"  📊 状态报告 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
            print(f"  缓存价格: {status['cached_prices']} 只 | 告警: {status['alerts_count']} 条")
            print(f"  Alpaca: {'✅' if status['alpaca_available'] else '❌'} | "
                  f"Finnhub: {'✅' if status['finnhub_available'] else '❌'}")

            # 打印所有缓存价格
            prices = realtime_manager.get_all_prices()
            if prices:
                print(f"  {'代码':5s} | {'价格':>12s} | {'涨跌幅':>8s} | {'来源':8s}")
                print(f"  {'─' * 50}")
                for sym, q in sorted(prices.items()):
                    chg = f"{q.change_pct:+.2f}%" if q.change_pct else "N/A"
                    print(f"  {sym:5s} | ${q.price:>10.2f} | {chg:>8s} | {q.source:8s}")
            print()

    except KeyboardInterrupt:
        print("\n\n⏹️ 正在停止实时行情...")
        await realtime_manager.stop()

        if args.monitor:
            try:
                await portfolio_monitor.stop_realtime_monitoring()
            except Exception:
                pass

        print("✅ 实时行情已停止。再见!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ 已中断。")
