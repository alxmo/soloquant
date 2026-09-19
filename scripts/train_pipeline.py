"""
本地训练工作流 - 一键完成数据更新→模型训练→回测验证→报告生成
=====================================================
功能:
  1. 数据检查与更新
  2. 因子计算 (Alpha158)
  3. 多模型训练 (LightGBM / Linear / LSTM)
  4. 自动回测验证
  5. 生成对比报告
  6. 标记最优模型，准备同步到云端

用法:
  python scripts/train_pipeline.py                # 默认训练流程
  python scripts/train_pipeline.py --quick        # 快速模式 (少股票+短周期)
  python scripts/train_pipeline.py --lstm         # 包含 LSTM 模型
  python scripts/train_pipeline.py --topk 10      # 持仓 Top-K
  python scripts/train_pipeline.py --years 2      # 训练数据年限
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.models import StrategyConfig, StrategyGrade
from src.utils.config import config


# ─── 默认配置 ───
DEFAULT_STOCK_POOL = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
    "NFLX", "INTC", "ORCL", "ADBE", "CRM", "PYPL", "UBER", "JPM",
    "V", "DIS", "BABA", "COIN",
]

QUICK_STOCK_POOL = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD",
]


def init_qlib():
    """初始化 Qlib"""
    from src.engine.qlib_wrapper import qlib_wrapper

    if not qlib_wrapper.init():
        logger.error("Qlib 初始化失败，请先运行数据初始化")
        return False
    return True


def get_stock_pool(quick: bool = False) -> list[str]:
    """获取股票池"""
    from src.engine.qlib_wrapper import qlib_wrapper

    available = qlib_wrapper.get_instruments()
    target = QUICK_STOCK_POOL if quick else DEFAULT_STOCK_POOL

    # 取交集
    stocks = [s for s in target if s in available]
    logger.info(f"可用股票: {len(stocks)}/{len(target)} 只")
    return stocks


def train_lightgbm_model(stocks: list[str], train_years: int = 3, top_k: int = 5) -> dict:
    """训练 LightGBM 模型"""
    logger.info("=" * 60)
    logger.info("🤖 训练 LightGBM 模型")
    logger.info("=" * 60)

    from src.engine.model_manager import ModelManager
    from src.engine.factor_search import FactorSearcher

    end_date = datetime.now().strftime("%Y-%m-%d")
    train_start = (datetime.now() - timedelta(days=365 * train_years)).strftime("%Y-%m-%d")
    val_end = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    val_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    logger.info(f"训练集: {train_start} ~ {val_end}")
    logger.info(f"验证集: {val_start} ~ {end_date}")
    logger.info(f"股票池: {len(stocks)} 只")

    # 1. 因子搜索
    logger.info("步骤 1/3: 因子搜索...")
    searcher = FactorSearcher()
    factors = searcher.search(
        stock_pool=stocks,
        start=train_start,
        end=val_end,
        top_n=20,
    )
    factor_names = [f.name for f in factors]
    logger.info(f"  筛选出 {len(factor_names)} 个有效因子")

    # 2. 训练模型
    logger.info("步骤 2/3: 训练 LightGBM 模型...")
    manager = ModelManager()

    # 调整线程数 (本地高配可以开满)
    import multiprocessing
    num_threads = max(2, multiprocessing.cpu_count() - 1)
    logger.info(f"  使用线程数: {num_threads}")

    result = manager.train_model(
        model_type="lightgbm",
        factors=factor_names,
        stock_pool=stocks,
        train_start=train_start,
        train_end=val_end,
        params={"num_threads": num_threads},
    )

    # 3. 回测验证
    logger.info("步骤 3/3: 回测验证...")
    strategy = StrategyConfig(
        strategy_id=f"local_lightgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        name="本地训练-LightGBM",
        stock_pool=stocks,
        factors=factor_names,
        model_type="lightgbm",
        top_k=top_k,
    )

    from src.engine.backtest_engine import backtest_engine
    bt_result = backtest_engine.run_backtest(
        strategy, val_start, end_date, initial_capital=100000
    )

    # 保存策略配置
    strategy_path = config.DATA_DIR / "strategies" / f"{strategy.strategy_id}.json"
    with open(strategy_path, "w") as f:
        json.dump(strategy.model_dump(), f, indent=2, default=str)

    # 保存回测结果
    bt_path = config.DATA_DIR / "strategies" / f"{strategy.strategy_id}_backtest.json"
    with open(bt_path, "w") as f:
        json.dump(bt_result.model_dump(), f, indent=2, default=str)

    logger.info(f"✅ LightGBM 训练完成!")
    logger.info(f"  年化收益: {bt_result.annual_return:.2%}")
    logger.info(f"  夏普比率: {bt_result.sharpe_ratio:.2f}")
    logger.info(f"  最大回撤: {bt_result.max_drawdown:.2%}")
    logger.info(f"  策略评级: {bt_result.grade}")
    logger.info(f"  策略文件: {strategy_path.name}")

    return {
        "model_type": "lightgbm",
        "strategy_id": strategy.strategy_id,
        "factors": factor_names,
        "annual_return": bt_result.annual_return,
        "sharpe_ratio": bt_result.sharpe_ratio,
        "max_drawdown": bt_result.max_drawdown,
        "grade": bt_result.grade,
        "strategy_path": str(strategy_path),
        "model_path": result.model_path if hasattr(result, 'model_path') else "",
    }


def train_lstm_model(stocks: list[str], train_years: int = 2, top_k: int = 5) -> dict:
    """训练 LSTM 模型"""
    logger.info("=" * 60)
    logger.info("🧠 训练 LSTM 深度学习模型")
    logger.info("=" * 60)

    try:
        import torch
    except ImportError:
        logger.warning("PyTorch 未安装，跳过 LSTM 训练")
        return {}

    end_date = datetime.now().strftime("%Y-%m-%d")
    train_start = (datetime.now() - timedelta(days=365 * train_years)).strftime("%Y-%m-%d")
    val_end = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    val_start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

    logger.info(f"训练集: {train_start} ~ {val_end}")
    logger.info(f"验证集: {val_start} ~ {end_date}")

    # 使用较少的因子加速训练
    from src.engine.factor_search import FactorSearcher
    searcher = FactorSearcher()
    factors = searcher.search(
        stock_pool=stocks,
        start=train_start,
        end=val_end,
        top_n=10,  # LSTM 用少一点因子
    )
    factor_names = [f.name for f in factors]
    logger.info(f"使用因子: {len(factor_names)} 个")

    # 训练 LSTM
    from src.engine.model_manager import ModelManager
    manager = ModelManager()

    result = manager.train_model(
        model_type="lstm",
        factors=factor_names,
        stock_pool=stocks,
        train_start=train_start,
        train_end=val_end,
        params={"n_epochs": 30, "hidden_size": 64},
    )

    # 回测
    strategy = StrategyConfig(
        strategy_id=f"local_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        name="本地训练-LSTM",
        stock_pool=stocks,
        factors=factor_names,
        model_type="lstm",
        top_k=top_k,
    )

    from src.engine.backtest_engine import backtest_engine
    bt_result = backtest_engine.run_backtest(
        strategy, val_start, end_date, initial_capital=100000
    )

    # 保存
    strategy_path = config.DATA_DIR / "strategies" / f"{strategy.strategy_id}.json"
    with open(strategy_path, "w") as f:
        json.dump(strategy.model_dump(), f, indent=2, default=str)

    bt_path = config.DATA_DIR / "strategies" / f"{strategy.strategy_id}_backtest.json"
    with open(bt_path, "w") as f:
        json.dump(bt_result.model_dump(), f, indent=2, default=str)

    logger.info(f"✅ LSTM 训练完成!")
    logger.info(f"  年化收益: {bt_result.annual_return:.2%}")
    logger.info(f"  夏普比率: {bt_result.sharpe_ratio:.2f}")
    logger.info(f"  最大回撤: {bt_result.max_drawdown:.2%}")

    return {
        "model_type": "lstm",
        "strategy_id": strategy.strategy_id,
        "factors": factor_names,
        "annual_return": bt_result.annual_return,
        "sharpe_ratio": bt_result.sharpe_ratio,
        "max_drawdown": bt_result.max_drawdown,
        "grade": bt_result.grade,
        "strategy_path": str(strategy_path),
    }


def generate_report(results: list[dict], elapsed: float):
    """生成训练报告"""
    logger.info("=" * 60)
    logger.info("📋 训练报告")
    logger.info("=" * 60)

    if not results:
        logger.warning("没有训练结果")
        return

    # 按夏普比率排序
    results.sort(key=lambda x: x.get("sharpe_ratio", 0), reverse=True)

    print(f"\n{'模型类型':<12} {'年化收益':<10} {'夏普比率':<10} {'最大回撤':<10} {'评级':<6}")
    print("-" * 55)
    for r in results:
        print(f"{r['model_type']:<12} {r['annual_return']:>8.2%}  {r['sharpe_ratio']:>8.2f}  {r['max_drawdown']:>8.2%}  {r['grade']:<6}")

    best = results[0]
    print(f"\n🏆 最优模型: {best['model_type']} (夏普 {best['sharpe_ratio']:.2f})")
    print(f"   策略ID: {best['strategy_id']}")

    # 保存报告
    report = {
        "generated_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "total_models": len(results),
        "best_model": best,
        "all_results": results,
    }

    report_path = config.REPORTS_DIR / f"train_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"报告已保存: {report_path}")

    # 标记最优模型 (用于同步到云端)
    best_marker = config.DATA_DIR / "models" / "best_model.json"
    with open(best_marker, "w") as f:
        json.dump(best, f, indent=2, default=str)
    logger.info(f"最优模型已标记: {best_marker}")


def main():
    parser = argparse.ArgumentParser(description="本地训练工作流")
    parser.add_argument("--quick", action="store_true", help="快速模式 (少股票)")
    parser.add_argument("--lstm", action="store_true", help="包含 LSTM 模型")
    parser.add_argument("--topk", type=int, default=5, help="持仓 Top-K")
    parser.add_argument("--years", type=int, default=3, help="训练数据年限")
    args = parser.parse_args()

    start_time = time.time()

    logger.info("🚀 本地训练工作流启动")
    logger.info(f"模式: {'快速' if args.quick else '完整'} | Top-K: {args.topk} | 年限: {args.years}年")

    # 1. 初始化 Qlib
    if not init_qlib():
        return

    # 2. 获取股票池
    stocks = get_stock_pool(quick=args.quick)
    if len(stocks) < 5:
        logger.error(f"可用股票太少 ({len(stocks)}只)，请先扩充数据集")
        return

    results = []

    # 3. 训练 LightGBM
    try:
        result = train_lightgbm_model(stocks, train_years=args.years, top_k=args.topk)
        results.append(result)
    except Exception as e:
        logger.error(f"LightGBM 训练失败: {e}")
        import traceback
        traceback.print_exc()

    # 4. 训练 LSTM (可选)
    if args.lstm:
        try:
            result = train_lstm_model(stocks, train_years=min(args.years, 2), top_k=args.topk)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"LSTM 训练失败: {e}")
            import traceback
            traceback.print_exc()

    # 5. 生成报告
    elapsed = time.time() - start_time
    generate_report(results, elapsed)

    logger.info(f"🎉 训练完成! 总耗时: {elapsed/60:.1f} 分钟")
    logger.info(f"💡 下一步: 运行 scripts/sync_to_cloud.py 同步到云端")


if __name__ == "__main__":
    main()
