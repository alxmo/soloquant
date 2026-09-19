"""
本地量化研究环境一键配置脚本
=====================================================
功能:
  1. 扩充 Qlib 数据集 (50只美股主流股票 + 3年数据)
  2. 验证数据完整性
  3. 计算 Alpha158 因子并缓存
  4. 生成环境报告

用法:
  python scripts/local_setup.py          # 完整初始化
  python scripts/local_setup.py --check  # 只检查环境
  python scripts/local_setup.py --update # 增量更新数据
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.utils.config import config


# ═══════════════════════════════════════════════════════
# 本地研究股票池 (50只主流美股)
# ═══════════════════════════════════════════════════════
LOCAL_STOCK_POOL = [
    # 科技巨头
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
    "NFLX", "INTC", "ORCL", "ADBE", "CRM", "PYPL", "UBER", "COIN",
    # 金融
    "JPM", "BAC", "GS", "MS", "V", "MA", "AXP", "BLK",
    # 消费
    "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "TGT", "LOW",
    # 医疗
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY",
    # 能源/工业
    "XOM", "CVX", "BA", "CAT", "GE", "DIS", "VZ", "T",
    # 其他热门
    "BABA", "BIDU", "JD",
]

# 数据年限
DATA_YEARS = 3


def check_environment() -> dict:
    """检查本地环境状态"""
    logger.info("=" * 60)
    logger.info("🔍 本地环境检查")
    logger.info("=" * 60)

    results = {}

    # 1. Python 版本
    results["python_version"] = sys.version
    logger.info(f"🐍 Python: {sys.version.split()[0]}")

    # 2. 核心依赖
    packages = [
        "pandas", "numpy", "lightgbm", "qlib", "sklearn",
        "streamlit", "torch", "akshare", "finnhub", "alpaca",
    ]
    pkg_status = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "ok")
            pkg_status[pkg] = ver
            logger.info(f"  ✅ {pkg}: {ver}")
        except ImportError:
            pkg_status[pkg] = "missing"
            logger.info(f"  ❌ {pkg}: 未安装")
    results["packages"] = pkg_status

    # 3. Qlib 数据
    qlib_path = config.QLIB_DATA_PATH
    results["qlib_path"] = str(qlib_path)

    if qlib_path.exists():
        features_dir = qlib_path / "features"
        if features_dir.exists():
            stocks = [d for d in features_dir.iterdir() if d.is_dir()]
            results["qlib_stocks"] = len(stocks)
            logger.info(f"📊 Qlib 数据: {len(stocks)} 只股票")
        else:
            results["qlib_stocks"] = 0
            logger.info("📊 Qlib 数据: 无")
    else:
        results["qlib_stocks"] = 0
        logger.info("📊 Qlib 数据: 目录不存在")

    # 4. 模型和策略
    models_dir = config.DATA_DIR / "models"
    strategies_dir = config.DATA_DIR / "strategies"
    results["model_count"] = len(list(models_dir.glob("*.txt"))) if models_dir.exists() else 0
    results["strategy_count"] = len(list(strategies_dir.glob("*.json"))) if strategies_dir.exists() else 0
    logger.info(f"🤖 已训练模型: {results['model_count']} 个")
    logger.info(f"📈 策略配置: {results['strategy_count']} 个")

    # 5. API 配置
    results["alpaca_configured"] = config.has_alpaca_keys()
    results["finnhub_configured"] = config.has_finnhub_key()
    logger.info(f"🔑 Alpaca API: {'✅ 已配置' if config.has_alpaca_keys() else '❌ 未配置'}")
    logger.info(f"🔑 Finnhub API: {'✅ 已配置' if config.has_finnhub_key() else '❌ 未配置'}")

    return results


def init_qlib_data() -> dict:
    """初始化 Qlib 数据 (全量)"""
    logger.info("=" * 60)
    logger.info("📥 初始化 Qlib 数据集")
    logger.info("=" * 60)

    from src.data.qlib_adapter import qlib_adapter

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * DATA_YEARS)).strftime("%Y-%m-%d")

    logger.info(f"股票池: {len(LOCAL_STOCK_POOL)} 只")
    logger.info(f"时间范围: {start_date} ~ {end_date}")
    logger.info("-" * 60)

    result = qlib_adapter.init_qlib_data(
        symbols=LOCAL_STOCK_POOL,
        start=start_date,
        end=end_date,
    )

    logger.info("-" * 60)
    logger.info(f"✅ 成功: {result['success']} 只")
    if result["failed"]:
        logger.warning(f"❌ 失败: {len(result['failed'])} 只 - {result['failed']}")
    logger.info(f"📊 总数据行数: {result['total_rows']:,}")

    return result


def update_qlib_data() -> dict:
    """增量更新 Qlib 数据"""
    logger.info("=" * 60)
    logger.info("🔄 增量更新 Qlib 数据")
    logger.info("=" * 60)

    from src.data.qlib_adapter import qlib_adapter

    # 读取现有股票列表
    instruments_file = config.QLIB_DATA_PATH / "instruments" / "all.txt"
    if not instruments_file.exists():
        logger.warning("现有数据不存在，执行全量初始化")
        return init_qlib_data()

    existing_stocks = []
    with open(instruments_file) as f:
        for line in f:
            if line.strip():
                symbol = line.strip().split("\t")[0]
                existing_stocks.append(symbol)

    logger.info(f"现有股票: {len(existing_stocks)} 只")

    # 合并股票池
    all_stocks = list(set(existing_stocks + LOCAL_STOCK_POOL))
    logger.info(f"更新后股票池: {len(all_stocks)} 只")

    # 重新初始化 (简化版，后续可优化为真正增量)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * DATA_YEARS)).strftime("%Y-%m-%d")

    result = qlib_adapter.init_qlib_data(
        symbols=all_stocks,
        start=start_date,
        end=end_date,
    )

    return result


def verify_data() -> dict:
    """验证数据完整性"""
    logger.info("=" * 60)
    logger.info("✅ 数据完整性验证")
    logger.info("=" * 60)

    from src.data.qlib_adapter import qlib_adapter

    checks = qlib_adapter.verify_qlib_data()

    all_pass = True
    for name, passed in checks.items():
        status = "✅" if passed else "❌"
        logger.info(f"  {status} {name}")
        if not passed:
            all_pass = False

    if all_pass:
        logger.info("🎉 数据验证全部通过!")
    else:
        logger.warning("⚠️ 部分验证未通过")

    return checks


def calc_factors() -> dict:
    """预计算 Alpha158 因子并缓存"""
    logger.info("=" * 60)
    logger.info("🧮 预计算 Alpha158 因子")
    logger.info("=" * 60)

    from src.engine.qlib_wrapper import qlib_wrapper

    if not qlib_wrapper.init():
        logger.error("Qlib 初始化失败，无法计算因子")
        return {"success": False}

    # 获取股票列表
    stocks = qlib_wrapper.get_instruments()
    if not stocks:
        logger.error("没有可用的股票数据")
        return {"success": False}

    logger.info(f"股票数量: {len(stocks)}")

    # 计算日期范围
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * DATA_YEARS)).strftime("%Y-%m-%d")

    logger.info(f"日期范围: {start_date} ~ {end_date}")
    logger.info("计算中... (这可能需要几分钟到十几分钟)")

    start_time = time.time()
    factor_df = qlib_wrapper.calc_alpha158_factors(stocks, start_date, end_date)
    elapsed = time.time() - start_time

    if factor_df.empty:
        logger.error("因子计算失败")
        return {"success": False}

    logger.info(f"✅ 因子计算完成! 耗时: {elapsed:.1f}秒")
    logger.info(f"  形状: {factor_df.shape}")
    logger.info(f"  因子数: {len(factor_df.columns)}")

    # 缓存因子数据
    cache_path = config.CACHE_DIR / "alpha158_factors.parquet"
    factor_df.to_parquet(cache_path)
    logger.info(f"  已缓存到: {cache_path}")

    return {
        "success": True,
        "shape": factor_df.shape,
        "elapsed": elapsed,
        "cache_path": str(cache_path),
    }


def generate_report(env_info: dict):
    """生成环境报告"""
    logger.info("=" * 60)
    logger.info("📋 本地环境配置报告")
    logger.info("=" * 60)

    report = f"""
╔══════════════════════════════════════════════════════════╗
║           AI 量化交易系统 - 本地环境报告                  ║
╠══════════════════════════════════════════════════════════╣
║ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
║ 项目路径: {PROJECT_ROOT}
║
║ ─── 系统信息 ───
║ Python: {env_info.get('python_version', 'unknown').split()[0]}
║
║ ─── 数据状态 ───
║ Qlib 股票数: {env_info.get('qlib_stocks', 0)}
║ 已训练模型: {env_info.get('model_count', 0)} 个
║ 策略配置: {env_info.get('strategy_count', 0)} 个
║
║ ─── API 状态 ───
║ Alpaca: {'已配置' if env_info.get('alpaca_configured') else '未配置'}
║ Finnhub: {'已配置' if env_info.get('finnhub_configured') else '未配置'}
║
╚══════════════════════════════════════════════════════════╝
"""
    print(report)

    # 保存报告
    report_path = config.REPORTS_DIR / f"env_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="本地量化研究环境配置")
    parser.add_argument("--check", action="store_true", help="只检查环境状态")
    parser.add_argument("--update", action="store_true", help="增量更新数据")
    parser.add_argument("--factors", action="store_true", help="预计算因子")
    parser.add_argument("--full", action="store_true", help="完整初始化 (数据+因子)")
    args = parser.parse_args()

    # 默认行为: 检查环境
    if not any([args.check, args.update, args.factors, args.full]):
        args.check = True

    # 1. 检查环境
    env_info = check_environment()

    if args.check:
        generate_report(env_info)
        return

    # 2. 数据初始化/更新
    if args.full or args.update:
        if args.update:
            update_qlib_data()
        else:
            init_qlib_data()
        verify_data()

    # 3. 因子计算
    if args.full or args.factors:
        calc_factors()

    # 4. 最终报告
    env_info = check_environment()
    generate_report(env_info)

    logger.info("🎉 本地环境配置完成!")
    logger.info("💡 下一步: 运行 python main.py 进入系统")


if __name__ == "__main__":
    main()
