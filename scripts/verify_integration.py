#!/usr/bin/env python3
"""
AI 全自动美股量化交易系统 - 集成可行性验证脚本
=====================================================
验证四大核心组件的集成可行性:
  1. AKShare  - 美股数据采集 (新浪源 stock_us_daily)
  2. Finnhub  - 金融数据 API (新闻/基本面/推荐)
  3. Alpaca   - 交易执行 API (Paper Trading)
  4. Qlib     - Microsoft 量化引擎 (模型/因子/回测)

用法:
  python scripts/verify_integration.py

注意:
  - 需要 .env 文件配置 FINNHUB_API_KEY / ALPACA_API_KEY / ALPACA_SECRET_KEY
  - AKShare 新浪源不需要 API Key
  - Qlib 需要先下载数据集 (scripts/dump_bin.py)
"""

import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# 工具函数
# ============================================================

def print_header(title: str):
    """打印阶段标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_step(num: int, desc: str):
    """打印步骤标题"""
    print(f"\n--- {num}. {desc} ---")

def print_ok(msg: str):
    """打印成功信息"""
    print(f"  ✅ {msg}")

def print_fail(msg: str):
    """打印失败信息"""
    print(f"  ❌ {msg}")

def print_info(msg: str):
    """打印普通信息"""
    print(f"  ℹ️  {msg}")

def run_test(test_name: str, test_func):
    """运行单个测试并捕获异常"""
    try:
        result = test_func()
        if result is False:
            print_fail(f"{test_name} - 测试返回 False")
            return False
        return True
    except Exception as e:
        print_fail(f"{test_name} - {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

# ============================================================
# Phase 1: 依赖包导入验证
# ============================================================

def test_imports():
    """测试所有核心依赖包是否可以导入"""
    print_header("Phase 1: 依赖包导入验证")
    
    results = {}
    
    def test_import(name, module_name, version_attr="__version__"):
        try:
            mod = __import__(module_name)
            ver = getattr(mod, version_attr, "unknown")
            print_ok(f"{name}: {ver}")
            results[name] = True
        except ImportError as e:
            print_fail(f"{name}: {e}")
            results[name] = False
    
    # 核心包
    test_import("akshare", "akshare")
    test_import("finnhub", "finnhub")
    test_import("alpaca-py", "alpaca")
    test_import("pyqlib", "qlib")
    test_import("pandas", "pandas")
    test_import("numpy", "numpy")
    test_import("lightgbm", "lightgbm")
    test_import("python-dotenv", "dotenv")
    
    passed = sum(results.values())
    total = len(results)
    print_info(f"\n导入验证: {passed}/{total} 通过")
    return all(results.values())

# ============================================================
# Phase 2: AKShare 美股数据采集验证
# ============================================================

def test_akshare():
    """验证 AKShare 美股数据采集能力"""
    print_header("Phase 2: AKShare 美股数据采集验证")
    
    import akshare as ak
    import pandas as pd
    
    results = []
    
    # --- 2.1 stock_us_daily (新浪源 - 主要数据获取方式) ---
    print_step("2.1", "stock_us_daily - 获取美股日线数据 (新浪源)")
    try:
        df = ak.stock_us_daily(symbol="AAPL", adjust="qfq")
        assert df is not None and len(df) > 0, "返回空 DataFrame"
        
        # 验证列名
        expected_cols = ["date", "open", "high", "low", "close", "volume"]
        for col in expected_cols:
            assert col in df.columns, f"缺少列: {col}"
        
        # 验证数据类型
        df["date"] = pd.to_datetime(df["date"])
        
        # 日期范围
        date_min = df["date"].min()
        date_max = df["date"].max()
        print_ok(f"AAPL: {len(df)} 条数据")
        print_info(f"日期范围: {date_min.date()} ~ {date_max.date()}")
        print_info(f"列名: {list(df.columns)}")
        print_info(f"最新收盘价: ${df.iloc[-1]['close']:.2f}")
        results.append(True)
    except Exception as e:
        print_fail(f"stock_us_daily 失败: {e}")
        results.append(False)
    
    # --- 2.2 多只股票验证 ---
    print_step("2.2", "多只美股数据获取验证")
    test_symbols = ["MSFT", "NVDA", "GOOGL", "AMZN", "TSLA"]
    success_count = 0
    for sym in test_symbols:
        try:
            df = ak.stock_us_daily(symbol=sym, adjust="qfq")
            if df is not None and len(df) > 0:
                print_ok(f"{sym}: {len(df)} 条数据")
                success_count += 1
            else:
                print_fail(f"{sym}: 返回空")
        except Exception as e:
            print_fail(f"{sym}: {type(e).__name__}: {str(e)[:60]}")
    print_info(f"成功率: {success_count}/{len(test_symbols)}")
    results.append(success_count >= 3)  # 至少3只成功
    
    # --- 2.3 日期筛选验证 ---
    print_step("2.3", "日期筛选能力验证")
    try:
        df = ak.stock_us_daily(symbol="AAPL", adjust="qfq")
        df["date"] = pd.to_datetime(df["date"])
        # 筛选最近1年
        one_year_ago = datetime.now() - timedelta(days=365)
        recent = df[df["date"] >= one_year_ago].copy()
        assert len(recent) > 200, f"筛选后数据过少: {len(recent)}"
        print_ok(f"近1年数据: {len(recent)} 条")
        print_info(f"日期范围: {recent['date'].min().date()} ~ {recent['date'].max().date()}")
        results.append(True)
    except Exception as e:
        print_fail(f"日期筛选失败: {e}")
        results.append(False)
    
    # --- 2.4 数据格式标准化验证 (转换为 Qlib 格式) ---
    print_step("2.4", "数据格式标准化 (Qlib 兼容格式)")
    try:
        df = ak.stock_us_daily(symbol="AAPL", adjust="qfq")
        df["date"] = pd.to_datetime(df["date"])
        
        # 标准化为 Qlib 格式
        qlib_df = pd.DataFrame({
            "date": df["date"],
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["volume"].astype(float),
            "factor": 1.0,  # 复权因子 (简化)
            "symbol": "AAPL",
        })
        qlib_df = qlib_df[["date", "open", "high", "low", "close", "volume", "factor", "symbol"]]
        
        assert len(qlib_df) > 0
        assert not qlib_df[["open", "high", "low", "close"]].isna().any().any(), "存在 NaN 值"
        
        print_ok(f"标准化成功: {len(qlib_df)} 条数据")
        print_info(f"列名: {list(qlib_df.columns)}")
        print_info(f"数据类型: {qlib_df.dtypes.to_dict()}")
        print_info(f"最新一行:\n{qlib_df.tail(1).to_string()}")
        results.append(True)
    except Exception as e:
        print_fail(f"数据标准化失败: {e}")
        results.append(False)
    
    passed = sum(results)
    total = len(results)
    print_info(f"\nAKShare 验证: {passed}/{total} 通过")
    return all(results)

# ============================================================
# Phase 3: Finnhub API 验证
# ============================================================

def test_finnhub():
    """验证 Finnhub API 金融数据获取能力"""
    print_header("Phase 3: Finnhub API 验证")
    
    # 加载 API Key
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        print_fail("FINNHUB_API_KEY 未配置，请检查 .env 文件")
        return False
    
    print_info(f"API Key: {'*'*(len(api_key)-4)}{api_key[-4:]}")
    
    import finnhub
    client = finnhub.Client(api_key=api_key)
    
    results = []
    
    # --- 3.1 公司信息 ---
    print_step("3.1", "公司信息 (company_profile2)")
    try:
        company = client.company_profile2(symbol="AAPL")
        assert company.get("name"), "公司名为空"
        print_ok(f"公司: {company.get('name')}")
        print_info(f"行业: {company.get('finnhubIndustry', 'N/A')}")
        print_info(f"交易所: {company.get('exchange', 'N/A')}")
        print_info(f"市值: ${company.get('marketCapitalization', 0)/1e9:.1f}B")
        results.append(True)
    except Exception as e:
        print_fail(f"公司信息失败: {e}")
        results.append(False)
    
    # --- 3.2 公司新闻 ---
    print_step("3.2", "公司新闻 (company_news)")
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        news = client.company_news("AAPL", week_ago, today)
        assert len(news) > 0, "无新闻数据"
        print_ok(f"获取到 {len(news)} 条 AAPL 新闻")
        print_info(f"最新: {news[0]['headline'][:80]}")
        print_info(f"来源: {news[0].get('source', 'N/A')}")
        print_info(f"新闻字段: {list(news[0].keys())}")
        results.append(True)
    except Exception as e:
        print_fail(f"公司新闻失败: {e}")
        results.append(False)
    
    # --- 3.3 新闻情绪分析 (可能需要付费) ---
    print_step("3.3", "新闻情绪分析 (news_sentiment)")
    try:
        sentiment = client.news_sentiment(symbol="AAPL")
        score = sentiment.get("sentiment", {}).get("score", "N/A")
        bull = sentiment.get("sentiment", {}).get("bullishPercent", "N/A")
        bear = sentiment.get("sentiment", {}).get("bearishPercent", "N/A")
        print_ok(f"情绪分数: {score}")
        print_info(f"看涨: {bull}, 看跌: {bear}")
        results.append(True)
    except Exception as e:
        print_fail(f"新闻情绪分析失败 (可能需要付费计划): {e}")
        print_info("替代方案: 使用 LLM 对 company_news 文本进行情绪分析")
        results.append(False)  # 标记为失败但不阻断
    
    # --- 3.4 基本面财务指标 ---
    print_step("3.4", "基本面财务指标 (company_basic_financials)")
    try:
        metrics = client.company_basic_financials(symbol="AAPL", metric="all")
        assert "metric" in metrics, "无指标数据"
        m = metrics["metric"]
        print_ok(f"获取到财务指标")
        print_info(f"P/E: {m.get('peNormalizedAnnual', 'N/A')}")
        print_info(f"ROE: {m.get('roeRfy', 'N/A')}")
        print_info(f"营收增长(5Y): {m.get('revenueGrowth5Y', 'N/A')}")
        print_info(f"52周最高: ${m.get('52WeekHigh', 'N/A')}")
        print_info(f"52周最低: ${m.get('52WeekLow', 'N/A')}")
        results.append(True)
    except Exception as e:
        print_fail(f"基本面指标失败: {e}")
        results.append(False)
    
    # --- 3.5 分析师推荐 ---
    print_step("3.5", "分析师推荐 (recommendation_trends)")
    try:
        recs = client.recommendation_trends(symbol="AAPL")
        assert len(recs) > 0, "无推荐数据"
        r = recs[0]
        print_ok(f"获取到 {len(recs)} 条推荐")
        print_info(f"买入: {r.get('buy', 0)}, 持有: {r.get('hold', 0)}, 卖出: {r.get('sell', 0)}")
        print_info(f"强烈买入: {r.get('strongBuy', 0)}, 强烈卖出: {r.get('strongSell', 0)}")
        results.append(True)
    except Exception as e:
        print_fail(f"分析师推荐失败: {e}")
        results.append(False)
    
    # --- 3.6 股价K线 (可能需要付费) ---
    print_step("3.6", "股价K线数据 (stock_candles)")
    try:
        now_ts = int(time.time())
        week_ts = now_ts - 7 * 24 * 3600
        candles = client.stock_candles("AAPL", "D", week_ts, now_ts)
        if candles.get("s") == "ok":
            n = len(candles.get("c", []))
            print_ok(f"获取到 {n} 条日线K线数据")
            print_info(f"最新收盘: ${candles['c'][-1]:.2f}")
            print_info(f"字段: {list(candles.keys())}")
            results.append(True)
        else:
            print_fail(f"K线数据不可用: {candles.get('s', 'N/A')}")
            print_info("替代方案: 使用 AKShare stock_us_daily 获取行情数据")
            results.append(False)
    except Exception as e:
        print_fail(f"股价K线失败 (可能需要付费计划): {e}")
        print_info("替代方案: 使用 AKShare stock_us_daily 获取行情数据")
        results.append(False)
    
    passed = sum(results)
    total = len(results)
    print_info(f"\nFinnhub 验证: {passed}/{total} 通过")
    # 核心3项(公司信息/新闻/基本面)通过即可
    core_passed = results[0] and results[1] and results[3]
    return core_passed

# ============================================================
# Phase 4: Alpaca 交易执行 API 验证
# ============================================================

def test_alpaca():
    """验证 Alpaca Paper Trading API"""
    print_header("Phase 4: Alpaca Paper Trading API 验证")
    
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    
    if not api_key or not secret_key or api_key.startswith("your_"):
        print_fail("ALPACA_API_KEY / ALPACA_SECRET_KEY 未配置")
        return False
    
    print_info(f"API Key: {'*'*(len(api_key)-4)}{api_key[-4:]}")
    
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest, MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce, QueryOrderStatus
    
    trade_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
    
    results = []
    
    # --- 4.1 账户信息 ---
    print_step("4.1", "账户信息 (get_account)")
    try:
        account = trade_client.get_account()
        print_ok(f"账户ID: {account.id}")
        print_info(f"状态: {account.status}")
        print_info(f"现金: ${float(account.cash):,.2f}")
        print_info(f"组合价值: ${float(account.portfolio_value):,.2f}")
        print_info(f"购买力: ${float(account.buying_power):,.2f}")
        results.append(True)
    except Exception as e:
        print_fail(f"账户信息失败: {e}")
        results.append(False)
    
    # --- 4.2 当前持仓 ---
    print_step("4.2", "当前持仓 (get_all_positions)")
    try:
        positions = trade_client.get_all_positions()
        print_ok(f"持仓数量: {len(positions)}")
        for p in positions[:5]:
            pnl_pct = float(p.unrealized_pl_pc) * 100
            print_info(f"  {p.symbol}: {p.qty} 股, 市值 ${float(p.market_value):,.2f}, "
                       f"盈亏 ${float(p.unrealized_pl):,.2f} ({pnl_pct:+.1f}%)")
        results.append(True)
    except Exception as e:
        print_fail(f"持仓查询失败: {e}")
        results.append(False)
    
    # --- 4.3 可交易资产 ---
    print_step("4.3", "可交易美股资产 (get_all_assets)")
    try:
        req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status="active")
        assets = trade_client.get_all_assets(req)
        tradable = [a for a in assets if a.tradable]
        print_ok(f"活跃可交易美股: {len(tradable)} 只")
        # 验证常见股票可用
        check_symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA"]
        for sym in check_symbols:
            found = [a for a in tradable if a.symbol == sym]
            if found:
                print_info(f"  {sym}: {found[0].name}")
        results.append(True)
    except Exception as e:
        print_fail(f"资产查询失败: {e}")
        results.append(False)
    
    # --- 4.4 模拟下单 (买入1股AAPL) ---
    print_step("4.4", "模拟下单 - 买入1股AAPL (Market Order)")
    try:
        order_req = MarketOrderRequest(
            symbol="AAPL",
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = trade_client.submit_order(order_req)
        print_ok(f"下单成功!")
        print_info(f"订单ID: {order.id}")
        print_info(f"股票: {order.symbol}, 方向: {order.side}, 数量: {order.qty}")
        print_info(f"状态: {order.status}, 类型: {order.order_type}")
        results.append(True)
    except Exception as e:
        print_fail(f"下单失败: {e}")
        results.append(False)
    
    # --- 4.5 查询订单 ---
    print_step("4.5", "查询订单 (get_orders)")
    try:
        time.sleep(2)  # 等待订单处理
        orders = trade_client.get_orders()
        print_ok(f"订单数量: {len(orders)}")
        for o in orders[:3]:
            print_info(f"  {o.symbol} {o.side} {o.qty} 状态={o.status}")
        results.append(True)
    except Exception as e:
        print_fail(f"订单查询失败: {e}")
        results.append(False)
    
    # --- 4.6 持仓更新验证 ---
    print_step("4.6", "持仓更新验证")
    try:
        time.sleep(1)
        positions = trade_client.get_all_positions()
        print_ok(f"持仓数量: {len(positions)}")
        for p in positions[:5]:
            print_info(f"  {p.symbol}: {p.qty} 股, 成本 ${float(p.avg_entry_price):,.2f}")
        results.append(True)
    except Exception as e:
        print_fail(f"持仓更新失败: {e}")
        results.append(False)
    
    passed = sum(results)
    total = len(results)
    print_info(f"\nAlpaca 验证: {passed}/{total} 通过")
    return all(results)

# ============================================================
# Phase 5: Qlib 量化引擎验证
# ============================================================

def test_qlib():
    """验证 Microsoft Qlib 量化引擎"""
    print_header("Phase 5: Qlib 量化引擎验证")
    
    import qlib
    results = []
    
    # --- 5.1 版本和初始化 ---
    print_step("5.1", "Qlib 版本与初始化")
    try:
        print_ok(f"Qlib 版本: {qlib.__version__}")
        # 尝试使用已下载的 CN 数据初始化
        from qlib.config import REG_CN
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
        print_ok("Qlib 初始化成功 (CN data)")
        results.append(True)
    except Exception as e:
        print_fail(f"Qlib 初始化失败: {e}")
        print_info("提示: 需要下载 Qlib 数据集，运行: python scripts/dump_bin.py")
        results.append(False)
    
    # --- 5.2 数据操作器 ---
    print_step("5.2", "数据操作器 (qlib.data.D)")
    try:
        from qlib.data import D
        print_ok(f"数据操作器 D 可用")
        print_info(f"D 方法: {[m for m in dir(D) if not m.startswith('_')][:10]}")
        results.append(True)
    except Exception as e:
        print_fail(f"数据操作器不可用: {e}")
        results.append(False)
    
    # --- 5.3 因子库 ---
    print_step("5.3", "因子库 (Alpha158 / Alpha360)")
    try:
        from qlib.contrib.data.handler import Alpha158, Alpha360
        print_ok("Alpha158 因子集可用")
        print_ok("Alpha360 因子集可用")
        results.append(True)
    except ImportError as e:
        print_fail(f"因子库不可用: {e}")
        results.append(False)
    
    # --- 5.4 预测模型 ---
    print_step("5.4", "预测模型 (LightGBM / Linear)")
    try:
        from qlib.contrib.model.gbdt import LGBModel
        print_ok("LightGBM 模型 (LGBModel) 可用")
        
        from qlib.contrib.model.linear import LinearModel
        print_ok("线性模型 (LinearModel) 可用")
        results.append(True)
    except ImportError as e:
        print_fail(f"模型不可用: {e}")
        results.append(False)
    
    # --- 5.5 策略模块 ---
    print_step("5.5", "策略模块 (TopkDropoutStrategy)")
    try:
        from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
        print_ok("TopkDropoutStrategy 可用")
        results.append(True)
    except ImportError as e:
        try:
            from qlib.contrib.strategy import TopkDropoutStrategy
            print_ok("TopkDropoutStrategy (alt path) 可用")
            results.append(True)
        except ImportError:
            print_fail(f"策略模块不可用: {e}")
            results.append(False)
    
    # --- 5.6 回测模块 ---
    print_step("5.6", "回测模块 (backtest_daily / risk_analysis)")
    try:
        from qlib.contrib.evaluate import backtest_daily, risk_analysis
        import inspect
        print_ok("backtest_daily 可用")
        print_ok("risk_analysis 可用")
        sig = inspect.signature(backtest_daily)
        print_info(f"backtest_daily 签名: {sig}")
        results.append(True)
    except ImportError as e:
        print_fail(f"回测模块不可用: {e}")
        results.append(False)
    
    # --- 5.7 工作流录制器 ---
    print_step("5.7", "工作流录制器 (qlib.workflow.R)")
    try:
        from qlib.workflow import R
        print_ok("R (Recorder) 可用")
        results.append(True)
    except ImportError as e:
        print_fail(f"R 不可用: {e}")
        results.append(False)
    
    passed = sum(results)
    total = len(results)
    print_info(f"\nQlib 验证: {passed}/{total} 通过")
    return all(results)

# ============================================================
# Phase 6: 全链路数据格式转换验证
# ============================================================

def test_data_pipeline():
    """验证 AKShare -> 标准化 -> Qlib 格式的数据管道"""
    print_header("Phase 6: 全链路数据格式转换验证")
    
    import akshare as ak
    import pandas as pd
    import numpy as np
    
    results = []
    
    # --- 6.1 AKShare 数据获取 ---
    print_step("6.1", "AKShare 获取 AAPL 原始数据")
    try:
        df_raw = ak.stock_us_daily(symbol="AAPL", adjust="qfq")
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        print_ok(f"原始数据: {len(df_raw)} 条")
        print_info(f"列名: {list(df_raw.columns)}")
        results.append(True)
    except Exception as e:
        print_fail(f"数据获取失败: {e}")
        results.append(False)
        return False
    
    # --- 6.2 数据清洗与标准化 ---
    print_step("6.2", "数据清洗与标准化")
    try:
        df = df_raw.copy()
        # 重命名列为 Qlib 标准格式
        df = df.rename(columns={
            "date": "date",
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "volume": "$volume",
        })
        # 添加复权因子
        df["$factor"] = 1.0
        # 添加 instrument 列
        df["instrument"] = "AAPL"
        # 确保数值类型
        for col in ["$open", "$high", "$low", "$close", "$volume", "$factor"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # 去除 NaN
        df = df.dropna(subset=["$open", "$high", "$low", "$close"])
        
        print_ok(f"清洗后: {len(df)} 条")
        print_info(f"列名: {list(df.columns)}")
        print_info(f"数据类型: {df.dtypes.to_dict()}")
        results.append(True)
    except Exception as e:
        print_fail(f"数据清洗失败: {e}")
        results.append(False)
    
    # --- 6.3 计算 Qlib 兼容因子 (示例) ---
    print_step("6.3", "计算示例因子 (Qlib Alpha158 兼容)")
    try:
        df = df.sort_values("date").reset_index(drop=True)
        
        # 计算常用因子
        df["$change"] = df["$close"].pct_change()
        df["$vol_ratio"] = df["$volume"] / df["$volume"].rolling(20).mean()
        df["$ma5"] = df["$close"].rolling(5).mean()
        df["$ma20"] = df["$close"].rolling(20).mean()
        df["$ma_ratio"] = df["$ma5"] / df["$ma20"]
        df["$rsi_14"] = _calc_rsi(df["$close"], 14)
        df["$return_5d"] = df["$close"].pct_change(5)
        df["$return_20d"] = df["$close"].pct_change(20)
        df["$volatility_20d"] = df["$change"].rolling(20).std()
        
        # 验证因子计算
        valid_count = df[["$change", "$vol_ratio", "$ma_ratio", "$rsi_14"]].notna().sum()
        print_ok("因子计算成功")
        print_info(f"有效数据量:\n{valid_count.to_string()}")
        print_info(f"最新因子值:\n{df[['date', '$close', '$change', '$rsi_14', '$ma_ratio']].tail(1).to_string()}")
        results.append(True)
    except Exception as e:
        print_fail(f"因子计算失败: {e}")
        results.append(False)
    
    # --- 6.4 验证 Qlib 数据格式兼容性 ---
    print_step("6.4", "Qlib 数据格式兼容性验证")
    try:
        # Qlib 需要的数据格式: MultiIndex(instrument, datetime), columns=$open/$high/$low/$close/$volume/$factor
        qlib_format = df[["instrument", "date", "$open", "$high", "$low", "$close", "$volume", "$factor"]].copy()
        qlib_format = qlib_format.set_index(["instrument", "date"])
        
        # 验证 MultiIndex
        assert isinstance(qlib_format.index, pd.MultiIndex), "不是 MultiIndex"
        assert qlib_format.index.names == ["instrument", "date"], f"索引名错误: {qlib_format.index.names}"
        
        # 验证列名以 $ 开头 (Qlib 约定)
        for col in ["$open", "$high", "$low", "$close", "$volume", "$factor"]:
            assert col in qlib_format.columns, f"缺少列: {col}"
        
        print_ok("Qlib 格式验证通过")
        print_info(f"索引类型: {type(qlib_format.index).__name__}")
        print_info(f"索引名: {qlib_format.index.names}")
        print_info(f"列名: {list(qlib_format.columns)}")
        print_info(f"数据形状: {qlib_format.shape}")
        results.append(True)
    except Exception as e:
        print_fail(f"格式验证失败: {e}")
        results.append(False)
    
    # --- 6.5 模拟因子数据用于模型训练 ---
    print_step("6.5", "模拟因子数据用于模型训练")
    try:
        # 准备特征和标签
        feature_cols = ["$change", "$vol_ratio", "$ma_ratio", "$rsi_14", "$return_5d", "$return_20d", "$volatility_20d"]
        features = df[feature_cols].dropna()
        
        # 标签: 未来5日收益率
        df["$label"] = df["$close"].shift(-5) / df["$close"] - 1
        labels = df["$label"].dropna()
        
        # 对齐
        common_idx = features.index.intersection(labels.index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        
        print_ok(f"特征矩阵: {X.shape}, 标签: {y.shape}")
        print_info(f"特征列: {list(X.columns)}")
        print_info(f"标签统计: mean={y.mean():.4f}, std={y.std():.4f}")
        
        # 快速训练 LightGBM 模型
        import lightgbm as lgb
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        model = lgb.LGBMRegressor(n_estimators=50, max_depth=5, verbose=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        # 简单评估
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_test, pred)
        print_ok(f"LightGBM 训练成功")
        print_info(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
        print_info(f"MSE: {mse:.6f}")
        print_info(f"特征重要性: {dict(zip(X.columns, model.feature_importances_))}")
        results.append(True)
    except Exception as e:
        print_fail(f"模型训练失败: {e}")
        results.append(False)
    
    passed = sum(results)
    total = len(results)
    print_info(f"\n数据管道验证: {passed}/{total} 通过")
    return all(results)

def _calc_rsi(prices, period=14):
    """计算 RSI 指标"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ============================================================
# 主函数
# ============================================================

def main():
    """主函数 - 运行所有验证测试"""
    print("=" * 60)
    print("  AI 全自动美股量化交易系统 - 集成可行性验证")
    print("  验证时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # 检查 .env 文件
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print_fail(".env 文件不存在，请先复制 .env.example 为 .env 并配置 API Key")
        return False
    
    # 运行所有测试
    all_results = {}
    
    # Phase 1: 依赖包导入
    all_results["Phase1_依赖导入"] = run_test("依赖导入", test_imports)
    
    # Phase 2: AKShare
    all_results["Phase2_AKShare"] = run_test("AKShare", test_akshare)
    
    # Phase 3: Finnhub
    all_results["Phase3_Finnhub"] = run_test("Finnhub", test_finnhub)
    
    # Phase 4: Alpaca
    all_results["Phase4_Alpaca"] = run_test("Alpaca", test_alpaca)
    
    # Phase 5: Qlib
    all_results["Phase5_Qlib"] = run_test("Qlib", test_qlib)
    
    # Phase 6: 数据管道
    all_results["Phase6_数据管道"] = run_test("数据管道", test_data_pipeline)
    
    # ============================================================
    # 汇总报告
    # ============================================================
    print("\n" + "=" * 60)
    print("  📊 验证结果汇总")
    print("=" * 60)
    
    passed = sum(1 for v in all_results.values() if v)
    total = len(all_results)
    
    for name, result in all_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n  总计: {passed}/{total} 阶段通过")
    
    if passed == total:
        print("\n  🎉 所有验证通过！系统可以开始开发了。")
    else:
        print("\n  ⚠️  部分验证失败，请检查上方详细输出。")
        print("  常见问题:")
        print("  - AKShare SSL错误: 使用 stock_us_daily (新浪源) 而非 stock_us_hist (东方财富源)")
        print("  - Finnhub 403错误: 免费计划不支持 stock_candles/news_sentiment, 用 AKShare 替代行情")
        print("  - Alpaca 连接失败: 检查 API Key 是否为 Paper Trading 环境")
        print("  - Qlib 数据缺失: 需要下载美股数据集到 ~/.qlib/qlib_data/")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
