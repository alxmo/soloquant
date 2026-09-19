"""
交互式 CLI 界面 - 小白友好的命令行对话
=====================================================
提供两种交互模式:
1. 自然语言对话 (默认) - 直接输入大白话
2. 菜单模式 - 输入数字选择功能

v0.3.0 更新:
- 新增 dashboard 命令: 启动 Streamlit 仪表盘
- 新增 scheduler 命令: 启动定时任务调度器
- 版本号统一为 v0.3.0

v0.2.1 修复:
- P6: chat_mode 中 nl_router.respond() 增加 try/except，防止异常中断对话
- P8: 版本号统一为 v0.2.1
- P4: _menu_strategy 修复 ResearchReport 对象/dict 兼容性
- 补充 health/demo/init 命令行参数支持
"""

import sys
from datetime import datetime

# i18n: 翻译函数
from src.ui.i18n import t

# 必须先导入 logger 并设置 handler，再导入其他模块
# 否则 config 加载时的日志会使用 loguru 默认格式（带代码路径）
from src.utils.logger import setup_logger
from loguru import logger

from src.agents.registry import agent_registry
from src.ui.nl_router import nl_router
from src.ui.plain_explainer import plain_explainer
from src.utils.config import config
from src.workflow.engine import workflow_engine


def print_banner():
    """打印系统横幅"""
    print(t("\n═══════════════════════════════════════════════════════════\n     🤖 AI 全自动美股量化交易系统 v0.4.0                    \n     ─────────────────────────────────────────────          \n     📊 数据: AKShare + Finnhub                             \n     🧠 引擎: Microsoft Qlib + LightGBM                     \n     🤖 Agent: 调研->策略->回测->执行->监控                     \n     💰 交易: Alpaca Paper Trading                          \n     🛡️  风控: 7 条规则全检                                 \n     📈 仪表盘: Streamlit 可视化                             \n     ⏰ 调度器: 定时自动运行                                  \n     🧠 LLM: 智能对话 (多模型支持)                            \n═══════════════════════════════════════════════════════════\n    "))


def print_status():
    """打印系统状态"""
    print(f"{'─' * 60}")
    print(t('  📋 系统配置:'))
    print(t('     交易模式: {v1}', v1='📝 模拟盘 (Paper)' if config.is_paper_trading() else '🔴 实盘 (Live)'))
    print(f"     Alpaca:   {'✅ 已配置' if config.has_alpaca_keys() else '❌ 未配置'}")
    print(f"     Finnhub:  {'✅ 已配置' if config.has_finnhub_key() else '❌ 未配置'}")
    print(t('     最大仓位: {v1:.0%} | 日最大亏损: {v2:.0%} | 最大回撤: {v3:.0%}', v1=config.MAX_POSITION_PCT, v2=config.MAX_DAILY_LOSS, v3=config.MAX_DRAWDOWN))

    # v0.4.0: LLM 状态
    try:
        from src.llm.llm_config import llm_config
        llm_config.load()
        llm_st = llm_config.status()
        if llm_st["enabled"]:
            print(t('     LLM: 🧠 智能模式 (模型: {v1})', v1=llm_st['default_model']))
        elif llm_st["total_models"] > 0:
            avail = llm_st["available_models"]
            if avail:
                print(t('     LLM: 🧠 智能模式 (模型: {v1})', v1=', '.join(avail)))
            else:
                print(t('     LLM: 📝 规则模式 (已配置 {v1} 个模型但Key未填)', v1=llm_st['total_models']))
        else:
            print(t('     LLM: 📝 规则模式 (未配置)'))
    except Exception:
        print(t('     LLM: 📝 规则模式'))

    print(f"{'─' * 60}")


def chat_mode():
    """自然语言对话模式"""
    # v0.4.0: 检测 LLM 状态并提示
    try:
        from src.llm.llm_config import llm_config
        llm_config.load()
        if llm_config.is_enabled():
            print(t('\n🧠 智能对话模式已启用 (LLM: {v1})', v1=llm_config.get_default_name()))
            print(t("   支持自然语言理解、多轮对话上下文记忆。"))
        else:
            print(t("\n📝 规则对话模式 (未配置 LLM API Key)"))
            print(t("   在 .env 中配置 LLM_{NAME}_* 环境变量可启用智能对话。"))
    except Exception:
        pass

    print("\n" + plain_explainer.friendly_greeting())

    while True:
        try:
            user_input = input(t("\n💬 你: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(t("\n\n👋 再见！祝投资顺利！"))
            break

        if not user_input:
            continue

        # 退出命令
        if user_input.lower() in ("quit", "exit", "q", "退出", "再见"):
            print(t("👋 再见！祝投资顺利！"))
            break

        # P6 修复: 处理输入时增加 try/except，防止异常中断对话
        try:
            response = nl_router.respond(user_input)
            print(f"\n🤖 {response}")
        except Exception as e:
            logger.error(t('对话处理异常: {v1}', v1=e))
            print(t('\n🤖 抱歉，处理你的请求时出现了错误: {v1}', v1=e))
            print(t("   请稍后重试，或输入 \"帮助\" 查看可用功能。"))


def menu_mode():
    """菜单模式"""
    while True:
        print(f"\n{'═' * 50}")
        print(t("  📋 功能菜单"))
        print(f"{'─' * 50}")
        print(t("  🎯  AI 自主交易 - 告诉我目标和资金，全自动搞定"))
        print(t("  1️⃣  市场调研 - 看看今天有什么机会"))
        print(t("  2️⃣  策略生成 - AI 自动生成交易策略"))
        print(t("  3️⃣  回测验证 - 测试策略历史表现"))
        print(t("  4️⃣  执行交易 - 模拟盘下单"))
        print(t("  5️⃣  持仓监控 - 查看持仓和风控"))
        print(t("  6️⃣  一键全自动 - 全流程自动运转"))
        print(t("  7️⃣  紧急止损 - 清仓所有持仓"))
        print(t("  8️⃣  初始化 Qlib 数据"))
        print(t("  9️⃣  启动仪表盘 - 可视化界面"))
        print(t("  🔟  设置关注股票"))
        print(t("  0️⃣  退出"))
        print(f"{'═' * 50}")

        try:
            choice = input(t("\n  请选择 (0-9, 或输入 g 进行AI自主交易): ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(t("\n\n👋 再见！"))
            break

        if choice == "1":
            _menu_research()
        elif choice == "2":
            _menu_strategy()
        elif choice == "3":
            _menu_backtest()
        elif choice == "4":
            _menu_execute()
        elif choice == "5":
            _menu_monitor()
        elif choice == "6":
            _menu_full_auto()
        elif choice == "7":
            _menu_emergency_stop()
        elif choice == "8":
            _menu_init_qlib()
        elif choice == "9":
            _launch_dashboard()
        elif choice == "10":
            _menu_watchlist()
        elif choice in ("g", "G", "goal"):
            _menu_goal_driven()
        elif choice == "0":
            print(t("👋 再见！祝投资顺利！"))
            break
        else:
            print(t("  ⚠️ 无效选择，请重试"))


def _menu_goal_driven():
    """菜单: 目标驱动 AI 自主交易"""
    print(t("\n  🎯 AI 自主交易模式"))
    print("  ─────────────────────────────────")
    print(t('  告诉我你的资金和目标，AI 全自动搞定:'))
    print(t('  例: "我有10万美元，想赚20%"'))
    print(t('  例: "5万本金，目标翻倍，风险别太大"'))
    print(t('  例: "10万块，想赚5万"'))
    print()

    try:
        goal_text = input(t("  💬 你的目标: ")).strip()
        if not goal_text:
            print(t("  已取消。"))
            return

        print(t("\n  🚀 AI 正在为你自主决策，请耐心等待..."))
        from src.engine.goal_engine import goal_engine
        report = goal_engine.run(goal_text)
        print(report)
    except Exception as e:
        print(t('  ❌ AI 自主交易出错: {v1}', v1=e))


def _menu_research():
    """菜单: 市场调研"""
    print(t("\n  🔍 正在扫描市场..."))
    try:
        agent = agent_registry.get_research_agent()
        report = agent.run(max_candidates=config.MAX_CANDIDATES)
        if report is None:
            print(t("  ❌ 调研失败"))
            return
        print(f"\n{agent.plain_explain()}")
    except Exception as e:
        print(t('  ❌ 调研过程出错: {v1}', v1=e))


def _menu_strategy():
    """菜单: 策略生成"""
    print(t("\n  🧪 正在生成策略..."))
    try:
        research_agent = agent_registry.get_research_agent()
        research_result = research_agent.last_result()
        if research_result is None:
            print(t("  ⚠️ 请先执行市场调研 (菜单 1)"))
            return

        agent = agent_registry.get_strategy_agent()
        result = agent.run(research_result=research_result)
        if result is None:
            print(t("  ❌ 策略生成失败"))
            return
        print(f"\n{agent.plain_explain()}")
    except Exception as e:
        print(t('  ❌ 策略生成过程出错: {v1}', v1=e))


def _menu_backtest():
    """菜单: 回测验证"""
    print(t("\n  📊 正在回测验证..."))
    try:
        strategy_agent = agent_registry.get_strategy_agent()
        strategy_result = strategy_agent.last_result()
        if strategy_result is None:
            print(t("  ⚠️ 请先生成策略 (菜单 2)"))
            return

        agent = agent_registry.get_backtest_agent()
        result = agent.run(strategy_result=strategy_result)
        if result is None:
            print(t("  ❌ 回测失败"))
            return
        print(f"\n{agent.plain_explain()}")
    except Exception as e:
        print(t('  ❌ 回测过程出错: {v1}', v1=e))


def _menu_execute():
    """菜单: 执行交易"""
    print(t("\n  💰 正在执行交易..."))
    try:
        strategy_agent = agent_registry.get_strategy_agent()
        strategy_result = strategy_agent.last_result()
        if strategy_result is None:
            print(t("  ⚠️ 请先生成策略 (菜单 2)"))
            return

        agent = agent_registry.get_execution_agent()
        result = agent.run(strategy_result=strategy_result)
        if result is None:
            print(t("  ❌ 交易执行失败"))
            return
        print(f"\n{agent.plain_explain()}")
    except Exception as e:
        print(t('  ❌ 交易执行过程出错: {v1}', v1=e))


def _menu_monitor():
    """菜单: 持仓监控"""
    print(t("\n  📊 正在查询持仓..."))
    try:
        agent = agent_registry.get_monitor_agent()
        result = agent.run()
        if result is None:
            print(t("  ❌ 监控失败，请检查 Alpaca 配置"))
            return
        print(f"\n{agent.plain_explain()}")
    except Exception as e:
        print(t('  ❌ 监控过程出错: {v1}', v1=e))


def _menu_full_auto():
    """菜单: 一键全自动"""
    print(t("\n  🚀 启动全自动循环..."))
    print(t("  这可能需要几分钟，请耐心等待..."))

    try:
        result = workflow_engine.run_daily_loop()

        print(f"\n{'═' * 50}")
        print(t('  📋 全自动循环完成!'))
        print(t('  步骤: {v1}', v1=result.get('steps', {})))

        # 打印各 Agent 报告
        for step_name in ["research", "strategy", "backtest", "execution", "monitor"]:
            agent_map = {
                "research": "research", "strategy": "strategy",
                "backtest": "backtest", "execution": "execution",
                "monitor": "monitor",
            }
            agent = agent_registry.get(agent_map.get(step_name, ""))
            if agent and agent.last_result() is not None:
                print(t('\n📋 {v1}报告:', v1=agent.role))
                print(agent.plain_explain())
    except Exception as e:
        print(t('  ❌ 全自动循环出错: {v1}', v1=e))


def _menu_emergency_stop():
    """菜单: 紧急止损"""
    print(t("\n  🚨 启动紧急止损流程..."))
    print(t("  ⚠️ 这将清仓所有持仓！"))

    try:
        confirm = input(t("  确认执行? (输入 yes 确认): ")).strip().lower()
        if confirm not in ("yes", "y", t("确认")):
            print(t("  已取消。"))
            return

        result = workflow_engine.run_emergency_stop()

        print(f"\n{'═' * 50}")
        print(t('  📋 紧急止损完成!'))
        print(t('  步骤: {v1}', v1=result.get('steps', {})))

        # 打印各步骤报告
        step_agent_map = {
            "risk_check": "monitor",
            "liquidate": "execution",
            "notify": "monitor",
        }
        for step_name, agent_name in step_agent_map.items():
            agent = agent_registry.get(agent_name)
            if agent and agent.last_result() is not None:
                print(t('\n📋 {v1}报告:', v1=agent.role))
                print(agent.plain_explain())
    except Exception as e:
        print(t('  ❌ 紧急止损失败: {v1}', v1=e))


def _menu_status():
    """菜单: 系统状态"""
    response = nl_router.respond(t("系统状态"))
    print(response)


def _menu_init_qlib():
    """菜单: 初始化 Qlib 数据"""
    symbols_input = input(t("  输入股票代码 (逗号分隔, 如 AAPL,MSFT,GOOGL): ")).strip()
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    if not symbols:
        print(t("  ⚠️ 未输入股票代码"))
        return

    print(t('\n  📦 正在初始化 Qlib 数据: {v1}', v1=symbols))
    from src.data.qlib_adapter import qlib_adapter
    result = qlib_adapter.init_qlib_data(symbols)

    print(t('  ✅ 成功: {v1}', v1=result['success']))
    print(t('  失败: {v1}', v1=result['failed']))


def _load_user_watchlist():
    """从本地 JSON 加载用户关注列表 (与 dashboard 共享同一文件)"""
    import json
    filepath = config.DATA_DIR / "user_watchlist.json"
    try:
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                wl = data.get("watchlist", [])
                return [s.strip().upper() for s in wl if isinstance(s, str)]
    except Exception as e:
        logger.debug(f"加载关注列表失败: {e}")
    return []


def _save_user_watchlist(watchlist):
    """保存用户关注列表到本地 JSON (与 dashboard 共享同一文件)"""
    import json
    filepath = config.DATA_DIR / "user_watchlist.json"
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"watchlist": watchlist, "updated": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"保存关注列表失败: {e}")
        return False


def _menu_watchlist():
    """菜单: 设置关注股票 (持久化到 data/user_watchlist.json, 与仪表盘共享)"""
    watchlist = _load_user_watchlist()

    print(t("\n  ⭐ 设置关注股票"))
    print("  ─────────────────────────────────")
    if watchlist:
        print(t('  当前关注 ({v1} 只):', v1=len(watchlist)))
        for i in range(0, len(watchlist), 8):
            print("     " + "  ".join(watchlist[i:i+8]))
    else:
        print(t("  当前无关注股票 (实时行情将使用默认列表)"))

    print(t("\n  操作: 1=添加  2=删除  3=清空  直接回车=返回"))
    op = input(t("  请选择: ")).strip()

    if op == "1":
        raw = input(t("  输入股票代码 (逗号分隔, 如 AAPL,MSFT): ")).strip()
        new_symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        added = [s for s in new_symbols if s not in watchlist]
        if not added:
            print(t("  ⚠️ 无新增 (均已存在)"))
            return
        watchlist.extend(added)
        if _save_user_watchlist(watchlist):
            print(t('  ✅ 已添加: {v1}', v1=", ".join(added)))
    elif op == "2":
        if not watchlist:
            print(t("  ⚠️ 关注列表为空"))
            return
        raw = input(t("  输入要删除的股票代码 (逗号分隔): ")).strip()
        del_symbols = {s.strip().upper() for s in raw.split(",") if s.strip()}
        removed = [s for s in watchlist if s in del_symbols]
        if not removed:
            print(t("  ⚠️ 未找到匹配的股票"))
            return
        watchlist = [s for s in watchlist if s not in del_symbols]
        if _save_user_watchlist(watchlist):
            print(t('  ✅ 已删除: {v1}', v1=", ".join(removed)))
    elif op == "3":
        confirm = input(t("  确认清空所有关注? (y/N): ")).strip().lower()
        if confirm == "y":
            if _save_user_watchlist([]):
                print(t("  ✅ 已清空关注列表"))
        else:
            print(t("  已取消"))
    else:
        return


def _is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    """检查端口是否被占用"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def _launch_dashboard():
    """启动 Streamlit 仪表盘 (含悬浮智能助手聊天页面)"""
    import subprocess
    import threading

    print(t("\n  📈 正在启动 Streamlit 仪表盘..."))
    print(t("  服务地址: http://localhost:8501"))
    print(t("  按 Ctrl+C 退出仪表盘\n"))

    is_frozen = getattr(sys, 'frozen', False)

    # 启动独立智能助手聊天服务 (FastAPI, 8502端口)
    chat_proc = None
    chat_thread = None

    # 检查 8502 端口是否已占用
    if _is_port_in_use(8502):
        print(t('  ⚠️ 端口 8502 已被占用, 跳过聊天服务启动 (可能已在运行)'))
    else:
        try:
            if is_frozen:
                # 打包模式: 用线程直接运行 uvicorn (无法用 subprocess -m uvicorn)
                def _run_chat_server():
                    try:
                        import uvicorn
                        from src.api.chat_server import app
                        uvicorn.run(app, host="0.0.0.0", port=8502, log_level="warning")
                    except Exception as e:
                        print(t('  ⚠️ 聊天服务异常: {v1}', v1=e))

                chat_thread = threading.Thread(target=_run_chat_server, daemon=True)
                chat_thread.start()
                print(t('  🤖 智能助手聊天服务已启动 (线程模式, 端口 8502)'))
            else:
                # 开发模式: 用 subprocess 启动独立进程
                chat_proc = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn",
                     "src.api.chat_server:app",
                     "--host", "0.0.0.0",
                     "--port", "8502"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                )
                print(t('  🤖 智能助手聊天服务已启动 (PID: {v1}, 端口 8502)', v1=chat_proc.pid))
        except Exception as e:
            print(t('  ⚠️ 智能助手聊天服务启动失败: {v1}', v1=e))

    # 启动 Streamlit 仪表盘
    try:
        if is_frozen:
            # 打包模式: 编程式启动 Streamlit (不依赖 python -m streamlit)
            _run_streamlit_frozen()
        else:
            # 开发模式: 用 subprocess 启动 streamlit
            if _is_port_in_use(8501):
                print(t("  ⚠️ 端口 8501 已被占用, 可能已有仪表盘在运行"))
                print(t("     如需重新启动, 请先关闭旧的仪表盘进程"))
                return

            dashboard_path = config.DATA_DIR.parent / "src" / "ui" / "dashboard.py"
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
                 "--server.port", "8501",
                 "--server.headless", "true",
                 "--browser.gatherUsageStats", "false"],
                check=True,
            )
    except subprocess.CalledProcessError as e:
        print(t('  ❌ 仪表盘启动失败: {v1}', v1=e))
    except KeyboardInterrupt:
        print(t("\n  仪表盘已退出。"))
    finally:
        # 退出时清理聊天服务
        if chat_proc and chat_proc.poll() is None:
            chat_proc.terminate()
            print(t("  🤖 智能助手聊天页面已停止"))
        # 线程模式无需手动清理 (daemon=True 会随主进程退出)


def _run_streamlit_frozen():
    """打包模式下编程式启动 Streamlit 仪表盘

    PyInstaller 打包后没有独立 Python 解释器，无法用 `python -m streamlit run`。
    改为直接调用 Streamlit 内部 API 启动。
    """
    import os

    # 获取 dashboard.py 路径 (打包在 _internal/src/ui/dashboard.py)
    if hasattr(sys, '_MEIPASS'):
        # onedir 模式: _MEIPASS 指向 _internal 目录
        dashboard_path = os.path.join(sys._MEIPASS, "src", "ui", "dashboard.py")
    else:
        # fallback: exe 同级目录
        dashboard_path = os.path.join(os.path.dirname(sys.executable), "src", "ui", "dashboard.py")

    if not os.path.exists(dashboard_path):
        print(t('  ❌ 找不到仪表盘文件: {v1}', v1=dashboard_path))
        print(t("     请确保打包时已包含 src/ui/dashboard.py"))
        return

    # 构造 streamlit run 命令行参数
    original_argv = sys.argv
    sys.argv = [
        "streamlit",
        "run",
        dashboard_path,
        "--server.port", "8501",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ]

    try:
        from streamlit.web import bootstrap
        bootstrap.run(dashboard_path, "", [], flag_options={})
    except ImportError:
        # 兼容旧版 streamlit
        try:
            from streamlit import bootstrap
            bootstrap.run(dashboard_path, "", [], flag_options={})
        except Exception as e:
            print(t('  ❌ Streamlit 启动失败: {v1}', v1=e))
            print(t("     请确保 streamlit 已正确打包"))
    except Exception as e:
        print(t('  ❌ Streamlit 启动失败: {v1}', v1=e))
    finally:
        sys.argv = original_argv


def _launch_scheduler():
    """启动定时任务调度器"""
    print(t("\n  ⏰ 正在启动定时任务调度器..."))

    try:
        from src.scheduler.scheduler import trading_scheduler

        # 注册默认任务
        trading_scheduler.register_daily_loop()
        trading_scheduler.register_strategy_iteration()
        trading_scheduler.register_monitor_snapshot(5)

        # Phase 8: 加密货币 24/7 调度 (仅在启用时注册)
        if config.has_crypto_enabled():
            trading_scheduler.register_crypto_trading_loop()
            trading_scheduler.register_crypto_stoploss_check()
            trading_scheduler.register_crypto_risk_check()
            print(t("  🔒 加密货币 24/7 调度已注册"))

        trading_scheduler.start()

        print(t("  ✅ 调度器已启动!"))
        print(t("  📋 已注册任务:"))
        for task in trading_scheduler.list_tasks():
            print(f"     • {task['name']}: {task['schedule']}")
        print(t("\n  按 Ctrl+C 停止调度器\n"))

        # 保持主线程运行
        import time
        while trading_scheduler._running:
            time.sleep(1)

    except KeyboardInterrupt:
        print(t("\n  正在停止调度器..."))
        trading_scheduler.stop()
        print(t("  ⏹️ 调度器已停止。"))
    except Exception as e:
        print(t('  ❌ 调度器启动失败: {v1}', v1=e))


def _launch_api():
    """启动模型同步 API (仅 API, 不含调度器和 Dashboard)"""
    import os

    print(t("\n  🔌 启动模型同步 API..."))

    if config.DEPLOY_MODE != "cloud":
        print(t("  ⚠️ 当前 DEPLOY_MODE 不是 cloud, 建议在 .env 中设置 DEPLOY_MODE=cloud"))

    port = int(os.getenv("CLOUD_API_PORT", "8000"))
    host = os.getenv("CLOUD_API_HOST", "0.0.0.0")

    print(t('  📡 监听: http://{v1}:{v2}', v1=host, v2=port))
    print(t('  📋 API 文档: http://{v1}:{v2}/docs', v1=host, v2=port))
    print(f"  🔑 API Key: {'已配置' if config.MODEL_API_KEY else '未配置 (无认证)'}")
    print(t("  按 Ctrl+C 停止\n"))

    try:
        import uvicorn
        from src.api.model_sync import app
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        print(t("  ❌ FastAPI/uvicorn 未安装"))
        print(t("     请运行: pip install fastapi uvicorn python-multipart"))
    except KeyboardInterrupt:
        print(t("\n  ⏹️ API 已停止。"))
    except Exception as e:
        print(t('  ❌ API 启动失败: {v1}', v1=e))


def _launch_cloud():
    """启动云端所有服务 (API + 调度器 + Dashboard)"""
    import subprocess
    import os
    import threading

    print(t("\n  ☁️ 启动云端服务..."))

    if config.DEPLOY_MODE != "cloud":
        print(t("  ⚠️ 当前 DEPLOY_MODE 不是 cloud, 建议在 .env 中设置 DEPLOY_MODE=cloud"))

    port = int(os.getenv("CLOUD_API_PORT", "8000"))
    host = os.getenv("CLOUD_API_HOST", "0.0.0.0")

    print(t('  📡 API 端口: {v1}', v1=port))
    print(f"  📊 Dashboard: 8501")
    print(t('  ⏰ 调度器: 自动注册默认任务'))
    print(t("  按 Ctrl+C 停止所有服务\n"))

    # 启动调度器 (后台线程)
    try:
        from src.scheduler.scheduler import trading_scheduler
        trading_scheduler.register_daily_loop()
        trading_scheduler.register_strategy_iteration()
        trading_scheduler.register_monitor_snapshot(5)
        trading_scheduler.start()
        print(t("  ✅ 调度器已启动"))
    except Exception as e:
        print(t('  ⚠️ 调度器启动失败: {v1}', v1=e))

    # 启动 Dashboard (后台进程/线程)
    dashboard_proc = None
    dashboard_thread = None
    chat_proc = None
    chat_thread = None

    is_frozen = getattr(sys, 'frozen', False)

    if is_frozen:
        # 打包模式: 用线程启动 Dashboard 和聊天服务
        # 检查 8501 是否已占用
        if _is_port_in_use(8501):
            print(t("  ⚠️ 端口 8501 已被占用, 跳过 Dashboard 启动 (可能已在运行)"))
        else:
            try:
                def _run_dashboard_cloud():
                    try:
                        _run_streamlit_frozen()
                    except Exception as e:
                        print(t('  ⚠️ Dashboard 异常: {v1}', v1=e))
                dashboard_thread = threading.Thread(target=_run_dashboard_cloud, daemon=True)
                dashboard_thread.start()
                print(t("  ✅ Dashboard 已启动 (线程模式)"))
            except Exception as e:
                print(t('  ⚠️ Dashboard 启动失败: {v1}', v1=e))

        # 检查 8502 是否已占用
        if _is_port_in_use(8502):
            print(t("  ⚠️ 端口 8502 已被占用, 跳过聊天服务启动 (可能已在运行)"))
        else:
            try:
                def _run_chat_cloud():
                    try:
                        import uvicorn
                        from src.api.chat_server import app
                        uvicorn.run(app, host="0.0.0.0", port=8502, log_level="warning")
                    except Exception as e:
                        print(t('  ⚠️ 聊天服务异常: {v1}', v1=e))
                chat_thread = threading.Thread(target=_run_chat_cloud, daemon=True)
                chat_thread.start()
                print(t("  ✅ 智能助手聊天服务已启动 (线程模式)"))
            except Exception as e:
                print(t('  ⚠️ 智能助手聊天服务启动失败: {v1}', v1=e))
    else:
        # 开发模式: 用 subprocess 启动独立进程
        # 检查 8501 是否已占用
        if _is_port_in_use(8501):
            print(t("  ⚠️ 端口 8501 已被占用, 跳过 Dashboard 启动 (可能已在运行)"))
        else:
            try:
                dashboard_path = config.DATA_DIR.parent / "src" / "ui" / "dashboard.py"
                dashboard_proc = subprocess.Popen(
                    [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
                     "--server.port", "8501", "--server.headless", "true",
                     "--server.enableCORS", "false",
                     "--server.enableXsrfProtection", "false"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(t('  ✅ Dashboard 已启动 (PID: {v1})', v1=dashboard_proc.pid))
            except Exception as e:
                print(t('  ⚠️ Dashboard 启动失败: {v1}', v1=e))

        # 检查 8502 是否已占用
        if _is_port_in_use(8502):
            print(t("  ⚠️ 端口 8502 已被占用, 跳过聊天服务启动 (可能已在运行)"))
        else:
            try:
                chat_proc = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn",
                     "src.api.chat_server:app",
                     "--host", "0.0.0.0",
                     "--port", "8502"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(t('  ✅ 智能助手聊天服务已启动 (PID: {v1})', v1=chat_proc.pid))
            except Exception as e:
                print(t('  ⚠️ 智能助手聊天服务启动失败: {v1}', v1=e))

    # 启动 API (主进程阻塞)
    try:
        import uvicorn
        from src.api.model_sync import app
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        print(t("  ❌ FastAPI/uvicorn 未安装"))
        print(t("     请运行: pip install fastapi uvicorn python-multipart"))
    except KeyboardInterrupt:
        print(t("\n  正在停止所有服务..."))
        if dashboard_proc and dashboard_proc.poll() is None:
            dashboard_proc.terminate()
            print(t("  ⏹️ Dashboard 已停止"))
        if chat_proc and chat_proc.poll() is None:
            chat_proc.terminate()
            print(t("  ⏹️ 智能助手聊天页面已停止"))
        # 打包模式下线程为 daemon=True，随主进程自动退出
        try:
            trading_scheduler.stop()
            print(t("  ⏹️ 调度器已停止"))
        except Exception:
            pass
        print(t("  ⏹️ 所有服务已停止。"))
    except Exception as e:
        print(t('  ❌ API 启动失败: {v1}', v1=e))


def _launch_realtime(symbols=None):
    """启动实时行情 WebSocket 数据流"""
    import asyncio

    # 提前导入: 确保 KeyboardInterrupt 分支可访问 realtime_manager (修复 NameError)
    try:
        from src.data.realtime_manager import realtime_manager
    except Exception as e:
        print(t('  ❌ 实时行情模块加载失败: {v1}', v1=e))
        return

    print(t("\n  ⚡ 正在启动实时行情 WebSocket..."))

    # 确定订阅股票列表
    if symbols is None:
        # 默认: 用户关注列表 (user_watchlist.json) > .env WATCHLIST > 默认列表
        user_watchlist = _load_user_watchlist()
        watchlist = user_watchlist or config.WATCHLIST or config.DEFAULT_WATCHLIST
        symbols = watchlist
    else:
        symbols = [s.strip().upper() for s in symbols]

    # 检查配置
    if not config.REALTIME_ENABLED:
        print(t("  ❌ 实时行情已禁用 (REALTIME_ENABLED=false)"))
        print(t("     请在 .env 中设置 REALTIME_ENABLED=true"))
        return

    alpaca_ok = config.has_alpaca_keys()
    finnhub_ok = config.has_finnhub_key()
    if not alpaca_ok and not finnhub_ok:
        print(t("  ❌ 无可用数据源!"))
        print(t("     请在 .env 中配置 ALPACA_API_KEY 或 FINNHUB_API_KEY"))
        return

    print(t('  📋 订阅股票: {v1}', v1=', '.join(symbols)))
    print(t('  📡 数据源: Alpaca {v1} | Finnhub {v2}', v1='✅' if alpaca_ok else '❌', v2='✅' if finnhub_ok else '❌'))
    print(t('  🔧 数据源模式: {v1}', v1=config.REALTIME_DATA_SOURCE))
    print(t('  🚨 价格异动告警阈值: {v1:.1%}', v1=config.REALTIME_PRICE_ALERT_THRESHOLD))
    print(t("  按 Ctrl+C 停止\n"))

    async def _run():
        # 注册回调: 价格更新打印
        def on_price_update(quote):
            change_str = f"{quote.change_pct:+.2f}%" if quote.change_pct else "N/A"
            chg_amt = f"{quote.change:+.2f}" if quote.change else "N/A"
            print(f"  💲 {quote.symbol:5s} | {chg_amt:>8s} | {change_str:>8s} | {quote.source}")

        def on_price_alert(alert):
            print(f"  🚨 ALERT: {alert.message}")

        def on_bar_complete(bar):
            print(t('  📊 K线完成: {v1} | O:{v2:.2f} H:{v3:.2f} L:{v4:.2f} C:{v5:.2f} V:{v6}', v1=bar.symbol, v2=bar.open, v3=bar.high, v4=bar.low, v5=bar.close, v6=bar.volume))

        realtime_manager.register_callback("on_price_update", on_price_update)
        realtime_manager.register_callback("on_price_alert", on_price_alert)
        realtime_manager.register_callback("on_bar_complete", on_bar_complete)

        # 启动
        await realtime_manager.start(symbols)

        if not realtime_manager.is_running:
            print(t("\n  ⚠️ 实时行情未能启动 (可能 API Key 无效或网络问题)"))
            print(t("     降级提示: 系统可使用 REST 轮询模式获取行情"))
            return

        print(t("  ✅ 实时行情已启动! 等待数据流入...\n"))

        # 保持运行, 定时打印状态
        while realtime_manager.is_running:
            await asyncio.sleep(30)
            status = realtime_manager.get_status()
            print(t('\n  ── 状态 [{v1}] ──', v1=datetime.now().strftime('%H:%M:%S')))
            print(t('  缓存价格: {v1} 只 | 告警: {v2} 条', v1=status['cached_prices'], v2=status['alerts_count']))
            prices = realtime_manager.get_all_prices()
            for sym, q in list(prices.items())[:5]:
                chg = f"{q.change_pct:+.2f}%" if q.change_pct else "N/A"
                print(f"  {sym:5s}  {chg} [{q.source}]")
            print()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print(t("\n  正在停止实时行情..."))
        try:
            asyncio.run(realtime_manager.stop())
        except Exception:
            pass
        print(t("  ⏹️ 实时行情已停止。"))


def _quick_health_check():
    """快速健康检查"""
    from src.data.collector import data_collector
    from src.execution.alpaca_client import alpaca_client

    print(t("  🔍 快速健康检查..."))

    # 检查数据源
    print(t("\n  📊 数据源:"))
    try:
        df = data_collector.get_single_stock_daily("AAPL", limit=5)
        if df is not None and len(df) > 0:
            print(t('  ✅ AKShare 数据获取正常 (AAPL: {v1} 条)', v1=len(df)))
        else:
            print(t("  ❌ AKShare 数据获取失败"))
    except Exception as e:
        print(t('  ❌ AKShare 异常: {v1}', v1=e))

    # 检查 Finnhub
    if config.has_finnhub_key():
        try:
            from src.data.finnhub_client import finnhub_client
            news = finnhub_client.get_market_news(limit=1)
            print(t('  ✅ Finnhub API 正常'))
        except Exception as e:
            print(t('  ⚠️ Finnhub API 异常: {v1}', v1=e))
    else:
        print(t("  ⚠️ Finnhub API Key 未配置"))

    # 检查 Alpaca
    if config.has_alpaca_keys():
        try:
            account = alpaca_client.get_account()
            if account:
                print(t('  ✅ Alpaca 连接正常 (Paper: ${v1:,.2f})', v1=account.cash))
            else:
                print(t("  ❌ Alpaca 连接失败"))
        except Exception as e:
            print(t('  ⚠️ Alpaca 异常: {v1}', v1=e))
    else:
        print(t("  ⚠️ Alpaca API Key 未配置"))

    # 检查 Qlib
    if config.QLIB_DATA_PATH.exists():
        print(t('  ✅ Qlib 数据目录: {v1}', v1=config.QLIB_DATA_PATH))
    else:
        print(t('  ⚠️ Qlib 数据未初始化'))

    # v0.4.0: 检查 LLM
    print(t("\n  🧠 LLM 智能对话:"))
    try:
        from src.llm.llm_config import llm_config
        llm_config.load()
        llm_st = llm_config.status()
        if llm_st["enabled"]:
            print(t('  ✅ LLM 已启用 (默认模型: {v1})', v1=llm_st['default_model']))
            for m in llm_config.get_available_models():
                print(f"     • {m.name}: {m.model_id} @ {m.endpoint}")
        elif llm_st["total_models"] > 0:
            print(t('  ⚠️ 已配置 {v1} 个模型但 Key 未填写', v1=llm_st['total_models']))
            print(t('     模型列表: {v1}', v1=', '.join(llm_st['all_models'])))
            print(t('     请在 .env 中填入对应的 API Key'))
        else:
            print(t('  ℹ️ 未配置 LLM (使用规则引擎模式)'))
    except Exception as e:
        print(t('  ⚠️ LLM 检测异常: {v1}', v1=e))

    print(f"\n  {'─' * 40}")
    print(t("  健康检查完成。"))


def main():
    """主入口"""
    # [打包修复] 强制 stdout/stderr 使用 UTF-8 编码
    # PyInstaller 打包后控制台默认编码为 GBK (Windows 中文系统)，
    # 无法编码 emoji (🤖 等)，导致 print_banner() 抛 UnicodeEncodeError 崩溃。
    # errors='replace' 兜底: 即使终端不支持 UTF-8 也不会崩溃。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # [打包版] 可选安装缺失的大依赖 (torch/qlib)
    # 必须在 UTF-8 修复之后调用，否则安装进度中的 emoji 会触发 GBK 编码崩溃
    try:
        if getattr(sys, 'frozen', False):
            from runtime_hook_deps import ensure_heavy_deps
            ensure_heavy_deps()
    except Exception:
        pass

    # 解析 --debug 标志（可出现在任意位置，如 `main.py --debug` 或 `main.py chat --debug`）
    debug_mode = "--debug" in sys.argv
    if debug_mode:
        sys.argv.remove("--debug")

    setup_logger(debug=debug_mode)

    if debug_mode:
        logger.info("🐛 Debug 模式已开启 - 日志将显示完整代码路径 (模块:函数:行号)")

    # 初始化 Agent 和工作流
    agent_registry.initialize()
    workflow_engine.initialize()

    print_banner()
    print_status()

    # 检查命令行参数
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "chat":
            chat_mode()
            return
        elif cmd == "menu":
            menu_mode()
            return
        elif cmd == "auto":
            # 命令行直接执行全自动循环
            interest = sys.argv[2] if len(sys.argv) > 2 else ""
            try:
                result = workflow_engine.run_daily_loop(user_interest=interest)
                print(t('\n全自动循环完成: {v1}', v1=result.get('steps', {})))
            except Exception as e:
                print(t('\n❌ 全自动循环失败: {v1}', v1=e))
            return
        elif cmd == "goal":
            # 目标驱动 AI 自主交易
            # 用法: python main.py goal "我有10万美元，想赚20%"
            goal_text = sys.argv[2] if len(sys.argv) > 2 else ""
            if not goal_text:
                print(t('  ❌ 请提供投资目标，例如:'))
                print(t('  python main.py goal "我有10万美元，想赚20%"'))
                print(t('  python main.py goal "5万本金，目标翻倍"'))
                return
            try:
                from src.engine.goal_engine import goal_engine
                report = goal_engine.run(goal_text)
                print(report)
            except Exception as e:
                print(t('\n❌ AI 自主交易失败: {v1}', v1=e))
            return
        elif cmd == "config":
            # 显示/修改配置
            if len(sys.argv) > 2:
                key = sys.argv[2]
                val = sys.argv[3] if len(sys.argv) > 3 else None
                if val is not None:
                    if hasattr(config, key.upper()):
                        setattr(config, key.upper(), type(getattr(config, key.upper()))(val))
                        print(f"  ✅ {key} = {val}")
                    else:
                        print(t('  ❌ 未知配置项: {v1}', v1=key))
                else:
                    print(f"  {key} = {getattr(config, key.upper(), 'N/A')}")
            else:
                print(t("\n  📋 当前配置:"))
                for k, v in config.to_dict().items():
                    print(f"     {k}: {v}")
                print()
            return
        elif cmd == "status":
            _menu_status()
            return
        elif cmd == "health":
            _quick_health_check()
            return
        elif cmd == "init":
            # 初始化 Qlib 数据
            symbols = sys.argv[2].split(",") if len(sys.argv) > 2 else ["AAPL", "MSFT", "GOOGL"]
            from src.data.qlib_adapter import qlib_adapter
            result = qlib_adapter.init_qlib_data([s.strip().upper() for s in symbols])
            print(t('初始化完成: {v1}', v1=result))
            return
        elif cmd == "emergency":
            # 命令行直接执行紧急止损
            try:
                result = workflow_engine.run_emergency_stop()
                print(t('\n紧急止损完成: {v1}', v1=result.get('steps', {})))
            except Exception as e:
                print(t('\n❌ 紧急止损失败: {v1}', v1=e))
            return
        elif cmd == "dashboard":
            # 启动 Streamlit 仪表盘
            _launch_dashboard()
            return
        elif cmd == "scheduler":
            # 启动定时任务调度器
            _launch_scheduler()
            return
        elif cmd == "realtime":
            # 启动实时行情 WebSocket
            symbols = sys.argv[2].split(",") if len(sys.argv) > 2 else None
            _launch_realtime(symbols)
            return
        elif cmd == "cloud":
            # 启动云端服务 (API + 调度器 + Dashboard)
            _launch_cloud()
            return
        elif cmd == "api":
            # 仅启动模型同步 API
            _launch_api()
            return

    # 默认进入对话模式
    chat_mode()


if __name__ == "__main__":
    main()
