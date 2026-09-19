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
    print("""
═══════════════════════════════════════════════════════════
     🤖 AI 全自动美股量化交易系统 v0.4.0                    
     ─────────────────────────────────────────────          
     📊 数据: AKShare + Finnhub                             
     🧠 引擎: Microsoft Qlib + LightGBM                     
     🤖 Agent: 调研->策略->回测->执行->监控                     
     💰 交易: Alpaca Paper Trading                          
     🛡️  风控: 7 条规则全检                                 
     📈 仪表盘: Streamlit 可视化                             
     ⏰ 调度器: 定时自动运行                                  
     🧠 LLM: 智能对话 (多模型支持)                            
═══════════════════════════════════════════════════════════
    """)


def print_status():
    """打印系统状态"""
    print(f"{'─' * 60}")
    print(f"  📋 系统配置:")
    print(f"     交易模式: {'📝 模拟盘 (Paper)' if config.is_paper_trading() else '🔴 实盘 (Live)'}")
    print(f"     Alpaca:   {'✅ 已配置' if config.has_alpaca_keys() else '❌ 未配置'}")
    print(f"     Finnhub:  {'✅ 已配置' if config.has_finnhub_key() else '❌ 未配置'}")
    print(f"     最大仓位: {config.MAX_POSITION_PCT:.0%} | 日最大亏损: {config.MAX_DAILY_LOSS:.0%} | 最大回撤: {config.MAX_DRAWDOWN:.0%}")

    # v0.4.0: LLM 状态
    try:
        from src.llm.llm_config import llm_config
        llm_config.load()
        llm_st = llm_config.status()
        if llm_st["enabled"]:
            print(f"     LLM: 🧠 智能模式 (模型: {llm_st['default_model']})")
        elif llm_st["total_models"] > 0:
            avail = llm_st["available_models"]
            if avail:
                print(f"     LLM: 🧠 智能模式 (模型: {', '.join(avail)})")
            else:
                print(f"     LLM: 📝 规则模式 (已配置 {llm_st['total_models']} 个模型但Key未填)")
        else:
            print(f"     LLM: 📝 规则模式 (未配置)")
    except Exception:
        print(f"     LLM: 📝 规则模式")

    print(f"{'─' * 60}")


def chat_mode():
    """自然语言对话模式"""
    # v0.4.0: 检测 LLM 状态并提示
    try:
        from src.llm.llm_config import llm_config
        llm_config.load()
        if llm_config.is_enabled():
            print(f"\n🧠 智能对话模式已启用 (LLM: {llm_config.get_default_name()})")
            print("   支持自然语言理解、多轮对话上下文记忆。")
        else:
            print("\n📝 规则对话模式 (未配置 LLM API Key)")
            print("   在 .env 中配置 LLM_{NAME}_* 环境变量可启用智能对话。")
    except Exception:
        pass

    print("\n" + plain_explainer.friendly_greeting())

    while True:
        try:
            user_input = input("\n💬 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再见！祝投资顺利！")
            break

        if not user_input:
            continue

        # 退出命令
        if user_input.lower() in ("quit", "exit", "q", "退出", "再见"):
            print("👋 再见！祝投资顺利！")
            break

        # P6 修复: 处理输入时增加 try/except，防止异常中断对话
        try:
            response = nl_router.respond(user_input)
            print(f"\n🤖 {response}")
        except Exception as e:
            logger.error(f"对话处理异常: {e}")
            print(f"\n🤖 抱歉，处理你的请求时出现了错误: {e}")
            print("   请稍后重试，或输入 \"帮助\" 查看可用功能。")


def menu_mode():
    """菜单模式"""
    while True:
        print(f"\n{'═' * 50}")
        print("  📋 功能菜单")
        print(f"{'─' * 50}")
        print("  🎯  AI 自主交易 - 告诉我目标和资金，全自动搞定")
        print("  1️⃣  市场调研 - 看看今天有什么机会")
        print("  2️⃣  策略生成 - AI 自动生成交易策略")
        print("  3️⃣  回测验证 - 测试策略历史表现")
        print("  4️⃣  执行交易 - 模拟盘下单")
        print("  5️⃣  持仓监控 - 查看持仓和风控")
        print("  6️⃣  一键全自动 - 全流程自动运转")
        print("  7️⃣  紧急止损 - 清仓所有持仓")
        print("  8️⃣  初始化 Qlib 数据")
        print("  9️⃣  启动仪表盘 - 可视化界面")
        print("  🔟  设置关注股票")
        print("  0️⃣  退出")
        print(f"{'═' * 50}")

        try:
            choice = input("\n  请选择 (0-9, 或输入 g 进行AI自主交易): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再见！")
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
            print("👋 再见！祝投资顺利！")
            break
        else:
            print("  ⚠️ 无效选择，请重试")


def _menu_goal_driven():
    """菜单: 目标驱动 AI 自主交易"""
    print("\n  🎯 AI 自主交易模式")
    print("  ─────────────────────────────────")
    print('  告诉我你的资金和目标，AI 全自动搞定:')
    print('  例: "我有10万美元，想赚20%"')
    print('  例: "5万本金，目标翻倍，风险别太大"')
    print('  例: "10万块，想赚5万"')
    print()

    try:
        goal_text = input("  💬 你的目标: ").strip()
        if not goal_text:
            print("  已取消。")
            return

        print("\n  🚀 AI 正在为你自主决策，请耐心等待...")
        from src.engine.goal_engine import goal_engine
        report = goal_engine.run(goal_text)
        print(report)
    except Exception as e:
        print(f"  ❌ AI 自主交易出错: {e}")


def _menu_research():
    """菜单: 市场调研"""
    print("\n  🔍 正在扫描市场...")
    try:
        agent = agent_registry.get_research_agent()
        report = agent.run(max_candidates=config.MAX_CANDIDATES)
        if report is None:
            print("  ❌ 调研失败")
            return
        print(f"\n{agent.plain_explain()}")
    except Exception as e:
        print(f"  ❌ 调研过程出错: {e}")


def _menu_strategy():
    """菜单: 策略生成"""
    print("\n  🧪 正在生成策略...")
    try:
        research_agent = agent_registry.get_research_agent()
        research_result = research_agent.last_result()
        if research_result is None:
            print("  ⚠️ 请先执行市场调研 (菜单 1)")
            return

        agent = agent_registry.get_strategy_agent()
        result = agent.run(research_result=research_result)
        if result is None:
            print("  ❌ 策略生成失败")
            return
        print(f"\n{agent.plain_explain()}")
    except Exception as e:
        print(f"  ❌ 策略生成过程出错: {e}")


def _menu_backtest():
    """菜单: 回测验证"""
    print("\n  📊 正在回测验证...")
    try:
        strategy_agent = agent_registry.get_strategy_agent()
        strategy_result = strategy_agent.last_result()
        if strategy_result is None:
            print("  ⚠️ 请先生成策略 (菜单 2)")
            return

        agent = agent_registry.get_backtest_agent()
        result = agent.run(strategy_result=strategy_result)
        if result is None:
            print("  ❌ 回测失败")
            return
        print(f"\n{agent.plain_explain()}")
    except Exception as e:
        print(f"  ❌ 回测过程出错: {e}")


def _menu_execute():
    """菜单: 执行交易"""
    print("\n  💰 正在执行交易...")
    try:
        strategy_agent = agent_registry.get_strategy_agent()
        strategy_result = strategy_agent.last_result()
        if strategy_result is None:
            print("  ⚠️ 请先生成策略 (菜单 2)")
            return

        agent = agent_registry.get_execution_agent()
        result = agent.run(strategy_result=strategy_result)
        if result is None:
            print("  ❌ 交易执行失败")
            return
        print(f"\n{agent.plain_explain()}")
    except Exception as e:
        print(f"  ❌ 交易执行过程出错: {e}")


def _menu_monitor():
    """菜单: 持仓监控"""
    print("\n  📊 正在查询持仓...")
    try:
        agent = agent_registry.get_monitor_agent()
        result = agent.run()
        if result is None:
            print("  ❌ 监控失败，请检查 Alpaca 配置")
            return
        print(f"\n{agent.plain_explain()}")
    except Exception as e:
        print(f"  ❌ 监控过程出错: {e}")


def _menu_full_auto():
    """菜单: 一键全自动"""
    print("\n  🚀 启动全自动循环...")
    print("  这可能需要几分钟，请耐心等待...")

    try:
        result = workflow_engine.run_daily_loop()

        print(f"\n{'═' * 50}")
        print(f"  📋 全自动循环完成!")
        print(f"  步骤: {result.get('steps', {})}")

        # 打印各 Agent 报告
        for step_name in ["research", "strategy", "backtest", "execution", "monitor"]:
            agent_map = {
                "research": "research", "strategy": "strategy",
                "backtest": "backtest", "execution": "execution",
                "monitor": "monitor",
            }
            agent = agent_registry.get(agent_map.get(step_name, ""))
            if agent and agent.last_result() is not None:
                print(f"\n📋 {agent.role}报告:")
                print(agent.plain_explain())
    except Exception as e:
        print(f"  ❌ 全自动循环出错: {e}")


def _menu_emergency_stop():
    """菜单: 紧急止损"""
    print("\n  🚨 启动紧急止损流程...")
    print("  ⚠️ 这将清仓所有持仓！")

    try:
        confirm = input("  确认执行? (输入 yes 确认): ").strip().lower()
        if confirm not in ("yes", "y", "确认"):
            print("  已取消。")
            return

        result = workflow_engine.run_emergency_stop()

        print(f"\n{'═' * 50}")
        print(f"  📋 紧急止损完成!")
        print(f"  步骤: {result.get('steps', {})}")

        # 打印各步骤报告
        step_agent_map = {
            "risk_check": "monitor",
            "liquidate": "execution",
            "notify": "monitor",
        }
        for step_name, agent_name in step_agent_map.items():
            agent = agent_registry.get(agent_name)
            if agent and agent.last_result() is not None:
                print(f"\n📋 {agent.role}报告:")
                print(agent.plain_explain())
    except Exception as e:
        print(f"  ❌ 紧急止损失败: {e}")


def _menu_status():
    """菜单: 系统状态"""
    response = nl_router.respond("系统状态")
    print(response)


def _menu_init_qlib():
    """菜单: 初始化 Qlib 数据"""
    symbols_input = input("  输入股票代码 (逗号分隔, 如 AAPL,MSFT,GOOGL): ").strip()
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    if not symbols:
        print("  ⚠️ 未输入股票代码")
        return

    print(f"\n  📦 正在初始化 Qlib 数据: {symbols}")
    from src.data.qlib_adapter import qlib_adapter
    result = qlib_adapter.init_qlib_data(symbols)

    print(f"  ✅ 成功: {result['success']}")
    print(f"  失败: {result['failed']}")


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

    print("\n  📈 正在启动 Streamlit 仪表盘...")
    print("  服务地址: http://localhost:8501")
    print("  按 Ctrl+C 退出仪表盘\n")

    is_frozen = getattr(sys, 'frozen', False)

    # 启动独立智能助手聊天服务 (FastAPI, 8502端口)
    chat_proc = None
    chat_thread = None

    # 检查 8502 端口是否已占用
    if _is_port_in_use(8502):
        print(f"  ⚠️ 端口 8502 已被占用, 跳过聊天服务启动 (可能已在运行)")
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
                        print(f"  ⚠️ 聊天服务异常: {e}")

                chat_thread = threading.Thread(target=_run_chat_server, daemon=True)
                chat_thread.start()
                print(f"  🤖 智能助手聊天服务已启动 (线程模式, 端口 8502)")
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
                print(f"  🤖 智能助手聊天服务已启动 (PID: {chat_proc.pid}, 端口 8502)")
        except Exception as e:
            print(f"  ⚠️ 智能助手聊天服务启动失败: {e}")

    # 启动 Streamlit 仪表盘
    try:
        if is_frozen:
            # 打包模式: 编程式启动 Streamlit (不依赖 python -m streamlit)
            _run_streamlit_frozen()
        else:
            # 开发模式: 用 subprocess 启动 streamlit
            if _is_port_in_use(8501):
                print("  ⚠️ 端口 8501 已被占用, 可能已有仪表盘在运行")
                print("     如需重新启动, 请先关闭旧的仪表盘进程")
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
        print(f"  ❌ 仪表盘启动失败: {e}")
    except KeyboardInterrupt:
        print("\n  仪表盘已退出。")
    finally:
        # 退出时清理聊天服务
        if chat_proc and chat_proc.poll() is None:
            chat_proc.terminate()
            print("  🤖 智能助手聊天页面已停止")
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
        print(f"  ❌ 找不到仪表盘文件: {dashboard_path}")
        print("     请确保打包时已包含 src/ui/dashboard.py")
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
            print(f"  ❌ Streamlit 启动失败: {e}")
            print("     请确保 streamlit 已正确打包")
    except Exception as e:
        print(f"  ❌ Streamlit 启动失败: {e}")
    finally:
        sys.argv = original_argv


def _launch_scheduler():
    """启动定时任务调度器"""
    print("\n  ⏰ 正在启动定时任务调度器...")

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
            print("  🔒 加密货币 24/7 调度已注册")

        trading_scheduler.start()

        print("  ✅ 调度器已启动!")
        print("  📋 已注册任务:")
        for task in trading_scheduler.list_tasks():
            print(f"     • {task['name']}: {task['schedule']}")
        print("\n  按 Ctrl+C 停止调度器\n")

        # 保持主线程运行
        import time
        while trading_scheduler._running:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n  正在停止调度器...")
        trading_scheduler.stop()
        print("  ⏹️ 调度器已停止。")
    except Exception as e:
        print(f"  ❌ 调度器启动失败: {e}")


def _launch_api():
    """启动模型同步 API (仅 API, 不含调度器和 Dashboard)"""
    import os

    print("\n  🔌 启动模型同步 API...")

    if config.DEPLOY_MODE != "cloud":
        print("  ⚠️ 当前 DEPLOY_MODE 不是 cloud, 建议在 .env 中设置 DEPLOY_MODE=cloud")

    port = int(os.getenv("CLOUD_API_PORT", "8000"))
    host = os.getenv("CLOUD_API_HOST", "0.0.0.0")

    print(f"  📡 监听: http://{host}:{port}")
    print(f"  📋 API 文档: http://{host}:{port}/docs")
    print(f"  🔑 API Key: {'已配置' if config.MODEL_API_KEY else '未配置 (无认证)'}")
    print("  按 Ctrl+C 停止\n")

    try:
        import uvicorn
        from src.api.model_sync import app
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        print("  ❌ FastAPI/uvicorn 未安装")
        print("     请运行: pip install fastapi uvicorn python-multipart")
    except KeyboardInterrupt:
        print("\n  ⏹️ API 已停止。")
    except Exception as e:
        print(f"  ❌ API 启动失败: {e}")


def _launch_cloud():
    """启动云端所有服务 (API + 调度器 + Dashboard)"""
    import subprocess
    import os
    import threading

    print("\n  ☁️ 启动云端服务...")

    if config.DEPLOY_MODE != "cloud":
        print("  ⚠️ 当前 DEPLOY_MODE 不是 cloud, 建议在 .env 中设置 DEPLOY_MODE=cloud")

    port = int(os.getenv("CLOUD_API_PORT", "8000"))
    host = os.getenv("CLOUD_API_HOST", "0.0.0.0")

    print(f"  📡 API 端口: {port}")
    print(f"  📊 Dashboard: 8501")
    print(f"  ⏰ 调度器: 自动注册默认任务")
    print("  按 Ctrl+C 停止所有服务\n")

    # 启动调度器 (后台线程)
    try:
        from src.scheduler.scheduler import trading_scheduler
        trading_scheduler.register_daily_loop()
        trading_scheduler.register_strategy_iteration()
        trading_scheduler.register_monitor_snapshot(5)
        trading_scheduler.start()
        print("  ✅ 调度器已启动")
    except Exception as e:
        print(f"  ⚠️ 调度器启动失败: {e}")

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
            print("  ⚠️ 端口 8501 已被占用, 跳过 Dashboard 启动 (可能已在运行)")
        else:
            try:
                def _run_dashboard_cloud():
                    try:
                        _run_streamlit_frozen()
                    except Exception as e:
                        print(f"  ⚠️ Dashboard 异常: {e}")
                dashboard_thread = threading.Thread(target=_run_dashboard_cloud, daemon=True)
                dashboard_thread.start()
                print("  ✅ Dashboard 已启动 (线程模式)")
            except Exception as e:
                print(f"  ⚠️ Dashboard 启动失败: {e}")

        # 检查 8502 是否已占用
        if _is_port_in_use(8502):
            print("  ⚠️ 端口 8502 已被占用, 跳过聊天服务启动 (可能已在运行)")
        else:
            try:
                def _run_chat_cloud():
                    try:
                        import uvicorn
                        from src.api.chat_server import app
                        uvicorn.run(app, host="0.0.0.0", port=8502, log_level="warning")
                    except Exception as e:
                        print(f"  ⚠️ 聊天服务异常: {e}")
                chat_thread = threading.Thread(target=_run_chat_cloud, daemon=True)
                chat_thread.start()
                print("  ✅ 智能助手聊天服务已启动 (线程模式)")
            except Exception as e:
                print(f"  ⚠️ 智能助手聊天服务启动失败: {e}")
    else:
        # 开发模式: 用 subprocess 启动独立进程
        # 检查 8501 是否已占用
        if _is_port_in_use(8501):
            print("  ⚠️ 端口 8501 已被占用, 跳过 Dashboard 启动 (可能已在运行)")
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
                print(f"  ✅ Dashboard 已启动 (PID: {dashboard_proc.pid})")
            except Exception as e:
                print(f"  ⚠️ Dashboard 启动失败: {e}")

        # 检查 8502 是否已占用
        if _is_port_in_use(8502):
            print("  ⚠️ 端口 8502 已被占用, 跳过聊天服务启动 (可能已在运行)")
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
                print(f"  ✅ 智能助手聊天服务已启动 (PID: {chat_proc.pid})")
            except Exception as e:
                print(f"  ⚠️ 智能助手聊天服务启动失败: {e}")

    # 启动 API (主进程阻塞)
    try:
        import uvicorn
        from src.api.model_sync import app
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        print("  ❌ FastAPI/uvicorn 未安装")
        print("     请运行: pip install fastapi uvicorn python-multipart")
    except KeyboardInterrupt:
        print("\n  正在停止所有服务...")
        if dashboard_proc and dashboard_proc.poll() is None:
            dashboard_proc.terminate()
            print("  ⏹️ Dashboard 已停止")
        if chat_proc and chat_proc.poll() is None:
            chat_proc.terminate()
            print("  ⏹️ 智能助手聊天页面已停止")
        # 打包模式下线程为 daemon=True，随主进程自动退出
        try:
            trading_scheduler.stop()
            print("  ⏹️ 调度器已停止")
        except Exception:
            pass
        print("  ⏹️ 所有服务已停止。")
    except Exception as e:
        print(f"  ❌ API 启动失败: {e}")


def _launch_realtime(symbols=None):
    """启动实时行情 WebSocket 数据流"""
    import asyncio

    print("\n  ⚡ 正在启动实时行情 WebSocket...")

    # 确定订阅股票列表
    if symbols is None:
        # 默认: 关注列表
        watchlist = config.WATCHLIST or config.DEFAULT_WATCHLIST
        symbols = watchlist
    else:
        symbols = [s.strip().upper() for s in symbols]

    # 检查配置
    if not config.REALTIME_ENABLED:
        print("  ❌ 实时行情已禁用 (REALTIME_ENABLED=false)")
        print("     请在 .env 中设置 REALTIME_ENABLED=true")
        return

    alpaca_ok = config.has_alpaca_keys()
    finnhub_ok = config.has_finnhub_key()
    if not alpaca_ok and not finnhub_ok:
        print("  ❌ 无可用数据源!")
        print("     请在 .env 中配置 ALPACA_API_KEY 或 FINNHUB_API_KEY")
        return

    print(f"  📋 订阅股票: {', '.join(symbols)}")
    print(f"  📡 数据源: Alpaca {'✅' if alpaca_ok else '❌'} | Finnhub {'✅' if finnhub_ok else '❌'}")
    print(f"  🔧 数据源模式: {config.REALTIME_DATA_SOURCE}")
    print(f"  🚨 价格异动告警阈值: {config.REALTIME_PRICE_ALERT_THRESHOLD:.1%}")
    print("  按 Ctrl+C 停止\n")

    async def _run():
        from src.data.realtime_manager import realtime_manager

        # 注册回调: 价格更新打印
        def on_price_update(quote):
            change_str = f"{quote.change_pct:+.2f}%" if quote.change_pct else "N/A"
            print(f"  💲 {quote.symbol:5s} |  | {change_str:>8s} | {quote.source}")

        def on_price_alert(alert):
            print(f"  🚨 ALERT: {alert.message}")

        def on_bar_complete(bar):
            print(f"  📊 K线完成: {bar.symbol} | O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} V:{bar.volume}")

        realtime_manager.register_callback("on_price_update", on_price_update)
        realtime_manager.register_callback("on_price_alert", on_price_alert)
        realtime_manager.register_callback("on_bar_complete", on_bar_complete)

        # 启动
        await realtime_manager.start(symbols)

        if not realtime_manager.is_running:
            print("\n  ⚠️ 实时行情未能启动 (可能 API Key 无效或网络问题)")
            print("     降级提示: 系统可使用 REST 轮询模式获取行情")
            return

        print("  ✅ 实时行情已启动! 等待数据流入...\n")

        # 保持运行, 定时打印状态
        while realtime_manager.is_running:
            await asyncio.sleep(30)
            status = realtime_manager.get_status()
            print(f"\n  ── 状态 [{datetime.now().strftime('%H:%M:%S')}] ──")
            print(f"  缓存价格: {status['cached_prices']} 只 | 告警: {status['alerts_count']} 条")
            prices = realtime_manager.get_all_prices()
            for sym, q in list(prices.items())[:5]:
                chg = f"{q.change_pct:+.2f}%" if q.change_pct else "N/A"
                print(f"  {sym:5s}  {chg} [{q.source}]")
            print()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n  正在停止实时行情...")
        try:
            asyncio.run(realtime_manager.stop())
        except Exception:
            pass
        print("  ⏹️ 实时行情已停止。")


def _quick_health_check():
    """快速健康检查"""
    from src.data.collector import data_collector
    from src.execution.alpaca_client import alpaca_client

    print("  🔍 快速健康检查...")

    # 检查数据源
    print("\n  📊 数据源:")
    try:
        df = data_collector.get_single_stock_daily("AAPL", limit=5)
        if df is not None and len(df) > 0:
            print(f"  ✅ AKShare 数据获取正常 (AAPL: {len(df)} 条)")
        else:
            print("  ❌ AKShare 数据获取失败")
    except Exception as e:
        print(f"  ❌ AKShare 异常: {e}")

    # 检查 Finnhub
    if config.has_finnhub_key():
        try:
            from src.data.finnhub_client import finnhub_client
            news = finnhub_client.get_market_news(limit=1)
            print(f"  ✅ Finnhub API 正常")
        except Exception as e:
            print(f"  ⚠️ Finnhub API 异常: {e}")
    else:
        print("  ⚠️ Finnhub API Key 未配置")

    # 检查 Alpaca
    if config.has_alpaca_keys():
        try:
            account = alpaca_client.get_account()
            if account:
                print(f"  ✅ Alpaca 连接正常 (Paper: ${account.cash:,.2f})")
            else:
                print("  ❌ Alpaca 连接失败")
        except Exception as e:
            print(f"  ⚠️ Alpaca 异常: {e}")
    else:
        print("  ⚠️ Alpaca API Key 未配置")

    # 检查 Qlib
    if config.QLIB_DATA_PATH.exists():
        print(f"  ✅ Qlib 数据目录: {config.QLIB_DATA_PATH}")
    else:
        print(f"  ⚠️ Qlib 数据未初始化")

    # v0.4.0: 检查 LLM
    print("\n  🧠 LLM 智能对话:")
    try:
        from src.llm.llm_config import llm_config
        llm_config.load()
        llm_st = llm_config.status()
        if llm_st["enabled"]:
            print(f"  ✅ LLM 已启用 (默认模型: {llm_st['default_model']})")
            for m in llm_config.get_available_models():
                print(f"     • {m.name}: {m.model_id} @ {m.endpoint}")
        elif llm_st["total_models"] > 0:
            print(f"  ⚠️ 已配置 {llm_st['total_models']} 个模型但 Key 未填写")
            print(f"     模型列表: {', '.join(llm_st['all_models'])}")
            print(f"     请在 .env 中填入对应的 API Key")
        else:
            print(f"  ℹ️ 未配置 LLM (使用规则引擎模式)")
    except Exception as e:
        print(f"  ⚠️ LLM 检测异常: {e}")

    print(f"\n  {'─' * 40}")
    print("  健康检查完成。")


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
                print(f"\n全自动循环完成: {result.get('steps', {})}")
            except Exception as e:
                print(f"\n❌ 全自动循环失败: {e}")
            return
        elif cmd == "goal":
            # 目标驱动 AI 自主交易
            # 用法: python main.py goal "我有10万美元，想赚20%"
            goal_text = sys.argv[2] if len(sys.argv) > 2 else ""
            if not goal_text:
                print('  ❌ 请提供投资目标，例如:')
                print('  python main.py goal "我有10万美元，想赚20%"')
                print('  python main.py goal "5万本金，目标翻倍"')
                return
            try:
                from src.engine.goal_engine import goal_engine
                report = goal_engine.run(goal_text)
                print(report)
            except Exception as e:
                print(f"\n❌ AI 自主交易失败: {e}")
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
                        print(f"  ❌ 未知配置项: {key}")
                else:
                    print(f"  {key} = {getattr(config, key.upper(), 'N/A')}")
            else:
                print("\n  📋 当前配置:")
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
            print(f"初始化完成: {result}")
            return
        elif cmd == "emergency":
            # 命令行直接执行紧急止损
            try:
                result = workflow_engine.run_emergency_stop()
                print(f"\n紧急止损完成: {result.get('steps', {})}")
            except Exception as e:
                print(f"\n❌ 紧急止损失败: {e}")
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
