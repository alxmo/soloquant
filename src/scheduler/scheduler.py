"""
定时任务调度器 - 自动化交易循环
=====================================================
v0.3.0 新增

功能:
- 每日盘前自动运行 daily_trading_loop 工作流
- 每周自动运行 strategy_iteration 策略迭代
- 定时监控持仓快照 (每5分钟)
- 支持手动启停、状态查看
- 与 workflow_engine 和 notifier 集成

基于 schedule 库实现，轻量级无需额外服务。
"""

import threading
from datetime import datetime
from typing import Callable, Optional

from src.models import AssetType
from src.utils.notifier import notifier
import schedule
from loguru import logger

from src.utils.config import config


class TaskInfo:
    """任务信息"""

    def __init__(
        self,
        name: str,
        description: str,
        schedule_desc: str,
        func: Callable,
    ):
        self.name = name
        self.description = description
        self.schedule_desc = schedule_desc
        self.func = func
        self.last_run: Optional[datetime] = None
        self.last_result: Optional[str] = None
        self.run_count: int = 0
        self.error_count: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "schedule": self.schedule_desc,
            "last_run": self.last_run.strftime("%Y-%m-%d %H:%M:%S") if self.last_run else "未运行",
            "last_result": self.last_result or "无",
            "run_count": self.run_count,
            "error_count": self.error_count,
        }


class TradingScheduler:
    """交易定时调度器"""

    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()  # 优雅停机信号
        self._current_task: Optional[str] = None  # 当前正在执行的任务
        self._lock = threading.Lock()

    def register_daily_loop(self, run_time: str = "06:00"):
        """
        注册每日自动交易循环

        Args:
            run_time: 运行时间 (HH:MM, 美东时间)
        """
        def _run_daily():
            self._execute_workflow("daily_trading_loop", "每日自动交易循环")

        schedule.every().day.at(run_time).do(_run_daily)

        self._tasks["daily_trading_loop"] = TaskInfo(
            name="daily_trading_loop",
            description="每日盘前自动: 调研→策略→回测→执行→监控",
            schedule_desc=f"每天 {run_time}",
            func=_run_daily,
        )
        logger.info(f"定时任务已注册: 每日交易循环 @ {run_time}")

    def register_strategy_iteration(self, run_time: str = "20:00", day: str = "sunday"):
        """
        注册每周策略迭代

        Args:
            run_time: 运行时间
            day: 星期几 (monday~sunday)
        """
        def _run_iteration():
            self._execute_workflow("strategy_iteration", "每周策略迭代")

        # 动态获取星期方法
        day_methods = {
            "monday": schedule.every().monday,
            "tuesday": schedule.every().tuesday,
            "wednesday": schedule.every().wednesday,
            "thursday": schedule.every().thursday,
            "friday": schedule.every().friday,
            "saturday": schedule.every().saturday,
            "sunday": schedule.every().sunday,
        }
        scheduler_day = day_methods.get(day.lower(), schedule.every().sunday)
        scheduler_day.at(run_time).do(_run_iteration)

        self._tasks["strategy_iteration"] = TaskInfo(
            name="strategy_iteration",
            description="每周策略复盘与优化迭代",
            schedule_desc=f"每周{day} {run_time}",
            func=_run_iteration,
        )
        logger.info(f"定时任务已注册: 每周策略迭代 @ {day} {run_time}")

    def register_monitor_snapshot(self, interval_minutes: int = 5):
        """
        注册定时监控快照

        Args:
            interval_minutes: 检查间隔 (分钟)
        """
        def _run_monitor():
            self._execute_monitor()

        schedule.every(interval_minutes).minutes.do(_run_monitor)

        self._tasks["monitor_snapshot"] = TaskInfo(
            name="monitor_snapshot",
            description=f"每{interval_minutes}分钟持仓风控快照",
            schedule_desc=f"每 {interval_minutes} 分钟",
            func=_run_monitor,
        )
        logger.info(f"定时任务已注册: 监控快照 @ 每{interval_minutes}分钟")

    # ═══════════════════════════════════════════════════════
    # Phase 8: 加密货币 24/7 调度
    # ═══════════════════════════════════════════════════════

    def register_crypto_trading_loop(self, interval_hours: int = None):
        """
        注册加密货币交易循环 (24/7 调度)

        加密货币市场 24/7 运行, 因此使用固定间隔调度而非每天特定时间。
        默认每 4 小时执行一次完整交易循环。

        Args:
            interval_hours: 执行间隔 (小时), 默认从配置读取
        """
        if interval_hours is None:
            interval_hours = getattr(config, "CRYPTO_SIGNAL_SCAN_INTERVAL_HOURS", 4)

        def _run_crypto():
            self._execute_workflow("crypto_trading_loop", "加密货币交易循环")

        schedule.every(interval_hours).hours.do(_run_crypto)

        self._tasks["crypto_trading_loop"] = TaskInfo(
            name="crypto_trading_loop",
            description=f"加密货币交易循环: 调研->信号->风控->执行->监控 (每{interval_hours}h)",
            schedule_desc=f"每 {interval_hours} 小时",
            func=_run_crypto,
        )
        logger.info(f"🔒 定时任务已注册: 加密货币交易循环 @ 每{interval_hours}小时")

    def register_crypto_stoploss_check(self, interval_minutes: int = None):
        """
        注册加密货币止损检查 (高频)

        加密货币波动大, 止损检查频率比股票更高。
        默认每 15 分钟检查一次。

        Args:
            interval_minutes: 检查间隔 (分钟), 默认从配置读取
        """
        if interval_minutes is None:
            interval_minutes = getattr(config, "CRYPTO_STOPLOSS_CHECK_INTERVAL_MINUTES", 15)

        def _run_crypto_stoploss():
            self._execute_crypto_stoploss()

        schedule.every(interval_minutes).minutes.do(_run_crypto_stoploss)

        self._tasks["crypto_stoploss_check"] = TaskInfo(
            name="crypto_stoploss_check",
            description=f"加密货币止损检查 (每{interval_minutes}分钟)",
            schedule_desc=f"每 {interval_minutes} 分钟",
            func=_run_crypto_stoploss,
        )
        logger.info(f"🔒 定时任务已注册: 加密货币止损检查 @ 每{interval_minutes}分钟")

    def register_crypto_risk_check(self, interval_hours: int = None):
        """
        注册加密货币风控检查

        定期检查加密货币组合风控状态 (回撤、日亏损等)。
        默认每 1 小时检查一次。

        Args:
            interval_hours: 检查间隔 (小时), 默认从配置读取
        """
        if interval_hours is None:
            interval_hours = getattr(config, "CRYPTO_RISK_CHECK_INTERVAL_HOURS", 1)

        def _run_crypto_risk():
            self._execute_crypto_risk_check()

        schedule.every(interval_hours).hours.do(_run_crypto_risk)

        self._tasks["crypto_risk_check"] = TaskInfo(
            name="crypto_risk_check",
            description=f"加密货币组合风控检查 (每{interval_hours}小时)",
            schedule_desc=f"每 {interval_hours} 小时",
            func=_run_crypto_risk,
        )
        logger.info(f"🔒 定时任务已注册: 加密货币风控检查 @ 每{interval_hours}小时")

    def _execute_crypto_stoploss(self):
        """执行加密货币止损检查"""
        task = self._tasks.get("crypto_stoploss_check")
        if task:
            task.last_run = datetime.now()
            task.run_count += 1

        with self._lock:
            self._current_task = "crypto_stoploss_check"

        try:
            from src.execution.risk_guard import risk_guard
            from src.execution.alpaca_client import alpaca_client

            if not alpaca_client.available:
                logger.debug("Alpaca 不可用，跳过加密货币止损检查")
                if task:
                    task.last_result = "Alpaca 不可用"
                return

            # 获取加密货币持仓 (返回 list[Position])
            all_positions = alpaca_client.get_positions()
            crypto_positions = [
                p for p in all_positions
                if "/" in p.symbol or p.asset_type == AssetType.CRYPTO
            ]

            if not crypto_positions:
                if task:
                    task.last_result = "无加密货币持仓"
                return

            stoploss_triggered = []
            for pos in crypto_positions:
                # pos 已经是 Position 对象, 有 unrealized_pnl_pct 属性
                unrealized_pnl_pct = pos.unrealized_pnl_pct

                if risk_guard.should_crypto_stop_loss(pos):
                    # 前置校验: 持仓金额需 >= $10 才能卖出
                    pos_value = pos.qty * pos.current_price if hasattr(pos, 'current_price') and pos.current_price else 0
                    if pos_value < 10.0:
                        logger.warning(
                            f"🔒 加密货币止损跳过: {pos.symbol} 持仓金额 ${pos_value:.2f} < $10"
                        )
                        continue

                    stoploss_triggered.append({
                        "symbol": pos.symbol,
                        "pnl_pct": unrealized_pnl_pct,
                    })
                    # 触发止损卖出
                    try:
                        alpaca_client.submit_crypto_order(
                            symbol=pos.symbol,
                            side="sell",
                            qty=pos.qty,
                            price=pos.current_price if hasattr(pos, 'current_price') else 0,
                        )
                        logger.warning(f"🔒 加密货币止损执行: {pos.symbol} 卖出 {pos.qty}")
                    except Exception as e:
                        logger.error(f"🔒 加密货币止损下单失败 [{pos.symbol}]: {e}")

            if task:
                if stoploss_triggered:
                    task.last_result = f"触发 {len(stoploss_triggered)} 个止损"
                    notifier.warning(
                        "加密货币止损触发",
                        f"以下币种触发止损:\n" + "\n".join(
                            f"  • {s['symbol']}: {s['pnl_pct']:.1%}" for s in stoploss_triggered
                        ),
                    )
                else:
                    task.last_result = "无止损触发"

        except Exception as e:
            logger.error(f"加密货币止损检查失败: {e}")
            if task:
                task.last_result = f"错误: {e}"
                task.error_count += 1
        finally:
            with self._lock:
                self._current_task = None

    def _execute_crypto_risk_check(self):
        """执行加密货币组合风控检查"""
        task = self._tasks.get("crypto_risk_check")
        if task:
            task.last_run = datetime.now()
            task.run_count += 1

        with self._lock:
            self._current_task = "crypto_risk_check"

        try:
            from src.execution.risk_guard import risk_guard
            from src.execution.alpaca_client import alpaca_client

            if not alpaca_client.available:
                logger.debug("Alpaca 不可用，跳过加密货币风控检查")
                if task:
                    task.last_result = "Alpaca 不可用"
                return

            # alpaca_client.get_account() 返回 Account 对象, get_positions() 返回 list[Position]
            account = alpaca_client.get_account()
            if not account:
                if task:
                    task.last_result = "无法获取账户信息"
                return

            # 获取所有持仓 (已经是 Position 对象)
            all_positions = alpaca_client.get_positions()
            positions = []
            for p in all_positions:
                # 为加密货币持仓标记 asset_type
                if "/" in p.symbol:
                    p.asset_type = AssetType.CRYPTO
                positions.append(p)

            # 加密货币组合风控检查
            risk_status = risk_guard.check_crypto_portfolio(account, positions)

            if task:
                if risk_status.is_alert:
                    task.last_result = f"告警: {len(risk_status.alerts)} 条"
                    for alert in risk_status.alerts:
                        notifier.warning("加密货币风控告警", alert)
                else:
                    task.last_result = "正常"

        except Exception as e:
            logger.error(f"加密货币风控检查失败: {e}")
            if task:
                task.last_result = f"错误: {e}"
                task.error_count += 1
        finally:
            with self._lock:
                self._current_task = None

    def register_custom(
        self,
        name: str,
        description: str,
        func: Callable,
        schedule_desc: str = "",
        every_seconds: int = None,
        every_minutes: int = None,
        every_hours: int = None,
        at_time: str = None,
    ):
        """
        注册自定义定时任务

        Args:
            name: 任务名称
            description: 任务描述
            func: 执行函数
            schedule_desc: 人类可读的调度描述
            every_seconds: 每N秒执行
            every_minutes: 每N分钟执行
            every_hours: 每N小时执行
            at_time: 指定时间 (HH:MM)
        """
        if every_seconds:
            schedule.every(every_seconds).seconds.do(func)
        elif every_minutes:
            if at_time:
                schedule.every(every_minutes).minutes.at(at_time).do(func)
            else:
                schedule.every(every_minutes).minutes.do(func)
        elif every_hours:
            if at_time:
                schedule.every(every_hours).hours.at(at_time).do(func)
            else:
                schedule.every(every_hours).hours.do(func)

        self._tasks[name] = TaskInfo(
            name=name,
            description=description,
            schedule_desc=schedule_desc or "自定义",
            func=func,
        )
        logger.info(f"自定义定时任务已注册: {name}")

    def _execute_workflow(self, workflow_name: str, display_name: str):
        """执行工作流 (带异常处理和通知)"""
        task = self._tasks.get(workflow_name)
        if task:
            task.last_run = datetime.now()
            task.run_count += 1

        with self._lock:
            self._current_task = workflow_name

        logger.info(f"⏰ 定时任务触发: {display_name}")
        notifier.info(f"定时任务启动: {display_name}", f"工作流: {workflow_name}")

        try:
            from src.workflow.engine import workflow_engine
            result = workflow_engine.run(workflow_name)

            status = result.get("status", "unknown")
            steps = result.get("steps", {})

            # 汇总结果
            step_summary = ", ".join(
                f"{k}: {v}" for k, v in steps.items()
            ) if steps else "无步骤数据"

            if task:
                task.last_result = f"状态: {status}"

            if status == "success":
                notifier.success(
                    f"{display_name}完成",
                    f"状态: 成功\n步骤: {step_summary}",
                )
            else:
                notifier.warning(
                    f"{display_name}完成 (有错误)",
                    f"状态: {status}\n步骤: {step_summary}",
                )

        except Exception as e:
            logger.error(f"定时任务执行失败 [{workflow_name}]: {e}")
            if task:
                task.last_result = f"错误: {e}"
                task.error_count += 1
            notifier.error(
                f"{display_name}执行失败",
                f"工作流: {workflow_name}\n错误: {e}",
            )
        finally:
            with self._lock:
                self._current_task = None

    def _execute_monitor(self):
        """执行监控快照"""
        task = self._tasks.get("monitor_snapshot")
        if task:
            task.last_run = datetime.now()
            task.run_count += 1

        with self._lock:
            self._current_task = "monitor_snapshot"

        try:
            from src.agents.registry import agent_registry
            monitor = agent_registry.get("monitor")
            if monitor:
                result = monitor.run(mode="snapshot")
                if task:
                    task.last_result = "监控完成"

                # 检查是否有告警
                if isinstance(result, dict):
                    alerts = result.get("alerts", [])
                    if alerts:
                        notifier.warning(
                            "持仓风控告警",
                            f"检测到 {len(alerts)} 条告警:\n" + "\n".join(f"  • {a}" for a in alerts),
                        )
            else:
                logger.warning("Monitor Agent 未就绪，跳过监控快照")

        except Exception as e:
            logger.error(f"监控快照失败: {e}")
            if task:
                task.last_result = f"错误: {e}"
                task.error_count += 1
        finally:
            with self._lock:
                self._current_task = None

    def start(self):
        """启动调度器 (非阻塞，后台线程)"""
        if self._running:
            logger.warning("调度器已在运行")
            return

        if not self._tasks:
            logger.warning("没有已注册的定时任务，请先注册任务")
            return

        self._running = True
        self._stop_event.clear()  # 清除停机信号
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"✅ 调度器已启动 | {len(self._tasks)} 个任务")
        notifier.info("调度器启动", f"已注册 {len(self._tasks)} 个定时任务")

    def stop(self):
        """停止调度器 (优雅停机，等待当前任务完成)"""
        self._running = False
        self._stop_event.set()  # 唤醒等待中的循环

        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        schedule.clear()
        logger.info("⏹️ 调度器已停止")
        notifier.info("调度器停止", "所有定时任务已清除")

    def graceful_stop(self, timeout: float = 60) -> bool:
        """
        优雅停机: 通知调度器停止并等待当前任务完成

        Args:
            timeout: 最大等待秒数

        Returns:
            True 表示正常停止, False 表示超时
        """
        if not self._running:
            return True

        with self._lock:
            current = self._current_task

        if current:
            logger.info(f"🛑 优雅停机: 等待当前任务 [{current}] 完成 (超时 {timeout}s)...")

        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(f"⚠️ 优雅停机超时 ({timeout}s)，强制停止")
                self._thread = None
                return False
            self._thread = None

        schedule.clear()
        logger.info("✅ 调度器优雅停机完成")
        return True

    def _loop(self):
        """调度循环 (后台线程, 使用 Event 实现即时唤醒)"""
        while self._running:
            try:
                schedule.run_pending()
            except Exception as e:
                logger.error(f"调度循环异常: {e}")
            # 使用 Event.wait 替代 time.sleep，支持即时唤醒
            self._stop_event.wait(timeout=1)

    def status(self) -> dict:
        """获取调度器状态"""
        with self._lock:
            return {
                "running": self._running,
                "task_count": len(self._tasks),
                "tasks": [t.to_dict() for t in self._tasks.values()],
                "next_jobs": [
                    {
                        "job": str(job),
                        "next_run": job.next_run.strftime("%Y-%m-%d %H:%M:%S") if job.next_run else "未知",
                    }
                    for job in schedule.get_jobs()
                ],
            }

    def trigger_now(self, task_name: str) -> str:
        """手动触发指定任务"""
        task = self._tasks.get(task_name)
        if not task:
            return f"任务不存在: {task_name}"

        logger.info(f"👆 手动触发任务: {task_name}")
        try:
            task.func()
            return f"任务 {task_name} 已触发执行"
        except Exception as e:
            return f"任务 {task_name} 触发失败: {e}"

    def list_tasks(self) -> list[dict]:
        """列出所有任务"""
        with self._lock:
            return [t.to_dict() for t in self._tasks.values()]


# 全局实例
trading_scheduler = TradingScheduler()
