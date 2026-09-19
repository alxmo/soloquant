"""
Agent 基类 - 所有量化交易 Agent 的公共基础设施
=====================================================
提供:
- 统一的 Agent 上下文管理
- 工具调用路由 (调用底层引擎模块)
- 通俗解读生成 (小白友好)
- 执行日志与结果追踪
"""

import functools
from datetime import datetime
from typing import Any, Optional

from loguru import logger


# ─── 哨兵值 ───
# 用于区分"调用方未传参"和"调用方显式传了 None"
# 不传参时回退到 _last_result；显式传 None 表示"无结果"
_NO_RESULT = object()


class AgentContext:
    """Agent 间共享上下文 - 传递中间结果"""

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._history: list[dict] = []  # 执行历史

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "key": key,
            "type": type(value).__name__,
        })

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._data

    def summary(self) -> dict:
        """返回上下文摘要"""
        return {
            "keys": list(self._data.keys()),
            "history_count": len(self._history),
            "history": self._history[-10:],  # 最近10条
        }


class BaseAgent:
    """
    量化交易 Agent 基类

    每个 Agent 负责交易链路中的一个环节:
    Research -> Strategy -> Backtest -> Execution -> Monitor

    子类可通过两种方式实现执行逻辑:
    1. 覆盖 _execute() —— 由基类 run() 调用，自动跟踪结果和异常
    2. 覆盖 run() —— 直接控制执行流程，__init_subclass__ 自动包装以跟踪结果和异常
    """

    def __init_subclass__(cls, **kwargs):
        """
        自动包装子类的 run() 方法，确保 _last_result / _last_error 被正确跟踪。

        当子类直接覆盖 run() 时，原始 run() 不会调用基类 run() 的跟踪逻辑，
        因此在此处自动注入结果跟踪和异常处理。
        """
        super().__init_subclass__(**kwargs)
        if 'run' in cls.__dict__:
            original_run = cls.__dict__['run']

            @functools.wraps(original_run)
            def wrapped_run(self, *args, **kw):
                try:
                    result = original_run(self, *args, **kw)
                    self._last_result = result
                    self._last_error = None
                    if self.context:
                        self.context.set(f"{self.name}_result", result)
                    return result
                except Exception as e:
                    self._last_error = str(e)
                    if self.context:
                        self.context.set(f"{self.name}_error", str(e))
                    raise

            cls.run = wrapped_run

    def __init__(self, name: str, role: str):
        self.name = name          # Agent 标识 (如 "research")
        self.role = role          # 角色描述 (如 "市场调研员")
        self.context: Optional[AgentContext] = None
        self._ctx: Optional[AgentContext] = None  # 别名，兼容子类直接访问
        self._last_result: Any = None
        self._last_error: Optional[str] = None

    def bind_context(self, ctx: AgentContext) -> None:
        """绑定共享上下文"""
        self.context = ctx
        self._ctx = ctx

    def last_result(self) -> Any:
        """返回上次执行的结果"""
        return self._last_result

    def last_error(self) -> Optional[str]:
        """返回上次执行的错误信息"""
        return self._last_error

    def run(self, **kwargs) -> Any:
        """
        执行 Agent 任务 (同步入口)

        基类 run() 调用 _execute()，子类可覆盖 _execute() 或直接覆盖 run()。
        异常会传播给调用方，同时记录到 _last_error。
        """
        logger.info(f"🤖 [{self.name}] {self.role} 开始执行...")
        start_time = datetime.now()

        try:
            result = self._execute(**kwargs)
            self._last_result = result
            self._last_error = None

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ [{self.name}] 执行完成 ({elapsed:.1f}s)")

            # 结果写入上下文
            if self.context:
                self.context.set(f"{self.name}_result", result)
                self.context.set(f"{self.name}_time", elapsed)

            return result

        except Exception as e:
            self._last_error = str(e)
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ [{self.name}] 执行失败 ({elapsed:.1f}s): {e}")
            raise

    def _execute(self, **kwargs) -> Any:
        """子类实现具体逻辑"""
        raise NotImplementedError(f"{self.name} 未实现 _execute()")

    def plain_explain(self, result: Any = _NO_RESULT) -> str:
        """
        生成通俗解读 (子类覆盖)

        将专业结果翻译成小白能懂的语言

        Args:
            result: 要解读的结果。
                    - 不传参 (默认哨兵): 回退到 _last_result
                    - 显式传 None: 视为"无结果"，返回暂无结果提示
                    - 传其他值: 直接解读该值
        """
        if result is _NO_RESULT:
            # 调用方未传参，回退到上次执行结果
            result = self._last_result
        if result is None:
            return f"[{self.role}] 暂无结果"
        return f"[{self.role}] 执行完成，结果已生成。"

    def status(self) -> dict:
        """返回 Agent 状态"""
        return {
            "name": self.name,
            "role": self.role,
            "has_result": self._last_result is not None,
            "has_error": self._last_error is not None,
            "error": self._last_error,
        }
