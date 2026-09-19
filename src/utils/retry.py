"""
API 重试与断路器
=====================================================
提供:
- @retry: 指数退避重试装饰器
- CircuitBreaker: 熔断器，连续失败后暂停调用

设计原则:
- 透明: 装饰器不改变函数签名
- 可配: 重试次数、退避策略、可重试异常类型
- 隔离: 断路器保护，防止故障扩散
"""

import functools
import random
import threading
import time
from typing import Any, Callable, Optional, Type, Tuple

from loguru import logger


# ─── 重试装饰器 ───

def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None,
) -> Callable:
    """
    指数退避重试装饰器

    Args:
        max_retries: 最大重试次数 (不含首次调用)
        base_delay: 首次重试延迟 (秒)
        max_delay: 最大延迟 (秒)
        backoff_factor: 退避因子 (delay *= backoff_factor)
        jitter: 是否添加随机抖动 (避免重试风暴)
        retryable_exceptions: 可重试的异常类型
        on_retry: 重试回调 fn(func_name, attempt, exception)

    Returns:
        装饰后的函数

    用法:
        @retry(max_retries=3, base_delay=0.5)
        def fetch_data(symbol):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            func_name = getattr(func, "__name__", str(func))
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt >= max_retries:
                        logger.error(
                            f"🔁 {func_name} 重试 {max_retries} 次后仍失败: {e}"
                        )
                        raise

                    # 计算延迟
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)

                    logger.warning(
                        f"🔁 {func_name} 第 {attempt+1}/{max_retries} 次重试 "
                        f"({delay:.1f}s 后): {e}"
                    )

                    if on_retry:
                        try:
                            on_retry(func_name, attempt + 1, e)
                        except Exception:
                            pass

                    time.sleep(delay)

            raise last_exception  # 理论上不会执行到这里

        return wrapper
    return decorator


def retry_on_network_error(
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> Callable:
    """
    专门针对网络错误的重试装饰器 (便捷方法)

    只重试网络相关异常，不重试业务逻辑错误
    """
    # 常见网络异常
    network_exceptions: list = []
    for exc_name in [
        "ConnectionError", "TimeoutError", "OSError",
        "ConnectionResetError", "ConnectionAbortedError",
        "RemoteDisconnected",
    ]:
        exc = globals().get(exc_name)
        if exc is None:
            # 尝试从 builtins 获取
            try:
                exc = eval(exc_name)
            except Exception:
                pass
        if exc is not None:
            network_exceptions.append(exc)

    # http.client.RemoteDisconnected
    try:
        from http.client import RemoteDisconnected
        network_exceptions.append(RemoteDisconnected)
    except ImportError:
        pass

    # urllib3.exceptions
    try:
        from urllib3.exceptions import MaxRetryError, ProtocolError
        network_exceptions.extend([MaxRetryError, ProtocolError])
    except ImportError:
        pass

    # requests.exceptions
    try:
        from requests.exceptions import (
            ConnectionError as RequestsConnectionError,
            Timeout as RequestsTimeout,
        )
        network_exceptions.extend([RequestsConnectionError, RequestsTimeout])
    except ImportError:
        pass

    if not network_exceptions:
        network_exceptions = [ConnectionError, TimeoutError, OSError]

    return retry(
        max_retries=max_retries,
        base_delay=base_delay,
        retryable_exceptions=tuple(network_exceptions),
    )


# ─── 断路器 ───

class CircuitState:
    """断路器状态"""
    CLOSED = "closed"      # 正常，允许调用
    OPEN = "open"          # 熔断，拒绝调用
    HALF_OPEN = "half_open"  # 半开，允许探测


class CircuitBreaker:
    """
    熔断器

    连续失败 threshold 次后熔断，冷却 recovery_timeout 后进入半开状态，
    半开状态下允许一次探测调用，成功则恢复，失败则继续熔断。

    用法:
        breaker = CircuitBreaker(name="finnhub", threshold=5, recovery_timeout=60)

        @breaker.protect
        def call_api():
            ...
    """

    def __init__(
        self,
        name: str = "default",
        threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        self.name = name
        self.threshold = threshold  # 连续失败多少次熔断
        self.recovery_timeout = recovery_timeout  # 熔断后冷却秒数

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # 检查是否已过冷却期
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(f"🔌 断路器 [{self.name}] 进入半开状态 (探测)")
            return self._state

    def _before_call(self) -> bool:
        """调用前检查: 返回 True 表示允许调用"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(f"🔌 断路器 [{self.name}] 半开探测中...")
                    return True
                return False  # 熔断中
            return True

    def _on_success(self):
        """调用成功"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(f"✅ 断路器 [{self.name}] 恢复正常")
            self._state = CircuitState.CLOSED
            self._failure_count = 0

    def _on_failure(self):
        """调用失败"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # 半开状态下失败，重新熔断
                self._state = CircuitState.OPEN
                logger.warning(f"🔌 断路器 [{self.name}] 半开探测失败，重新熔断")
            elif self._failure_count >= self.threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"🔌 断路器 [{self.name}] 熔断! "
                    f"连续失败 {self._failure_count} 次, 冷却 {self.recovery_timeout}s"
                )

    def protect(self, func: Callable) -> Callable:
        """装饰器: 保护函数调用"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self._before_call():
                raise CircuitBreakerOpenError(
                    f"断路器 [{self.name}] 已熔断, 请稍后重试"
                )
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """直接调用 (非装饰器方式)"""
        if not self._before_call():
            raise CircuitBreakerOpenError(
                f"断路器 [{self.name}] 已熔断, 请稍后重试"
            )
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def reset(self):
        """手动重置断路器"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            logger.info(f"🔌 断路器 [{self.name}] 已手动重置")

    def status(self) -> dict:
        """获取断路器状态"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state,
                "failure_count": self._failure_count,
                "threshold": self.threshold,
                "recovery_timeout": self.recovery_timeout,
                "last_failure_time": self._last_failure_time,
            }


class CircuitBreakerOpenError(Exception):
    """断路器已熔断异常"""
    pass


# ─── 全局断路器注册表 ───

_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str, **kwargs) -> CircuitBreaker:
    """获取或创建断路器 (单例)"""
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name=name, **kwargs)
    return _breakers[name]
