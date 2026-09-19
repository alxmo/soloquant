"""
通用并行执行工具
=====================================================
基于 ThreadPoolExecutor 封装，提供:
- parallel_map: 并行映射，异常隔离
- parallel_fetch: 并行数据获取，带结果过滤
- 进度日志与耗时统计

设计原则:
- 单个任务异常不影响其他任务
- 返回结果与输入顺序对应（None 占位失败项）
- 线程数可配置，默认 min(4, len(items))
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from loguru import logger


def parallel_map(
    func: Callable,
    items: list,
    max_workers: int = 4,
    desc: str = "并行执行",
    timeout: float = 120,
) -> list:
    """
    并行执行 func(item) 并返回结果列表

    - 异常隔离: 单个 item 失败返回 None，不中断其他任务
    - 结果顺序: 与输入 items 顺序一致
    - 自动跳过: items 为空时直接返回空列表

    Args:
        func: 单参数函数，签名 func(item) -> result
        items: 输入列表
        max_workers: 最大线程数
        desc: 日志描述
        timeout: 整体超时秒数

    Returns:
        list: 结果列表，失败项为 None
    """
    if not items:
        return []

    n = len(items)
    workers = min(max_workers, n)
    results: list[Any] = [None] * n  # 预分配，保证顺序

    start = time.time()
    logger.info(f"⚡ {desc}: {n} 项, {workers} 线程")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 使用 index 映射，保证结果顺序
        future_to_idx = {
            executor.submit(func, item): idx
            for idx, item in enumerate(items)
        }

        done_count = 0
        for future in as_completed(future_to_idx, timeout=timeout):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.debug(f"{desc} 第 {idx+1}/{n} 项失败: {e}")
                results[idx] = None
            done_count += 1
            if done_count % 10 == 0 or done_count == n:
                logger.info(f"  {desc} 进度: {done_count}/{n}")

    elapsed = time.time() - start
    success = sum(1 for r in results if r is not None)
    logger.info(f"✅ {desc} 完成: {success}/{n} 成功 ({elapsed:.1f}s)")
    return results


def parallel_fetch(
    func: Callable,
    items: list,
    max_workers: int = 4,
    desc: str = "并行获取",
    filter_none: bool = True,
    timeout: float = 120,
) -> dict:
    """
    并行获取数据，返回 {item: result} 字典

    适用于: 批量获取股票数据、API 查询等场景

    Args:
        func: 单参数函数，签名 func(item) -> result
        items: 输入列表 (如股票代码)
        max_workers: 最大线程数
        desc: 日志描述
        filter_none: 是否过滤掉 None 结果
        timeout: 整体超时秒数

    Returns:
        dict: {item: result}，失败项不包含
    """
    results_raw = parallel_map(func, items, max_workers, desc, timeout)

    result_map: dict = {}
    for item, result in zip(items, results_raw):
        if result is not None or not filter_none:
            result_map[item] = result
    return result_map


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: float = 30,
) -> Optional[Any]:
    """
    在线程中执行函数，带超时保护

    适用于: 可能卡住的外部 API 调用

    Args:
        func: 要执行的函数
        args: 位置参数
        kwargs: 关键字参数
        timeout: 超时秒数

    Returns:
        函数返回值，超时返回 None
    """
    if kwargs is None:
        kwargs = {}

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            logger.warning(f"函数执行超时 ({timeout}s): {func.__name__}")
            return None
