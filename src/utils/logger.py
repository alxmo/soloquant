"""
日志系统 - 基于 loguru
=====================================================
T1.8: 日志系统
"""

import sys
from loguru import logger


# ─── 模块级默认 handler ───
# 在 config 加载等早期阶段就使用简洁格式，避免显示代码路径
logger.remove()  # 清除 loguru 默认 handler
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    colorize=True,
)


def setup_logger(debug=False):
    """配置全局日志

    Args:
        debug: 是否开启 debug 模式。
               True  → 控制台显示完整格式 (模块:函数:行号)
               False → 控制台显示简洁格式 (仅时间+级别+消息)
    """
    # 延迟导入 config，避免循环依赖和早期触发
    from src.utils.config import config

    logger.remove()  # 清除之前的 handler

    # 控制台输出
    if debug:
        # Debug 模式: 完整格式，包含代码路径
        console_format = (
            "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
        )
        console_level = "DEBUG"
    else:
        # 正常模式: 简洁格式，不显示代码路径
        console_format = (
            "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<level>{message}</level>"
        )
        console_level = config.LOG_LEVEL

    logger.add(
        sys.stderr,
        level=console_level,
        format=console_format,
        colorize=True,
    )

    # 文件输出
    log_file = config.LOGS_DIR / "Soloquant_{time:YYYY-MM-DD}.log"
    logger.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="00:00",       # 每天轮换
        retention="30 days",    # 保留30天
        encoding="utf-8",
    )

    # 错误日志单独文件
    error_file = config.LOGS_DIR / "error_{time:YYYY-MM-DD}.log"
    logger.add(
        str(error_file),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention="90 days",
        encoding="utf-8",
    )

    return logger
