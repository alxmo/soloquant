"""
风控状态持久化 (SQLite)
=====================================================
将风控引擎的易失状态持久化到 SQLite，确保重启后恢复:
- 日交易次数 (_daily_trades)
- 峰值权益 (_peak_equity)
- Paper Trading 起始日期 (_paper_start_date)

数据库: data/risk_state.db
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from src.utils.config import config


class RiskStateDB:
    """风控状态 SQLite 持久化"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (config.DATA_DIR / "risk_state.db")
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS risk_state (
                        key   TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS daily_trades (
                        date TEXT PRIMARY KEY,
                        count INTEGER NOT NULL DEFAULT 0,
                        updated_at TEXT NOT NULL
                    )
                """)
                conn.commit()
                conn.close()
            logger.debug(f"风控状态数据库就绪: {self.db_path}")
        except Exception as e:
            logger.error(f"风控状态数据库初始化失败: {e}")

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ─── 日交易次数 ───

    def get_daily_trade_count(self, date_str: str) -> int:
        """获取指定日期的交易次数"""
        try:
            with self._lock:
                conn = self._get_conn()
                row = conn.execute(
                    "SELECT count FROM daily_trades WHERE date = ?", (date_str,)
                ).fetchone()
                conn.close()
                return row["count"] if row else 0
        except Exception as e:
            logger.error(f"读取日交易次数失败 [{date_str}]: {e}")
            return 0

    def set_daily_trade_count(self, date_str: str, count: int) -> None:
        """设置指定日期的交易次数"""
        try:
            with self._lock:
                conn = self._get_conn()
                conn.execute(
                    """INSERT INTO daily_trades (date, count, updated_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(date) DO UPDATE SET count=?, updated_at=?""",
                    (date_str, count, datetime.now().isoformat(), count, datetime.now().isoformat())
                )
                conn.commit()
                conn.close()
        except Exception as e:
            logger.error(f"保存日交易次数失败 [{date_str}]: {e}")

    def get_all_daily_trades(self) -> dict:
        """获取所有日交易记录"""
        try:
            with self._lock:
                conn = self._get_conn()
                rows = conn.execute("SELECT date, count FROM daily_trades").fetchall()
                conn.close()
                return {row["date"]: row["count"] for row in rows}
        except Exception as e:
            logger.error(f"读取日交易记录失败: {e}")
            return {}

    def cleanup_old_trades(self, keep_days: int = 30) -> int:
        """清理旧交易记录，只保留最近 N 天"""
        try:
            cutoff = (datetime.now().replace(hour=0, minute=0, second=0))
            from datetime import timedelta
            cutoff_str = (cutoff - timedelta(days=keep_days)).strftime("%Y-%m-%d")
            with self._lock:
                conn = self._get_conn()
                cursor = conn.execute(
                    "DELETE FROM daily_trades WHERE date < ?", (cutoff_str,)
                )
                deleted = cursor.rowcount
                conn.commit()
                conn.close()
            if deleted > 0:
                logger.info(f"清理 {deleted} 条过期日交易记录 (保留 {keep_days} 天)")
            return deleted
        except Exception as e:
            logger.error(f"清理日交易记录失败: {e}")
            return 0

    # ─── 通用键值存储 (peak_equity, paper_start_date 等) ───

    def get(self, key: str, default=None):
        """获取键值"""
        try:
            with self._lock:
                conn = self._get_conn()
                row = conn.execute(
                    "SELECT value FROM risk_state WHERE key = ?", (key,)
                ).fetchone()
                conn.close()
                if row:
                    return json.loads(row["value"])
                return default
        except Exception as e:
            logger.error(f"读取风控状态 [{key}] 失败: {e}")
            return default

    def set(self, key: str, value) -> None:
        """设置键值"""
        try:
            with self._lock:
                conn = self._get_conn()
                conn.execute(
                    """INSERT INTO risk_state (key, value, updated_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?""",
                    (key, json.dumps(value, default=str), datetime.now().isoformat(),
                     json.dumps(value, default=str), datetime.now().isoformat())
                )
                conn.commit()
                conn.close()
        except Exception as e:
            logger.error(f"保存风控状态 [{key}] 失败: {e}")

    # ─── 便捷方法 ───

    def get_peak_equity(self) -> float:
        """获取峰值权益"""
        return self.get("peak_equity", 0.0)

    def set_peak_equity(self, value: float) -> None:
        """设置峰值权益"""
        self.set("peak_equity", value)

    def get_paper_start_date(self) -> Optional[datetime]:
        """获取 Paper Trading 起始日期"""
        val = self.get("paper_start_date")
        if val:
            try:
                return datetime.fromisoformat(val)
            except (ValueError, TypeError):
                return None
        return None

    def set_paper_start_date(self, date: datetime) -> None:
        """设置 Paper Trading 起始日期"""
        self.set("paper_start_date", date.isoformat())

    def clear(self) -> None:
        """清空所有风控状态"""
        try:
            with self._lock:
                conn = self._get_conn()
                conn.execute("DELETE FROM risk_state")
                conn.execute("DELETE FROM daily_trades")
                conn.commit()
                conn.close()
            logger.info("风控状态已清空")
        except Exception as e:
            logger.error(f"清空风控状态失败: {e}")


# 全局实例
risk_state_db = RiskStateDB()
