"""
通知系统 - 多渠道告警推送
=====================================================
v0.3.0 增强:
- 控制台 + 文件持久化 (原有)
- Webhook 推送 (Discord / Slack / 飞书)
- 邮件通知 (SMTP)
- 告警分级: INFO / WARNING / CRITICAL
- 通知历史查询

T1.8: 通知系统
"""

import json
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Optional

from loguru import logger

from src.utils.config import config


class AlertLevel(str, Enum):
    """告警级别"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SUCCESS = "SUCCESS"


class WebhookChannel:
    """Webhook 推送渠道"""

    def __init__(self, url: str, name: str = "webhook"):
        self.url = url
        self.name = name

    def send(self, title: str, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        """发送 Webhook 通知"""
        try:
            import httpx
        except ImportError:
            logger.debug("httpx 未安装，跳过 Webhook 推送")
            return False

        # 根据平台格式化消息
        payload = self._format_payload(title, message, level)

        try:
            with httpx.Client(timeout=10) as client:
                resp = client.post(self.url, json=payload)
                if resp.status_code < 300:
                    logger.debug(f"Webhook [{self.name}] 推送成功")
                    return True
                else:
                    logger.warning(
                        f"Webhook [{self.name}] 推送失败: "
                        f"HTTP {resp.status_code}"
                    )
                    return False
        except Exception as e:
            logger.warning(f"Webhook [{self.name}] 推送异常: {e}")
            return False

    def _format_payload(self, title: str, message: str, level: AlertLevel) -> dict:
        """根据 Webhook URL 判断平台并格式化"""
        icons = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.SUCCESS: "✅",
        }
        icon = icons.get(level, "ℹ️")

        # Discord Webhook
        if "discord.com/api/webhooks" in self.url:
            return {
                "content": f"{icon} **[{level.value}] {title}**\n{message}"
            }

        # Slack Webhook
        if "hooks.slack.com" in self.url:
            color_map = {
                AlertLevel.INFO: "good",
                AlertLevel.WARNING: "warning",
                AlertLevel.ERROR: "danger",
                AlertLevel.CRITICAL: "#ff0000",
                AlertLevel.SUCCESS: "good",
            }
            return {
                "attachments": [{
                    "color": color_map.get(level, "good"),
                    "title": f"{icon} [{level.value}] {title}",
                    "text": message,
                    "ts": int(datetime.now().timestamp()),
                }]
            }

        # 飞书 Webhook
        if "open.feishu.cn" in self.url:
            return {
                "msg_type": "text",
                "content": {
                    "text": f"{icon} [{level.value}] {title}\n{message}"
                },
            }

        # 通用 JSON 格式
        return {
            "title": f"[{level.value}] {title}",
            "message": message,
            "level": level.value,
            "timestamp": datetime.now().isoformat(),
        }


class EmailChannel:
    """邮件通知渠道"""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: list[str],
        use_tls: bool = True,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls

    def send(self, title: str, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        """发送邮件通知"""
        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)
            msg["Subject"] = f"[{level.value}] {title}"

            # 纯文本
            text_part = MIMEText(message, "plain", "utf-8")
            msg.attach(text_part)

            # HTML 版本
            color_map = {
                AlertLevel.INFO: "#17a2b8",
                AlertLevel.WARNING: "#ffc107",
                AlertLevel.ERROR: "#dc3545",
                AlertLevel.CRITICAL: "#dc3545",
                AlertLevel.SUCCESS: "#28a745",
            }
            color = color_map.get(level, "#17a2b8")
            html = f"""
            <html><body>
            <h2 style="color:{color}">[{level.value}] {title}</h2>
            <pre style="font-size:14px;white-space:pre-wrap;">{message}</pre>
            <hr><p style="color:#999;font-size:12px;">
            AI 量化交易系统 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
            </body></html>
            """
            html_part = MIMEText(html, "html", "utf-8")
            msg.attach(html_part)

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            logger.debug(f"邮件通知已发送至 {self.to_addrs}")
            return True

        except Exception as e:
            logger.warning(f"邮件通知发送失败: {e}")
            return False


class Notifier:
    """通知管理器 - 多渠道告警"""

    def __init__(self):
        self.history_file = config.DATA_DIR / "notifications.jsonl"
        self._webhooks: list[WebhookChannel] = []
        self._email_channel: Optional[EmailChannel] = None
        self._load_channels()

    def _load_channels(self):
        """从环境变量加载通知渠道"""
        # Webhook (支持多个, 逗号分隔)
        webhook_url = os.getenv("NOTIFY_WEBHOOK_URL", "")
        if webhook_url:
            for url in webhook_url.split(","):
                url = url.strip()
                if url:
                    self.add_webhook(url, name=f"env_{len(self._webhooks)}")

        # 邮件
        smtp_host = os.getenv("NOTIFY_SMTP_HOST", "")
        if smtp_host:
            try:
                port = int(os.getenv("NOTIFY_SMTP_PORT", "587"))
                username = os.getenv("NOTIFY_SMTP_USER", "")
                password = os.getenv("NOTIFY_SMTP_PASS", "")
                from_addr = os.getenv("NOTIFY_EMAIL_FROM", username)
                to_addrs_str = os.getenv("NOTIFY_EMAIL_TO", "")
                to_addrs = [a.strip() for a in to_addrs_str.split(",") if a.strip()]
                use_tls = os.getenv("NOTIFY_SMTP_TLS", "true").lower() == "true"

                if username and to_addrs:
                    self.set_email_channel(
                        smtp_host=smtp_host,
                        smtp_port=port,
                        username=username,
                        password=password,
                        from_addr=from_addr,
                        to_addrs=to_addrs,
                        use_tls=use_tls,
                    )
            except Exception as e:
                logger.warning(f"邮件渠道加载失败: {e}")

    def add_webhook(self, url: str, name: str = "webhook"):
        """添加 Webhook 推送渠道"""
        channel = WebhookChannel(url, name)
        self._webhooks.append(channel)
        logger.info(f"Webhook 渠道已添加: {name} -> {url}")

    def set_email_channel(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: list[str],
        use_tls: bool = True,
    ):
        """设置邮件通知渠道"""
        self._email_channel = EmailChannel(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            username=username,
            password=password,
            from_addr=from_addr,
            to_addrs=to_addrs,
            use_tls=use_tls,
        )
        logger.info(f"邮件渠道已设置: {from_addr} -> {to_addrs}")

    def send(
        self,
        title: str,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
    ):
        """广播通知至所有渠道"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. 控制台输出
        icons = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.SUCCESS: "✅",
        }
        icon = icons.get(level, "ℹ️")
        print(f"\n{icon} [{level.value}] {title}")
        print(f"   {message}\n")

        # 2. 日志记录
        log_msg = f"[NOTIFY] {title}: {message}"
        if level == AlertLevel.CRITICAL:
            logger.critical(log_msg)
        elif level == AlertLevel.ERROR:
            logger.error(log_msg)
        elif level == AlertLevel.WARNING:
            logger.warning(log_msg)
        elif level == AlertLevel.SUCCESS:
            logger.success(log_msg)
        else:
            logger.info(log_msg)

        # 3. Webhook 推送
        for wh in self._webhooks:
            try:
                wh.send(title, message, level)
            except Exception as e:
                logger.warning(f"Webhook [{wh.name}] 推送异常: {e}")

        # 4. 邮件推送
        if self._email_channel:
            try:
                self._email_channel.send(title, message, level)
            except Exception as e:
                logger.warning(f"邮件推送异常: {e}")

        # 5. 持久化到文件
        record = {
            "timestamp": timestamp,
            "level": level.value,
            "title": title,
            "message": message,
        }
        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"通知持久化失败: {e}")

    def get_history(self, limit: int = 50, level: Optional[AlertLevel] = None) -> list[dict]:
        """查询通知历史"""
        if not self.history_file.exists():
            return []

        records = []
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if level and record.get("level") != level.value:
                            continue
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"读取通知历史失败: {e}")
            return []

        return records[-limit:]

    def info(self, title: str, message: str):
        self.send(title, message, AlertLevel.INFO)

    def warning(self, title: str, message: str):
        self.send(title, message, AlertLevel.WARNING)

    def error(self, title: str, message: str):
        self.send(title, message, AlertLevel.ERROR)

    def critical(self, title: str, message: str):
        self.send(title, message, AlertLevel.CRITICAL)

    def success(self, title: str, message: str):
        self.send(title, message, AlertLevel.SUCCESS)


# 全局实例
notifier = Notifier()
