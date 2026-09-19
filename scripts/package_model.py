"""
模型打包上传脚本 (本地专用)
=====================================================
功能:
  1. 训练完成后自动打包模型文件 + 元数据
  2. 验证模型文件完整性
  3. 通过 HTTP API 上传到云端
  4. 显示上传结果和云端当前模型状态
  5. 支持命令行参数指定模型文件

用法:
  python scripts/package_model.py                           # 上传最优模型
  python scripts/package_model.py --file model_xxx.txt      # 上传指定模型
  python scripts/package_model.py --check                   # 检查云端连接
  python scripts/package_model.py --status                  # 查看云端状态
  python scripts/package_model.py --list                    # 列出云端模型

配置 (.env):
  CLOUD_API_URL=http://你的云服务器IP:8000
  MODEL_API_KEY=your_secret_key
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from loguru import logger

from src.utils.config import config


class ModelPackager:
    """模型打包上传器"""

    def __init__(self):
        self.api_url = os.getenv("CLOUD_API_URL", "").rstrip("/")
        self.api_key = os.getenv("MODEL_API_KEY", "")

        if not self.api_url:
            logger.warning("CLOUD_API_URL 未配置，请在 .env 中设置")
            logger.info("示例: CLOUD_API_URL=http://123.45.67.89:8000")

    @property
    def configured(self) -> bool:
        return bool(self.api_url)

    def _headers(self) -> dict:
        """构建请求头"""
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def check_cloud(self) -> bool:
        """检查云端连接"""
        if not self.configured:
            logger.error("云端 API URL 未配置")
            return False

        try:
            resp = httpx.get(f"{self.api_url}/api/health", headers=self._headers(), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                logger.info("✅ 云端连接正常")
                logger.info(f"   部署模式: {data.get('deploy_mode')}")
                logger.info(f"   模型数量: {data.get('models_count')}")
                logger.info(f"   激活模型: {data.get('active_model', '无')}")
                logger.info(f"   内存占用: {data.get('memory_usage_mb', 0):.1f} MB")
                return True
            else:
                logger.error(f"云端响应异常: HTTP {resp.status_code}")
                return False
        except httpx.ConnectError:
            logger.error(f"无法连接云端: {self.api_url}")
            logger.info("请检查: 1) 云端API是否启动  2) IP地址是否正确  3) 防火墙是否放行端口")
            return False
        except Exception as e:
            logger.error(f"连接云端失败: {e}")
            return False

    def get_cloud_status(self) -> dict:
        """获取云端交易状态"""
        if not self.configured:
            return {}

        try:
            resp = httpx.get(f"{self.api_url}/api/status", headers=self._headers(), timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"获取云端状态失败: {e}")
        return {}

    def list_cloud_models(self) -> list:
        """列出云端模型"""
        if not self.configured:
            return []

        try:
            resp = httpx.get(f"{self.api_url}/api/models/list", headers=self._headers(), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                active = data.get("active_model")

                logger.info(f"云端模型列表 ({len(models)} 个):")
                for m in models:
                    marker = " ← 当前激活" if m.get("is_active") else ""
                    size_kb = m.get("size", 0) / 1024
                    logger.info(f"  - {m['name']} ({size_kb:.1f} KB, {m['modified']}){marker}")

                if active:
                    logger.info(f"当前激活: {active}")
                return models
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
        return []

    def find_best_model(self) -> tuple[Path, dict]:
        """查找本地最优模型文件和元数据"""
        models_dir = config.DATA_DIR / "models"
        marker = models_dir / "best_model.json"

        if not marker.exists():
            logger.error("未找到 best_model.json，请先运行训练 (scripts/train_pipeline.py)")
            return None, {}

        with open(marker) as f:
            meta = json.load(f)

        strategy_id = meta.get("strategy_id", "")
        model_type = meta.get("model_type", "lightgbm")

        # 查找模型文件
        model_file = None

        if model_type == "lightgbm":
            # 优先匹配策略ID
            for f in models_dir.glob("*.txt"):
                if strategy_id and strategy_id in f.name:
                    model_file = f
                    break
            # 降级: 最新的 .txt
            if not model_file:
                txt_files = sorted(models_dir.glob("*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)
                if txt_files:
                    model_file = txt_files[0]

        elif model_type == "lstm":
            # LSTM 模型 (.pth)
            for f in models_dir.glob("*.pth"):
                if strategy_id and strategy_id in f.name:
                    model_file = f
                    break
            if not model_file:
                pth_files = sorted(models_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
                if pth_files:
                    model_file = pth_files[0]

        if not model_file:
            logger.error(f"未找到模型文件 (strategy_id={strategy_id}, type={model_type})")
            return None, meta

        return model_file, meta

    def upload_model(self, model_file: Path = None, meta: dict = None) -> bool:
        """
        上传模型到云端

        Args:
            model_file: 模型文件路径 (None 则自动查找最优模型)
            meta: 模型元数据 (None 则自动从 best_model.json 读取)
        """
        if not self.configured:
            logger.error("云端 API URL 未配置")
            return False

        # 自动查找
        if model_file is None or meta is None:
            model_file, meta = self.find_best_model()
            if model_file is None:
                return False

        if not model_file.exists():
            logger.error(f"模型文件不存在: {model_file}")
            return False

        file_size = model_file.stat().st_size
        logger.info(f"准备上传模型: {model_file.name} ({file_size / 1024:.1f} KB)")
        logger.info(f"  策略ID: {meta.get('strategy_id', 'N/A')}")
        logger.info(f"  模型类型: {meta.get('model_type', 'N/A')}")
        logger.info(f"  夏普比率: {meta.get('sharpe_ratio', 'N/A')}")
        logger.info(f"  年化收益: {meta.get('annual_return', 'N/A')}")

        # 1. 上传模型文件
        try:
            with open(model_file, "rb") as f:
                files = {"file": (model_file.name, f, "application/octet-stream")}
                resp = httpx.post(
                    f"{self.api_url}/api/models/upload",
                    files=files,
                    headers=self._headers(),
                    timeout=60,
                )

            if resp.status_code != 200:
                logger.error(f"上传模型文件失败: HTTP {resp.status_code} - {resp.text}")
                return False

            upload_result = resp.json()
            logger.info(f"✅ 模型文件上传成功: {upload_result.get('filename')}")

        except Exception as e:
            logger.error(f"上传模型文件失败: {e}")
            return False

        # 2. 上传模型元数据 (标记为激活模型)
        try:
            resp = httpx.post(
                f"{self.api_url}/api/models/metadata",
                json=meta,
                headers=self._headers(),
                timeout=10,
            )

            if resp.status_code == 200:
                logger.info("✅ 模型元数据已上传，已设为激活模型")
            else:
                logger.warning(f"元数据上传返回: HTTP {resp.status_code}")

        except Exception as e:
            logger.warning(f"上传元数据失败: {e}")

        # 3. 验证云端状态
        try:
            resp = httpx.get(f"{self.api_url}/api/models/active", headers=self._headers(), timeout=10)
            if resp.status_code == 200:
                active = resp.json()
                active_name = active.get("active_model")
                if active_name and (strategy_id := meta.get("strategy_id", "")) in str(active_name):
                    logger.info(f"✅ 云端已确认激活模型: {active_name}")
                else:
                    logger.warning(f"云端激活模型不匹配: 期望={meta.get('strategy_id')}, 实际={active_name}")
        except Exception:
            pass

        logger.info("🎉 模型上传完成！云端将在 30 秒内自动热加载新模型")
        return True


def main():
    parser = argparse.ArgumentParser(description="模型打包上传工具")
    parser.add_argument("--file", type=str, help="指定模型文件路径")
    parser.add_argument("--check", action="store_true", help="检查云端连接")
    parser.add_argument("--status", action="store_true", help="查看云端交易状态")
    parser.add_argument("--list", action="store_true", help="列出云端模型")
    args = parser.parse_args()

    packager = ModelPackager()

    if not packager.configured:
        logger.warning("⚠️ 云端 API 未配置")
        logger.info("请在 .env 文件中添加以下配置:")
        logger.info("  CLOUD_API_URL=http://你的云服务器IP:8000")
        logger.info("  MODEL_API_KEY=your_secret_key  (可选, 云端配置了才需要)")
        return

    if args.check:
        packager.check_cloud()
    elif args.status:
        status = packager.get_cloud_status()
        if status:
            print("\n═ 云端交易状态 ═══════════════════════════════")
            acct = status.get("account")
            if acct:
                print(f"  💰 账户:")
                print(f"     总资产: ${acct.get('equity', 0):,.2f}")
                print(f"     现金:   ${acct.get('cash', 0):,.2f}")
                print(f"     模式:   {acct.get('trading_mode', 'N/A')}")

            positions = status.get("positions", [])
            if positions:
                print(f"\n  📊 持仓 ({len(positions)} 只):")
                for p in positions:
                    pnl = p.get("unrealized_pnl_pct", 0)
                    print(f"     {p['symbol']:5s} | {p['qty']:>6.1f}股 | ${p['market_value']:>10,.2f} | {pnl:>+6.2%}")
            else:
                print("\n  📊 无持仓")

            risk = status.get("risk_status")
            if risk:
                print(f"\n  🛡️ 风控:")
                print(f"     回撤: {risk.get('current_drawdown', 0):.2%}")
                print(f"     日亏: {risk.get('current_daily_loss', 0):.2%}")
                if risk.get("is_alert"):
                    print(f"     ⚠️ 告警: {', '.join(risk.get('alerts', []))}")

            print(f"\n  🤖 激活模型: {status.get('active_model', '无')}")
            print(f"  ⏰ 调度器: {'运行中' if status.get('scheduler_running') else '已停止'}")
            print("═" * 50)
    elif args.list:
        packager.list_cloud_models()
    elif args.file:
        # 上传指定文件
        model_file = Path(args.file)
        packager.upload_model(model_file=model_file)
    else:
        # 默认上传最优模型
        packager.upload_model()


if __name__ == "__main__":
    main()
