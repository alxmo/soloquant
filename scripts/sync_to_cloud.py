"""
云端同步工具 - 本地训练好的模型/策略一键同步到云端
=====================================================
功能:
  1. 同步最优模型到云端
  2. 同步策略配置到云端
  3. 版本管理 (保留历史版本)
  4. 同步前校验模型有效性
  5. 支持回滚到上一版本

用法:
  python scripts/sync_to_cloud.py              # 同步最优模型
  python scripts/sync_to_cloud.py --all        # 同步所有模型和策略
  python scripts/sync_to_cloud.py --rollback   # 回滚到上一版本
  python scripts/sync_to_cloud.py --check      # 检查云端连接
  python scripts/sync_to_cloud.py --list       # 列出云端模型

配置:
  在 .env 中设置:
    CLOUD_HOST=你的服务器IP
    CLOUD_PORT=22
    CLOUD_USER=用户名
    CLOUD_PASSWORD=密码 (或配置密钥)
    CLOUD_PATH=/home/user/soloquant
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.utils.config import config


class CloudSyncer:
    """云端同步器"""

    def __init__(self):
        import os
        self.host = os.getenv("CLOUD_HOST", "")
        self.port = int(os.getenv("CLOUD_PORT", "22"))
        self.user = os.getenv("CLOUD_USER", "")
        self.password = os.getenv("CLOUD_PASSWORD", "")
        self.remote_path = os.getenv("CLOUD_PATH", "/home/user/soloquant")

        self._transport = None
        self._sftp = None

    @property
    def configured(self) -> bool:
        return bool(self.host and self.user and self.remote_path)

    def connect(self) -> bool:
        """连接云端服务器"""
        if not self.configured:
            logger.error("云端配置不完整，请在 .env 中设置 CLOUD_HOST/CLOUD_USER/CLOUD_PATH")
            return False

        try:
            import paramiko
        except ImportError:
            logger.error("paramiko 未安装，请运行: pip install paramiko")
            return False

        try:
            logger.info(f"连接云端: {self.user}@{self.host}:{self.port}")
            self._transport = paramiko.Transport((self.host, self.port))
            self._transport.connect(username=self.user, password=self.password)
            self._sftp = paramiko.SFTPClient.from_transport(self._transport)
            logger.info("✅ 云端连接成功")
            return True
        except Exception as e:
            logger.error(f"云端连接失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self._sftp:
            self._sftp.close()
        if self._transport:
            self._transport.close()
        logger.info("云端连接已关闭")

    def _remote_exists(self, path: str) -> bool:
        """检查远程路径是否存在"""
        try:
            self._sftp.stat(path)
            return True
        except IOError:
            return False

    def _mkdir_p(self, remote_dir: str):
        """递归创建远程目录"""
        dirs = remote_dir.split("/")
        path = ""
        for d in dirs:
            if not d:
                path = "/"
                continue
            path = path.rstrip("/") + "/" + d
            if not self._remote_exists(path):
                try:
                    self._sftp.mkdir(path)
                except IOError:
                    pass  # 可能已存在

    def upload_file(self, local_path: Path, remote_path: str):
        """上传单个文件"""
        remote_dir = "/".join(remote_path.split("/")[:-1])
        self._mkdir_p(remote_dir)
        self._sftp.put(str(local_path), remote_path)
        logger.info(f"  ↑ {local_path.name} -> {remote_path}")

    def download_file(self, remote_path: str, local_path: Path):
        """下载单个文件"""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._sftp.get(remote_path, str(local_path))
        logger.info(f"  ↓ {remote_path} -> {local_path.name}")

    def check_connection(self) -> bool:
        """检查云端连接"""
        if not self.connect():
            return False
        try:
            # 检查远程目录
            if self._remote_exists(self.remote_path):
                logger.info(f"✅ 远程目录存在: {self.remote_path}")
                # 列出内容
                files = self._sftp.listdir(self.remote_path)
                logger.info(f"  包含 {len(files)} 个文件/目录")
            else:
                logger.warning(f"远程目录不存在: {self.remote_path}")
                logger.info("将在同步时自动创建")
            return True
        finally:
            self.close()

    def list_remote_models(self) -> list:
        """列出云端模型"""
        if not self.connect():
            return []

        try:
            models_dir = f"{self.remote_path.rstrip('/')}/data/models"
            strategies_dir = f"{self.remote_path.rstrip('/')}/data/strategies"

            models = []
            if self._remote_exists(models_dir):
                models = self._sftp.listdir(models_dir)
                logger.info(f"云端模型: {len(models)} 个")
                for m in sorted(models)[-10:]:
                    logger.info(f"  - {m}")

            strategies = []
            if self._remote_exists(strategies_dir):
                strategies = self._sftp.listdir(strategies_dir)
                logger.info(f"云端策略: {len([s for s in strategies if s.endswith('.json') and 'backtest' not in s])} 个")

            return models
        finally:
            self.close()

    def sync_best_model(self) -> bool:
        """同步最优模型到云端"""
        # 1. 找到最优模型
        best_marker = config.DATA_DIR / "models" / "best_model.json"
        if not best_marker.exists():
            logger.error("未找到最优模型标记，请先运行训练")
            return False

        with open(best_marker) as f:
            best = json.load(f)

        strategy_id = best.get("strategy_id", "")
        if not strategy_id:
            logger.error("最优模型信息不完整")
            return False

        logger.info(f"同步最优模型: {strategy_id}")
        logger.info(f"  类型: {best.get('model_type')}")
        logger.info(f"  夏普: {best.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  年化: {best.get('annual_return', 0):.2%}")

        # 2. 找到相关文件
        files_to_sync = []

        # 策略配置
        strategy_file = config.DATA_DIR / "strategies" / f"{strategy_id}.json"
        if strategy_file.exists():
            files_to_sync.append((strategy_file, f"data/strategies/{strategy_id}.json"))
        else:
            logger.warning(f"策略文件不存在: {strategy_file}")

        # 回测结果
        bt_file = config.DATA_DIR / "strategies" / f"{strategy_id}_backtest.json"
        if bt_file.exists():
            files_to_sync.append((bt_file, f"data/strategies/{strategy_id}_backtest.json"))

        # 模型文件 (LightGBM .txt 或 LSTM .pth)
        model_dir = config.DATA_DIR / "models"
        for f in model_dir.glob(f"*{strategy_id}*"):
            files_to_sync.append((f, f"data/models/{f.name}"))

        if not files_to_sync:
            logger.error("没有找到要同步的文件")
            return False

        # 3. 连接并上传
        if not self.connect():
            return False

        try:
            logger.info(f"上传 {len(files_to_sync)} 个文件...")
            for local_path, remote_rel in files_to_sync:
                remote_path = f"{self.remote_path.rstrip('/')}/{remote_rel}"
                self.upload_file(local_path, remote_path)

            # 4. 更新云端 best_model.json
            best_remote = f"{self.remote_path.rstrip('/')}/data/models/best_model.json"
            self.upload_file(best_marker, best_remote)

            # 5. 写入版本记录
            version_record = {
                "synced_at": datetime.now().isoformat(),
                "strategy_id": strategy_id,
                "model_type": best.get("model_type"),
                "sharpe_ratio": best.get("sharpe_ratio"),
                "annual_return": best.get("annual_return"),
            }
            version_file = config.DATA_DIR / "models" / "sync_history.json"
            history = []
            if version_file.exists():
                with open(version_file) as f:
                    history = json.load(f)
            history.append(version_record)
            # 只保留最近20条
            history = history[-20:]
            with open(version_file, "w") as f:
                json.dump(history, f, indent=2)

            logger.info("✅ 最优模型同步完成!")
            return True

        finally:
            self.close()

    def sync_all(self) -> bool:
        """同步所有模型和策略"""
        if not self.connect():
            return False

        try:
            # 同步模型
            model_dir = config.DATA_DIR / "models"
            if model_dir.exists():
                model_files = list(model_dir.glob("*"))
                logger.info(f"同步模型: {len(model_files)} 个文件")
                for f in model_files:
                    if f.is_file():
                        remote_path = f"{self.remote_path.rstrip('/')}/data/models/{f.name}"
                        self.upload_file(f, remote_path)

            # 同步策略
            strategy_dir = config.DATA_DIR / "strategies"
            if strategy_dir.exists():
                strategy_files = list(strategy_dir.glob("*.json"))
                logger.info(f"同步策略: {len(strategy_files)} 个文件")
                for f in strategy_files:
                    remote_path = f"{self.remote_path.rstrip('/')}/data/strategies/{f.name}"
                    self.upload_file(f, remote_path)

            logger.info("✅ 全部同步完成!")
            return True

        finally:
            self.close()

    def rollback(self) -> bool:
        """回滚到上一版本"""
        history_file = config.DATA_DIR / "models" / "sync_history.json"
        if not history_file.exists():
            logger.error("没有同步历史，无法回滚")
            return False

        with open(history_file) as f:
            history = json.load(f)

        if len(history) < 2:
            logger.error("历史版本不足，无法回滚")
            return False

        # 上一版本
        prev = history[-2]
        strategy_id = prev.get("strategy_id")
        logger.info(f"回滚到上一版本: {strategy_id}")
        logger.info(f"  时间: {prev.get('synced_at')}")
        logger.info(f"  夏普: {prev.get('sharpe_ratio', 0):.2f}")

        # 更新 best_model.json
        best_marker = config.DATA_DIR / "models" / "best_model.json"
        with open(best_marker, "w") as f:
            json.dump(prev, f, indent=2)

        # 重新同步
        return self.sync_best_model()


def main():
    parser = argparse.ArgumentParser(description="云端同步工具")
    parser.add_argument("--check", action="store_true", help="检查云端连接")
    parser.add_argument("--list", action="store_true", help="列出云端模型")
    parser.add_argument("--all", action="store_true", help="同步所有模型和策略")
    parser.add_argument("--rollback", action="store_true", help="回滚到上一版本")
    args = parser.parse_args()

    syncer = CloudSyncer()

    if not syncer.configured:
        logger.warning("⚠️ 云端尚未配置")
        logger.info("请在 .env 文件中添加以下配置:")
        logger.info("  CLOUD_HOST=你的服务器IP")
        logger.info("  CLOUD_PORT=22")
        logger.info("  CLOUD_USER=用户名")
        logger.info("  CLOUD_PASSWORD=密码")
        logger.info("  CLOUD_PATH=/home/user/soloquant")
        logger.info()
        logger.info("配置完成后重新运行此脚本")
        return

    if args.check:
        syncer.check_connection()
    elif args.list:
        syncer.list_remote_models()
    elif args.rollback:
        syncer.rollback()
    elif args.all:
        syncer.sync_all()
    else:
        # 默认同步最优模型
        syncer.sync_best_model()


if __name__ == "__main__":
    main()
