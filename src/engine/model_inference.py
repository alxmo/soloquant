"""
模型推理加载器 (云端专用)
=====================================================
功能:
  - 自动扫描 models/ 目录，加载最新/激活的 LightGBM 模型
  - 支持热加载：API 上传新模型后无需重启即可切换
  - 仅加载 LightGBM 模型 (云端不安装 PyTorch)
  - 提供预测接口供 signal_generator 调用

云端专用模块，不依赖 torch / qlib 训练相关库
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.utils.config import config


class ModelInference:
    """云端模型推理加载器"""

    def __init__(self):
        self._model = None               # 当前加载的模型对象
        self._model_name: str = ""        # 当前模型文件名
        self._model_meta: dict = {}       # 模型元数据
        self._model_mtime: float = 0      # 模型文件最后修改时间
        self._last_check: float = 0       # 上次检查时间
        self._check_interval: float = 30  # 热加载检查间隔 (秒)

    @property
    def models_dir(self) -> Path:
        """模型目录"""
        return config.DATA_DIR / "models"

    @property
    def available(self) -> bool:
        """是否已有可用模型"""
        return self._model is not None

    @property
    def model_name(self) -> str:
        """当前模型名"""
        return self._model_name

    @property
    def model_meta(self) -> dict:
        """模型元数据"""
        return self._model_meta

    def _get_active_model_file(self) -> Optional[Path]:
        """
        获取激活模型文件路径
        优先读取 best_model.json 中的标记，否则使用最新的 .txt 文件
        """
        models_dir = self.models_dir
        if not models_dir.exists():
            return None

        # 1. 尝试从 best_model.json 获取
        marker = models_dir / "best_model.json"
        if marker.exists():
            try:
                with open(marker) as f:
                    data = json.load(f)
                strategy_id = data.get("strategy_id", "")
                model_type = data.get("model_type", "lightgbm")

                # 查找匹配的模型文件
                # 优先 LightGBM (.txt)
                if model_type == "lightgbm" or not model_type:
                    # 尝试精确匹配
                    for f_path in models_dir.glob("*.txt"):
                        if strategy_id and strategy_id in f_path.name:
                            return f_path
                    # 如果没有精确匹配，取最新的 .txt
                    txt_files = sorted(models_dir.glob("*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)
                    if txt_files:
                        return txt_files[0]

                # LSTM 模型 (.pth) - 云端不支持，跳过
                if model_type == "lstm":
                    logger.warning("云端不支持 LSTM 模型推理 (未安装 PyTorch)，尝试降级到 LightGBM")
                    txt_files = sorted(models_dir.glob("*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)
                    if txt_files:
                        return txt_files[0]

            except Exception as e:
                logger.warning(f"读取 best_model.json 失败: {e}")

        # 2. 降级: 取最新的 .txt 文件
        txt_files = sorted(models_dir.glob("*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if txt_files:
            return txt_files[0]

        return None

    def _check_and_reload(self):
        """检查模型是否更新，如更新则热加载"""
        now = time.time()
        if now - self._last_check < self._check_interval:
            return  # 还没到检查时间
        self._last_check = now

        model_file = self._get_active_model_file()
        if model_file is None:
            return

        mtime = model_file.stat().st_mtime

        # 模型未变化
        if model_file.name == self._model_name and mtime == self._model_mtime:
            return

        # 模型已更新，重新加载
        logger.info(f"检测到模型更新，热加载: {model_file.name}")
        self._load_model(model_file)

    def _load_model(self, model_file: Path):
        """加载 LightGBM 模型"""
        try:
            import lightgbm as lgb

            model = lgb.Booster(model_file=str(model_file))
            self._model = model
            self._model_name = model_file.name
            self._model_mtime = model_file.stat().st_mtime

            # 加载元数据
            marker = self.models_dir / "best_model.json"
            if marker.exists():
                try:
                    with open(marker) as f:
                        self._model_meta = json.load(f)
                except Exception:
                    self._model_meta = {}

            logger.info(f"✅ 模型已加载: {model_file.name} (大小: {model_file.stat().st_size / 1024:.1f} KB)")
            if self._model_meta:
                logger.info(f"   策略ID: {self._model_meta.get('strategy_id', 'N/A')}")
                logger.info(f"   模型类型: {self._model_meta.get('model_type', 'N/A')}")
                logger.info(f"   夏普比率: {self._model_meta.get('sharpe_ratio', 'N/A')}")
                logger.info(f"   年化收益: {self._model_meta.get('annual_return', 'N/A')}")

        except ImportError:
            logger.error("LightGBM 未安装，无法加载模型")
        except Exception as e:
            logger.error(f"模型加载失败 [{model_file.name}]: {e}")
            self._model = None

    def ensure_loaded(self) -> bool:
        """确保模型已加载 (首次加载或热加载检查)"""
        self._check_and_reload()
        if self._model is None:
            # 尝试首次加载
            model_file = self._get_active_model_file()
            if model_file:
                self._load_model(model_file)
        return self.available

    def predict(self, features: pd.DataFrame) -> Optional[pd.Series]:
        """
        模型预测

        Args:
            features: 特征 DataFrame (行=股票, 列=特征)

        Returns:
            预测分数 Series (index=股票代码)
        """
        if not self.ensure_loaded():
            logger.error("模型未加载，无法预测")
            return None

        try:
            pred = self._model.predict(features.values)
            scores = pd.Series(pred, index=features.index)
            logger.info(f"推理预测完成: {len(scores)} 只股票, 分数范围 [{scores.min():.4f}, {scores.max():.4f}]")
            return scores
        except Exception as e:
            logger.error(f"推理预测失败: {e}")
            return None

    def get_status(self) -> dict:
        """获取推理加载器状态"""
        return {
            "available": self.available,
            "model_name": self._model_name,
            "model_meta": self._model_meta,
            "last_check": datetime.fromtimestamp(self._last_check).isoformat() if self._last_check else None,
            "check_interval_sec": self._check_interval,
        }


# 全局实例
model_inference = ModelInference()
